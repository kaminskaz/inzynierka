import subprocess
import time
from typing import List, Optional, Dict, Type, Any
import portpicker
import requests
import logging
import sys
import os
from pydantic import BaseModel

from code.technical.content import Content, ImageContent, TextContent
from code.technical.prompt_formatter import PromptFormatter
from code.models.vllm import VLLM
from code.technical.utils import _parse_response, _get_field

logger = logging.getLogger(__name__)


class LLMJudge(VLLM):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_output_tokens: int = 512,
        chat_template_path: str = "mistral_template.jinja",
        **kwargs,
    ):
        # forcing text-only evaluation
        limit_mm_per_prompt = 0

        here = os.path.dirname(os.path.abspath(__file__))
        chat_template_path = os.path.join(here, "technical", "chat_templates", chat_template_path)

        custom_args = kwargs.get("custom_args", [])
        custom_args += [
            "--chat-template",
            chat_template_path
        ]

        print("LLMJudge is starting with chat template:", chat_template_path)
        print("Custom args:", custom_args)

        if not os.path.exists(chat_template_path):
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            limit_mm_per_prompt=limit_mm_per_prompt,
            custom_args=custom_args
        )

        self.judge_mode = "text_only"
        logger.info(
            f"Initialized LLMJudge for text-only evaluation with model {model_name}"
        )

    def evaluate_similarity(
        self, 
        prompt: str, 
        answer: str, 
        key: str, 
        response_schema: Optional[Type[BaseModel]]
    ) -> float:
        try:
            prompt = (
                f"{prompt}\n"
                f"Answer: {answer}\n"
                f"Key Answer: {key}\n"
            )

            if response_schema:
                response = self.ask(
                    [TextContent(prompt)], response_schema
                )
            else:
                response = self.ask([TextContent(prompt)])

            response = _parse_response(response)
            similarity_label = _get_field(response, "similarity_label", "No similarity label provided.").strip()
            reasoning = _get_field(response, "reasoning", "No reasoning provided.").strip

            # if response.similarity_label is None:
            #     logger.info("Received None similarity_label from LLM.")
            #     similarity_label = "No similarity label provided."
            # else:
            #     similarity_label = response.similarity_label
            # if response.reasoning is None:
            #     logger.info("Received None reasoning from LLM.")
            #     reasoning = "No reasoning provided."
            # else:
            #     reasoning = response.reasoning

            # if isinstance(similarity_label, str):
            #     similarity_label = similarity_label.strip()
            #     reasoning = reasoning.strip()
            return similarity_label, reasoning

        except Exception as e:
            logger.error(f"Similarity evaluation failed: {e}")
            return "None"
