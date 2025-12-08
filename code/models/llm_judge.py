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
from code.technical.utils import get_field 

logger = logging.getLogger(__name__)


class LLMJudge(VLLM):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_output_tokens: int = 512,
        chat_template_path: str = "mistral_template.jinja",
        **kwargs,
    ):
        # forcing text-only evaluation
        limit_mm_per_prompt = 0

        here = os.path.dirname(os.path.abspath(__file__))
        code_root = os.path.abspath(os.path.join(here, ".."))
        chat_template_path = os.path.join(code_root, "technical", "chat_templates", "mistral_template.jinja")

        custom_args = kwargs.get("custom_args", [])
        custom_args += [
            "--chat-template",
            chat_template_path
        ]

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

            similarity_label = get_field(response, "similarity_label", "No similarity label provided.")
            reasoning = get_field(response, "reasoning", "No reasoning provided.")

            return similarity_label, reasoning

        except Exception as e:
            logger.error(f"Similarity evaluation failed: {e}")
            return "None"
