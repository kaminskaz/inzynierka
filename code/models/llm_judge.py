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

logger = logging.getLogger(__name__)


class LLMJudge(VLLM):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.3",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_output_tokens: int = 512,
        **kwargs,
    ):
        # turn off multi-modal capabilities for judge
        limit_mm_per_prompt = 0

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            limit_mm_per_prompt=limit_mm_per_prompt,
            custom_args=kwargs.get("custom_args", []),
        )

        self.judge_mode = "text_only"
        logger.info(
            f"Initialized LLMJudge for text-only evaluation with model {model_name}"
        )

    async def evaluate_similarity(
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
                response = await self.ask(
                    [TextContent(prompt)], response_schema
                )
                similarity_label = response.similarity_label
                reasoning = response.reasoning
                if isinstance(similarity_label, str):
                    similarity_label = similarity_label.strip()
                    reasoning = reasoning.strip()
                return similarity_label, reasoning
            else:
                response = await self.ask([TextContent(prompt)])
                similarity_label = response[0].text.strip()
                return similarity_label

        except Exception as e:
            logger.error(f"Similarity evaluation failed: {e}")
            return "None"
