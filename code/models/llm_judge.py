import subprocess
import time
from typing import List, Optional, Dict, Type, Any
import portpicker
import requests
import logging
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity

from code.technical.content import Content, ImageContent, TextContent
from code.technical.prompt_formatter import PromptFormatter
from code.models.vllm import VLLM

logger = logging.getLogger(__name__)


class LLMJudge(VLLM):
    def __init__(
        self,
        model_name: str = "intfloat/e5-small",
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
        logger.info(f"Initialized LLMJudge for text-only evaluation with model {model_name}")

    async def evaluate_similarity(self, text_a: str, text_b: str) -> float:
        try:
            embeddings = self.client.embeddings.create(
                model=self.model_name,
                input=[text_a, text_b]
            )
            sim = cosine_similarity(
                [embeddings.data[0].embedding],
                [embeddings.data[1].embedding]
            )[0][0]
            return float(sim)
        except Exception as e:
            logger.error(f"Similarity evaluation failed: {e}")
            return 0.0
