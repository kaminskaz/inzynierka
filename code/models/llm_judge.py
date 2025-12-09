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
from code.technical.utils import get_field, get_model_config 

logger = logging.getLogger(__name__)


class LLMJudge(VLLM):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        param_set_number: Optional[int] = None
    ):
        super().__init__(
            model_name=model_name,
            param_set_number=param_set_number,
        )

        print("Initializing LLMJudge 2")

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
    ):
        try:
            print("Evaluating similarity...")
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
            return None, None
