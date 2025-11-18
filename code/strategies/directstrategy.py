import os
from typing import Optional, Dict
import asyncio

from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import ResponseSchema, BPResponseSchema


class DirectStrategy(StrategyBase):

    def run_single_problem(self, image_path: str, prompt: str) -> ResponseSchema:
        contents = [TextContent(prompt), ImageContent(image_path)]

        if self.config.category == "BP":
            response_schema = BPResponseSchema
        else:
            response_schema = ResponseSchema

        response = asyncio.run(self.model.ask_structured(contents, schema=response_schema))

        return response

    def _execute_problem(
        self, problem_id: str
    ) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        image_path = self.get_question_panel(problem_id)

        # Use self.main_prompt from the base class
        response = self.run_single_problem(image_path, self.main_prompt)

        # Return response, image name, and None for descriptions
        return response, problem_id, None
