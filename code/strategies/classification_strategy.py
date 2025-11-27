import os
from typing import Optional, Dict


from code.strategies.strategy_base import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import ResponseSchema, BPResponseSchema


class ClassificationStrategy(StrategyBase):
    def run_single_problem(self, image_path: str, prompt: str) -> ResponseSchema:

        contents_to_send = [TextContent(prompt), ImageContent(image_path)]

        if self.config.category == "BP":
            response_schema = BPResponseSchema
        else:
            response_schema = ResponseSchema

        response = self.model.ask(
            contents=contents_to_send, schema=response_schema
        )
        return response

    def _execute_problem(
        self, problem_id: str
    ) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        image_path = self.get_classification_panel(problem_id)

        prompt_with_example = f"{self.main_prompt}\n{self.example_prompt}"
        # Use self.main_prompt from the base class
        response = self.run_single_problem(image_path, prompt_with_example)

        # Return response, image name, and None for descriptions
        return response, problem_id, None
