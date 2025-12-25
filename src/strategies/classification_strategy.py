import os
from typing import Optional, Dict


from src.strategies.strategy_base import StrategyBase
from src.technical.content import ImageContent, TextContent
from src.technical.response_schema import ResponseSchema


class ClassificationStrategy(StrategyBase):

    def _execute_problem(self, problem_id: str) -> list[Dict[str, str], str, Optional[Dict[str, str]]]:

        image_path = self.get_classification_panel(problem_id)
        prompt = f"{self.main_prompt}\n{self.example_prompt}"

        contents_to_send = [TextContent(prompt), ImageContent(image_path)]
       
        response = self.model.ask(contents=contents_to_send, schema=ResponseSchema)

        return response, problem_id, None # None for descriptions
