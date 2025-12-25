import os
from typing import Optional, Dict


from src.strategies.strategy_base import StrategyBase
from src.technical.content import ImageContent, TextContent
from src.technical.response_schema import ResponseSchema


class DirectStrategy(StrategyBase):

    def _execute_problem(self, problem_id: str) -> list[Dict[str, str], str, Optional[Dict[str, str]]]:
        
        image_path = self.get_question_panel(problem_id)
        prompt = f"{self.main_prompt}\n{self.example_prompt}"

        contents = [TextContent(prompt), ImageContent(image_path)]

        response = self.model.ask(contents, schema=ResponseSchema)

        return response, problem_id, None # None for descriptions
