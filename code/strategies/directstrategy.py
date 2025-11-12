import os
from typing import Optional, Dict

from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import ResponseSchema

class DirectStrategy(StrategyBase):

    def run_single_problem(self, image_path: str, prompt: str) -> ResponseSchema:
        contents = [
            TextContent(prompt),
            ImageContent(image_path)
        ]
        response = self.model.ask_structured(contents, schema=ResponseSchema)

        return response

    def _execute_problem(self, problem_id: str) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single direct problem.
        """
        image_path = self.get_question_image(problem_id)
        
        # Use self.main_prompt from the base class
        response = self.run_single_problem(image_path, self.main_prompt)
        
        image_name = os.path.basename(image_path) if image_path else f"{problem_id}_question.png"
        
        # Return response, image name, and None for descriptions
        return response, image_name, None