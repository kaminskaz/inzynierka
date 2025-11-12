import os
from typing import Optional, Dict

from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import ResponseSchema

class ClassificationStrategy(StrategyBase):
    def run_single_problem(self, image_path: str, prompt: str) -> ResponseSchema:

        contents_to_send = [
            TextContent(prompt),
            ImageContent(image_path)
        ]

        response = self.model.ask_structured(contents=contents_to_send, schema=ResponseSchema)
        return response

    def _execute_problem(self, problem_id: str) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single classification problem.
        """
        image_path = self.get_classification_panel(problem_id)
        
        # Use self.main_prompt from the base class
        response = self.run_single_problem(image_path, self.main_prompt)
        
        image_name = os.path.basename(image_path) if image_path else f"{problem_id}_classification_panel.png"
        
        # Return response, image name, and None for descriptions
        return response, image_name, None