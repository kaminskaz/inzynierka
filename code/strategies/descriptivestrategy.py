from traitlets import List
import os
from typing import Optional, Dict, Any

from code.technical.content import ImageContent, TextContent
from code.strategies.strategybase import StrategyBase
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema
from code.models.vllm import VLLM
from code.preprocessing.processorconfig import ProcessorConfig

class DescriptiveStrategy(StrategyBase):

    def __init__(self, dataset_name: str, model: VLLM, dataset_config: ProcessorConfig, results_dir: str):
        super().__init__(dataset_name, model, dataset_config, results_dir)
        
        self.descriptions_prompt = self.get_prompt("describe_main")
        self.descriptions_path = os.path.join(self.results_dir, "descriptions.json")

    async def run_single_problem(self, problem_id: str, descriptions_prompt: str, main_prompt: str) -> list[Optional[ResponseSchema], Dict[str, str]]:
        descriptions = []
        problem_descriptions_dict = {}

        for i in range(self.config.num_choices):
            letter=chr(65 + i)

            if self.config.category != 'BP':
                choice_image_input = self.get_choice_image(problem_id, index=letter)
                index_key = letter
            else:
                choice_image_input = self.get_choice_image(problem_id, index=i)
                index_key = i

            contents_to_send_descriptions = [
                TextContent(descriptions_prompt),
                ImageContent(choice_image_input)
            ]
            description_response = await self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            raw_description = description_response.description
            if raw_description is not None and description_response.description is not None:
                description_response.description = f"{index_key}: {raw_description}"
                problem_descriptions_dict[index_key]=raw_description

            descriptions.append(description_response)

        all_descriptions_text = "\n\n".join([r.description for r in descriptions if r is not None and r.description is not None])

        prompt = f'{main_prompt}\nDescriptions:\n{all_descriptions_text}'

        if self.config.category == 'BP' or self.config.category == 'choice_only':
            contents_to_send = [
                    TextContent(prompt)
            ]
        else:
            image_input = self.get_choice_panel(problem_id)
            if image_input is None:
                self.logger.error(f"Could not get choice panel for problem {problem_id}. Skipping image content.")
                contents_to_send = [TextContent(prompt)]
            else:
                contents_to_send = [
                        TextContent(prompt),
                        ImageContent(image_input)
                ]
            
        response = await self.model.ask_structured(contents_to_send, schema=ResponseSchema)
    
        return response, problem_descriptions_dict
    
    async def _execute_problem(self, problem_id: str) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single descriptive problem.
        """
        image_path = self.get_choice_panel(problem_id)
        
        response, problem_descriptions = self.run_single_problem(
            problem_id, self.descriptions_prompt, self.main_prompt
        )
        
        if image_path:
            image_name = os.path.basename(image_path)
        else:
            image_name = "placeholder_name"
        
        return response, image_name, problem_descriptions

    def _get_metadata_prompts(self) -> Dict[str, Optional[str]]:
        """
        Overrides the base method to add the 'describe_prompt'.
        """
        prompts = super()._get_metadata_prompts()
        prompts["describe_prompt"] = self.descriptions_prompt
        return prompts