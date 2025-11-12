from traitlets import List
import os
from typing import Optional, Dict, Any

from code.strategies.strategybase import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema, BPDescriptionResponseSchemaContrastive
from code.models.vllm import VLLM
from code.preprocessing.processorconfig import ProcessorConfig


class ContrastiveStrategy(StrategyBase):

    def __init__(self, dataset_name: str, model: VLLM, dataset_config: ProcessorConfig, results_dir: str):
        super().__init__(dataset_name, model, dataset_config, results_dir)
        
        self.descriptions_prompt = self.get_prompt("describe_main")
        self.descriptions_path = os.path.join(self.results_dir, "descriptions.json")

    def run_single_problem(self, problem_id: str, descriptions_prompt: str, main_prompt: str) -> list[Optional[ResponseSchema], Dict[str, str]]:
        
        problem_descriptions_dict = {}

        if self.config.category == 'BP' or self.config.category == 'choice_only': 
            descriptions = []
            
            for i in range(self.config.num_choices):
                if self.config.category == 'BP':
                    if i >= self.config.num_choices // 2:
                        break
                    choice_image_input_1 = self.get_choice_image(problem_id, index=i)
                    choice_image_input_2 = self.get_choice_image(problem_id, index=i+6)
                    contents_to_send_descriptions = [
                        TextContent(descriptions_prompt),
                        ImageContent(choice_image_input_1),
                        ImageContent(choice_image_input_2)
                    ]
                    description_response = self.model.ask_structured(contents_to_send_descriptions, schema=BPDescriptionResponseSchemaContrastive)

                    if description_response and description_response.description:
                        key = f"{i}"
                        problem_descriptions_dict[key] = description_response.description

                else: # 'choice_only'
                    letter_index = chr(65 + i) # Assuming choice_only uses letters
                    choice_image_input = self.get_blackout_image(problem_id, index=letter_index)
                    contents_to_send_descriptions = [
                        TextContent(descriptions_prompt),
                        ImageContent(choice_image_input)
                    ]
                    description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema)
         
                    if description_response and description_response.description:
                        problem_descriptions_dict[letter_index] = description_response.description
 
                descriptions.append(description_response)
        
            all_descriptions_text = "\n\n".join([r.description for r in descriptions if r is not None and r.description is not None])

            prompt = f'{main_prompt}\nDescriptions:\n{all_descriptions_text}'
            
            contents_to_send = [
                    TextContent(prompt)
            ]

        else: # 'standard' category
            question_panel_input = self.get_question_panel(problem_id)

            contents_to_send_descriptions = [
                    TextContent(descriptions_prompt),
                    ImageContent(question_panel_input)
            ]

            description_response = self.model.ask_structured(contents_to_send_descriptions, schema=DescriptionResponseSchema) 
            
            if description_response and description_response.description:
                problem_descriptions_dict["question_panel"] = description_response.description
                all_descriptions_text = description_response.description
            else:
                all_descriptions_text = ""

            prompt = f'{main_prompt}\nDescriptions:\n{all_descriptions_text}'
            choice_panel_input = self.get_choice_panel(problem_id)

            if choice_panel_input is None:
                self.logger.error(f"Could not get choice panel for problem {problem_id}. Skipping image content.")
                contents_to_send = [TextContent(prompt)]
            else:
                contents_to_send = [
                        TextContent(prompt),
                        ImageContent(choice_panel_input)
                ]

        response = self.model.ask_structured(contents_to_send, schema=ResponseSchema)
        
        return response, problem_descriptions_dict
    

    def _execute_problem(self, problem_id: str) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single contrastive problem.
        """
        image_path = self.get_choice_panel(problem_id)
        
        response, problem_descriptions = self.run_single_problem(
            problem_id, self.descriptions_prompt, self.main_prompt
        )
        
        # handle potential None from get_choice_panel (e.g., for 'BP' category)
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