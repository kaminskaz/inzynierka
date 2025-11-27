from traitlets import List
import os
from typing import Optional, Dict, Any
import asyncio

from code.strategies.strategy_base import StrategyBase
from code.technical.content import ImageContent, TextContent
from code.technical.response_schema import (
    DescriptionResponseSchema,
    ResponseSchema,
    BPDescriptionResponseSchemaContrastive,
    BPResponseSchema
)
from code.models.vllm import VLLM
from code.preprocessing.processor_config import ProcessorConfig


class ContrastiveStrategy(StrategyBase):

    def __init__(
        self,
        dataset_name: str,
        model: VLLM,
        dataset_config: ProcessorConfig,
        results_dir: str,
        strategy_name:str,
    ):
        super().__init__(dataset_name, model, dataset_config, results_dir,strategy_name)

        self.descriptions_prompt = self.get_prompt("describe_main")
        self.descriptions_path = os.path.join(self.results_dir, "descriptions.json")

    def run_single_problem(
        self, problem_id: str, descriptions_prompt: str, main_prompt: str
    ) -> list[Optional[ResponseSchema], Dict[str, str]]:

        problem_descriptions_dict = {}

        if self.config.category == "BP" or self.config.category == "choice_only":
            collected_descriptions = []
            
            # FIX: Define response_schema early to avoid UnboundLocalError
            if self.config.category == "BP":
                response_schema = BPResponseSchema
            else:
                response_schema = ResponseSchema

            for i in range(self.config.num_choices):
                if self.config.category == "BP":
                    if i >= self.config.num_choices // 2:
                        break
                    choice_image_input_1 = self.get_choice_image(problem_id, image_index=i)
                    choice_image_input_2 = self.get_choice_image(
                        problem_id, image_index=i + 6
                    )
                    description_example_bp = self.get_prompt("contrast_example_main")
                    contents_to_send_descriptions = [
                        TextContent(f"{descriptions_prompt}\n{description_example_bp}"),
                        ImageContent(choice_image_input_1),
                        ImageContent(choice_image_input_2),
                    ]
                    description_response = asyncio.run(self.model.ask_structured(
                        contents_to_send_descriptions,
                        schema=BPDescriptionResponseSchemaContrastive,
                    ))

                    # FIX: Handle Left/Right split in schema
                    if description_response:
                        desc_left = getattr(description_response, 'description_left', '')
                        desc_right = getattr(description_response, 'description_right', '')
                        combined_desc = f"Left: {desc_left} | Right: {desc_right}"
                        
                        # Store string directly, do not attempt to patch the Pydantic object
                        collected_descriptions.append(combined_desc)
                        
                        key = f"{i}"
                        problem_descriptions_dict[key] = combined_desc

                else:  # 'choice_only'
                    letter_index = chr(65 + i)
                    choice_image_input = self.get_blackout_image(
                        problem_id, image_index=letter_index
                    )
                    description_example = self.get_prompt("contrast_example_main")
                    contents_to_send_descriptions = [
                        TextContent(f"{descriptions_prompt}\n{description_example}"),
                        ImageContent(choice_image_input),
                    ]
                    description_response = asyncio.run(self.model.ask_structured(
                        contents_to_send_descriptions, schema=DescriptionResponseSchema
                    ))

                    desc_text = getattr(description_response, 'description', None)
                    if description_response and desc_text:
                        collected_descriptions.append(desc_text)
                        problem_descriptions_dict[letter_index] = desc_text

            all_descriptions_text = "\n\n".join(collected_descriptions)

            prompt = f"{main_prompt}\nDescriptions:\n{all_descriptions_text}\n{self.example_prompt}"
            contents_to_send = [TextContent(prompt)]

        else:  # 'standard' category
            response_schema = ResponseSchema
            question_image_input = self.get_question_image(problem_id)

            contents_to_send_descriptions = [
                TextContent(descriptions_prompt),
                ImageContent(question_image_input),
            ]

            description_response = asyncio.run(self.model.ask_structured(
                contents_to_send_descriptions, schema=DescriptionResponseSchema
            ))

            desc_text = getattr(description_response, 'description', None)
            if description_response and desc_text:
                problem_descriptions_dict["question_panel"] = desc_text
                all_descriptions_text = desc_text
            else:
                all_descriptions_text = ""

            prompt = f"{main_prompt}\nDescriptions:\n{all_descriptions_text}\n{self.example_prompt}"
            choice_panel_input = self.get_choice_panel(problem_id)

            if choice_panel_input is None:
                self.logger.error(
                    f"Could not get choice panel for problem {problem_id}. Skipping image content."
                )
                contents_to_send = [TextContent(prompt)]
            else:
                contents_to_send = [
                    TextContent(prompt),
                    ImageContent(choice_panel_input),
                ]

        response = asyncio.run(self.model.ask_structured(
            contents_to_send, schema=response_schema
        ))

        return response, problem_descriptions_dict

    def _execute_problem(
        self, problem_id: str
    ) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single contrastive problem.
        """

        response, problem_descriptions = self.run_single_problem(
            problem_id, self.descriptions_prompt, self.main_prompt
        )

        return response, problem_id, problem_descriptions

    def _get_metadata_prompts(self) -> Dict[str, Optional[str]]:
        prompts = super()._get_metadata_prompts()
        prompts["describe_prompt"] = self.descriptions_prompt
        return prompts