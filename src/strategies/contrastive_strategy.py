from traitlets import List
import os
from typing import Optional, Dict, Any


from src.strategies.strategy_base import StrategyBase
from src.technical.content import ImageContent, TextContent
from src.technical.utils import get_field
from src.technical.response_schema import (
    DescriptionResponseSchema,
    ResponseSchema
)
from src.models.vllm import VLLM
from src.technical.configs.dataset_config import DatasetConfig


class ContrastiveStrategy(StrategyBase):
    def __init__(
        self,
        dataset_name: str,
        model: VLLM,
        dataset_config: DatasetConfig,
        results_dir: str,
        strategy_name:str,
        prompt_number: int,
        param_set_number: Optional[int]=None,
    ):
        super().__init__(dataset_name, model, dataset_config, results_dir,strategy_name, prompt_number=prompt_number, param_set_number=param_set_number)

        self.descriptions_prompt = self.get_prompt(f"describe", self.prompt_number)
        self.contrast_example_prompt = self.get_prompt(f"contrast_example", self.prompt_number)
        self.descriptions_path = os.path.join(self.results_dir, "descriptions.json")

    def _execute_problem(self, problem_id: str) -> list[Dict[str, str], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single contrastive problem.
        """

        problem_descriptions_dict = {}
        contrastive_prompt = f"{self.descriptions_prompt}\n{self.contrast_example_prompt}"

        if self.config.category in {"BP", "choice_only"}:
            collected_descriptions = []

            for i in range(self.config.num_choices):
                
                if self.config.category == "BP":
                    string_index = f"{i}"
                    if i >= self.config.num_choices // 2:
                        break
                    choice_image_input_1 = self.get_choice_image(problem_id, image_index=i)
                    choice_image_input_2 = self.get_choice_image(problem_id, image_index=i + 6)

                    contents_to_send_descriptions = [
                        TextContent(contrastive_prompt),
                        ImageContent(choice_image_input_1),
                        ImageContent(choice_image_input_2),
                    ]

                elif self.config.category == "choice_only":
                    string_index = chr(65 + i)
                    choice_image_input = self.get_blackout_image(problem_id, image_index=string_index)

                    contents_to_send_descriptions = [
                        TextContent(contrastive_prompt),
                        ImageContent(choice_image_input),
                    ]

                description_response = self.model.ask(contents_to_send_descriptions, schema=DescriptionResponseSchema)

                desc_text = get_field(description_response, 'description', None)
                if desc_text:
                    collected_descriptions.append(str(desc_text))
                    problem_descriptions_dict[string_index] = str(desc_text)

            all_descriptions_text = "\n\n".join(collected_descriptions)

            prompt = f"{self.main_prompt}\nDescriptions:\n{all_descriptions_text}\n{self.example_prompt}"
            contents_to_send = [TextContent(prompt)]

        elif self.config.category == "standard":
            question_image_input = self.get_question_image(problem_id)

            contents_to_send_descriptions = [
                TextContent(contrastive_prompt),
                ImageContent(question_image_input),
            ]

            description_response = self.model.ask(contents_to_send_descriptions, schema=DescriptionResponseSchema)

            desc_text = get_field(description_response, 'description', None)
            if desc_text:
                problem_descriptions_dict["question_image"] = desc_text
                all_descriptions_text = desc_text
            else:
                all_descriptions_text = ""

            prompt = f"{self.main_prompt}\nDescription of question image:\n{all_descriptions_text}\n{self.example_prompt}"
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

        response = self.model.ask(contents_to_send, schema=ResponseSchema)

        return response, problem_id, problem_descriptions_dict