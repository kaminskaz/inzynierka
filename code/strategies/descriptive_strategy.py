from traitlets import List
import os
from typing import Optional, Dict, Any
from PIL import Image


from code.technical.content import ImageContent, TextContent
from code.strategies.strategy_base import StrategyBase
from code.technical.response_schema import DescriptionResponseSchema, ResponseSchema
from code.models.vllm import VLLM
from code.technical.configs.dataset_config import DatasetConfig
from code.technical.utils import get_field

class DescriptiveStrategy(StrategyBase):

    def __init__(
        self,
        dataset_name: str,
        model: VLLM,
        dataset_config: DatasetConfig,
        results_dir: str,
        strategy_name: str,
    ):
        super().__init__(dataset_name, model, dataset_config, results_dir, strategy_name)

        self.descriptions_prompt = self.get_prompt("describe_main")
        self.descriptions_path = os.path.join(self.results_dir, "descriptions.json")

    def run_single_problem(
        self, problem_id: str, descriptions_prompt: str, main_prompt: str
    ) -> list[Optional[ResponseSchema], Dict[str, str]]:
        descriptions = []
        problem_descriptions_dict = {}

        for i in range(self.config.num_choices):
            letter = chr(65 + i)

            if self.config.category != "BP":
                choice_image_input = self.get_choice_image(problem_id, image_index=letter)
                index_key = letter
            else:
                choice_image_input = self.get_choice_image(problem_id, image_index=i)
                index_key = i

            describe_example_prompt = self.get_prompt("describe_example_main")
            contents_to_send_descriptions = [
                TextContent(f"{descriptions_prompt}\n{describe_example_prompt}"),
                ImageContent(choice_image_input),
            ]

            description_response = self.model.ask(
                contents_to_send_descriptions, schema=DescriptionResponseSchema
            )

            raw_description = get_field(description_response, 'description', None)
            self.logger.debug(f"Description for choice {index_key}: {raw_description}")

            if raw_description:
                problem_descriptions_dict[index_key] = raw_description

            descriptions.append(f"{index_key}: {raw_description}" if raw_description else None)

        all_descriptions_text = "\n\n".join(
            [
                desc
                for desc in descriptions
                if desc is not None
            ]
        )

        prompt = f"{main_prompt}\nDescriptions:\n{all_descriptions_text}\n{self.example_prompt}"

        if self.config.category == "BP" or self.config.category == "choice_only":
            contents_to_send = [TextContent(prompt)]
        else:
            image_input = self.get_choice_panel(problem_id)
            prompt = f"{prompt}\n{self.example_prompt}"
            if image_input is None:
                self.logger.error(
                    f"Could not get choice panel for problem {problem_id}. Skipping image content."
                )
                contents_to_send = [TextContent(prompt)]
            else:
                contents_to_send = [TextContent(prompt), ImageContent(image_input)]

        response_schema = ResponseSchema
            
        response = self.model.ask(
            contents_to_send, schema=response_schema
        )

        return response, problem_descriptions_dict

    def _execute_problem(
        self, problem_id: str
    ) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:
        """
        Executes the logic for a single descriptive problem.
        """

        response, problem_descriptions = self.run_single_problem(
            problem_id, self.descriptions_prompt, self.main_prompt
        )

        return response, problem_id, problem_descriptions