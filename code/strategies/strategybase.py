import logging
from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional, Dict
import os
import csv
import json

from code.preprocessing.processorconfig import ProcessorConfig
from code.models.vllm import VLLM
from code.technical.response_schema import ResponseSchema


class StrategyBase(ABC):
    def __init__(
        self,
        dataset_name: str,
        model: VLLM,
        dataset_config: ProcessorConfig,
        results_dir: str,
        strategy_name: str,
    ):
        self.dataset_name: str = dataset_name
        self.model: VLLM = model
        self.config: ProcessorConfig = dataset_config
        self.strategy_name: str = strategy_name

        self.logger = logging.getLogger(self.__class__.__name__)
        self.results_dir = results_dir
        self.data_dir = "data_test"
        self.dataset_dir = os.path.join(self.data_dir, self.dataset_name, "problems")
        self.problem_description_prompt = self.get_prompt("problem_description_main")
        self.question_prompt = self.get_prompt("question_main")
        self.main_prompt = f"{self.problem_description_prompt}\n{self.question_prompt}"

        # path for descriptions, to be set by subclasses if needed - for contrastive and descriptive
        self.descriptions_path: Optional[str] = None

        self.logger.info(f"Initialized strategy for dataset: '{self.dataset_name}'")

    @abstractmethod
    def run_single_problem(self, *args, **kwargs) -> Any:
        """
        This method's signature will vary by strategy, so it remains abstract
        but will be called by the subclass's _execute_problem, not by the base run().
        """
        pass

    @abstractmethod
    def _execute_problem(self, problem_id: str) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:  # type: ignore
        """
        The core logic for processing a single problem.

        Args:
            problem_id (str): The ID of the problem to process.

        Calls:
            run_single_problem(problem_id)

        Returns:
            A tuple containing:
            - Optional[ResponseSchema]: The model's response.
            - str: The name of the image file to save in the results.
            - Optional[Dict[str, str]]: A dictionary of descriptions, if any.
        """
        pass

    def _get_metadata_prompts(self) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary of prompts to be saved in the metadata file.
        Subclasses can override this to add more prompts.
        """
        return {
            "question_prompt": self.question_prompt,
            "problem_description_prompt": self.problem_description_prompt,
            "describe_prompt": None,
        }

    def run(self) -> None:
        """
        Main execution loop (Template Method).
        Common to all strategies.
        """
        results = []
        all_descriptions_data = {}

        for problem_entry in os.scandir(self.dataset_dir):
            problem_id = problem_entry.name
            try:
                if not problem_entry.is_dir():
                    continue
                problem_id = problem_entry.name

                response, problem_id, problem_descriptions = (
                    self._execute_problem(problem_id)
                )

                if problem_descriptions:
                    all_descriptions_data[problem_id] = problem_descriptions

                if response:
                    result = {
                        "problem_id": problem_id,
                        "answer": response.answer,
                        "confidence": response.confidence,
                        "rationale": response.rationale,
                    }
                else:
                    result = {
                        "problem_id": problem_id,
                        "answer": "",
                        "confidence": "",
                        "rationale": "",
                    }
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {problem_id}: {e}")

        self.save_raw_answers_to_csv(results)

        metadata_prompts = self._get_metadata_prompts()
        self.save_metadata(
            question_prompt=metadata_prompts["question_prompt"],
            problem_description_prompt=metadata_prompts["problem_description_prompt"],
            describe_prompt=metadata_prompts.get(
                "describe_prompt"
            ),  # used .get() for safety
        )

        if all_descriptions_data and self.descriptions_path:
            self.save_descriptions_to_json(
                self.descriptions_path, all_descriptions_data
            )

        self.logger.info(
            f"{self.strategy_name} run completed for dataset: {self.dataset_name} "
            f"using model: {self.model.get_model_name()}"  # Corrected to use get_model_name()
        )

    def get_prompt(self, prompt_type: str) -> str:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))

            repo_root = os.path.dirname(os.path.dirname(current_dir))

            if prompt_type == "problem_description_main":
                prompt_path = os.path.join(
                    repo_root, "prompts", self.dataset_name, f"{prompt_type}.txt"
                )
            else:
                prompt_path = os.path.join(
                    repo_root,
                    "prompts",
                    self.dataset_name,
                    self.strategy_name,
                    f"{prompt_type}.txt",
                )

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            return prompt

        except Exception as e:
            self.logger.exception(
                f"Error reading prompt for strategy '{self.strategy_name}' and dataset '{self.dataset_name}': {e}"
            )
            return ""

    def save_metadata(
        self,
        question_prompt: str,
        problem_description_prompt: str,
        describe_prompt: Optional[str] = None,
    ) -> None:
        """Save dataset, strategy, model, and config info into a metadata file."""

        metadata = {
            "dataset": self.dataset_name,
            "strategy": self.strategy_name,
            "model": self.model.get_model_name(),
            "config": self.config,
            "problem_description_prompt": problem_description_prompt,
            "question_prompt": question_prompt,
            "describe_prompt": describe_prompt,
        }
        try:
            metadata_path = os.path.join(self.results_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            self.logger.exception(f"Failed to save metadata: {e}")

    def save_raw_answers_to_csv(self, results: List[dict]) -> None:
        if not results:
            self.logger.warning("No results to save.")
            return

        output_path = os.path.join(self.results_dir, "results.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fieldnames = list(results[0].keys())

        write_header = not os.path.exists(output_path)
        with open(output_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"Saved {len(results)} results to {output_path}")

    # FUNCTIONS FOR GETTING IMAGES AND PANELS

    def get_choice_panel(self, problem_id: str) -> Optional[str]:
        # applicable only to "standard" datasets
        if hasattr(self.config, "category") and self.config.category != "standard":
            self.logger.warning(
                f"get_choice_panel called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return None

        image_path = f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/choice_panel.png"
        return image_path

    def get_choice_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        # image index is either string A-D, A-H or integer 0-11 (for BPs)
        # applicable to all datasets
        if not self.verify_choice_index(image_index):
            return ""

        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/choices/{image_index}.png"
        )
        return image_path

    def get_question_panel(self, problem_id: str) -> str:
        # applicable to all datasets
        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/question_panel.png"
        )
        return image_path

    def get_question_image(self, problem_id: str) -> str:
        # applicable only to "standard" datasets
        if (
            hasattr(self, "config")
            and hasattr(self.config, "category")
            and self.config.category != "standard"
        ):
            self.logger.warning(
                f"get_question_image called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return ""

        image_path = f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/question.png"
        return image_path

    def get_blackout_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        # applicable only to "choice_only" datasets
        if hasattr(self.config, "category") and self.config.category != "choice_only":
            self.logger.warning(
                f"get_blackout_image called for non-choice_only dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return ""

        if not self.verify_choice_index(image_index):
            return ""

        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/blackout/{image_index}.png"
        )
        return image_path

    def get_classification_panel(self, problem_id: str) -> str:
        # applicable to all datasets
        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/classification_panel.png"
        )
        return image_path

    def get_list_of_choice_images(
        self, problem_id: str, image_indices: List[Union[str, int]]
    ) -> List[str]:
        # applicable to all datasets, used for BPs
        # verify if all indices are the same format
        if not all(isinstance(index, (str, int)) for index in image_indices):
            self.logger.error(
                f"Image indices list contained mixed types: {image_indices}. "
                f"All indices must be str or int. Returning empty list."
            )
            return []

        images = []
        for index in image_indices:
            images.append(self.get_choice_image(problem_id, index))
        return images

    def verify_choice_index(self, image_index: Union[str, int]) -> bool:
        if not hasattr(self.config, "category"):
            self.logger.error(
                "Config object has no 'category' attribute. Cannot verify index."
            )
            return False

        try:
            if (
                self.config.category == "standard"
                or self.config.category == "choice_only"
            ):
                valid_indices = [
                    chr(i) for i in range(ord("A"), ord("A") + self.config.num_choices)
                ]
                # Check type consistency
                if not isinstance(image_index, str):
                    self.logger.warning(
                        f"Index '{image_index}' is not a string, but category is '{self.config.category}'."
                    )
                    return False
            elif self.config.category == "BP":
                valid_indices = [i for i in range(self.config.num_choices)]
                # Check type consistency
                if not isinstance(image_index, int):
                    self.logger.warning(
                        f"Index '{image_index}' is not an int, but category is 'BP'."
                    )
                    return False
            else:
                self.logger.error(f"Unknown dataset category '{self.config.category}'.")
                return False

            if image_index not in valid_indices:
                self.logger.warning(
                    f"image_index '{image_index}' is not in the list of valid indices "
                    f"for dataset category '{self.config.category}'. Valid: {valid_indices}"
                )
                return False

        except AttributeError as e:
            self.logger.exception(f"Config is missing an attribute: {e}")
            return False

        return True

    def save_descriptions_to_json(
        self, descriptions_path: str, all_descriptions_data: dict
    ):
        """
        Saves the collected descriptions dictionary to a JSON file.

        Args:
            descriptions_path (str): The full file path to save the JSON to.
            all_descriptions_data (dict): The dictionary containing problem IDs
                                          and their corresponding descriptions.
        """
        try:
            # Ensure the directory exists
            directory = os.path.dirname(descriptions_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            # Write the data to the JSON file
            with open(descriptions_path, "w", encoding="utf-8") as f:
                json.dump(all_descriptions_data, f, indent=4)

            self.logger.info(f"Saved descriptions to {descriptions_path}")

        except (IOError, OSError) as e:
            self.logger.error(
                f"Failed to create directory or write to file {descriptions_path}: {e}"
            )
        except TypeError as e:
            self.logger.error(f"Error serializing descriptions to JSON: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred in save_descriptions_to_json: {e}"
            )
