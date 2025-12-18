import logging
from abc import ABC, abstractmethod
import sys
from typing import Any, List, Union, Optional, Dict
import os
import csv
import json
from dataclasses import asdict, is_dataclass
import time

from src.technical.configs.dataset_config import DatasetConfig
from src.models.vllm import VLLM
from src.technical.response_schema import ResponseSchema
from src.technical.utils import get_field


class StrategyBase(ABC):
    def __init__(
        self,
        dataset_name: str,
        model: VLLM,
        dataset_config: DatasetConfig,
        results_dir: str,
        strategy_name: str,
        param_set_number: Optional[int] = None,
        prompt_number: Optional[int] = 1
    ):
        self.dataset_name: str = dataset_name
        self.model: VLLM = model
        self.config: DatasetConfig = dataset_config
        self.strategy_name: str = strategy_name
        self.param_set_number: Optional[int] = param_set_number
        self.prompt_number = prompt_number

        self.logger = logging.getLogger(self.__class__.__name__)
        self.results_dir = results_dir
        self.data_dir = "data"
        self.dataset_dir = os.path.join(self.data_dir, self.dataset_name, "problems")
        self.problem_description_prompt = self.get_prompt(f"problem_description_{self.prompt_number}")
        self.sample_answer_prompt = self.get_prompt(f"sample_answers_{self.prompt_number}")
        self.question_prompt = self.get_prompt(f"question_{self.prompt_number}")
        self.main_prompt = f"{self.problem_description_prompt}\n{self.question_prompt}"
        self.descriptions_prompt = None
        self.example_prompt = self.get_prompt(f"example_{prompt_number}")

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
        """
        pass

    def _load_existing_results(self, csv_path: str) -> List[dict]:
        """Helper to read existing CSV results into a list of dicts."""
        existing_data = []
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
            self.logger.info(f"Loaded {len(existing_data)} existing results from {csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to load existing results from {csv_path}: {e}")
        return existing_data

    def run(self, restart_problem_id: str = "") -> None:
        """
        Main execution loop (Template Method).
        """
        results = []
        all_descriptions_data = {}

        self.save_metadata(
            question_prompt=self.question_prompt,
            problem_description_prompt=self.problem_description_prompt,
            describe_prompt=self.descriptions_prompt,
            sample_answer_prompt=self.sample_answer_prompt,
        )

        output_path = os.path.join(self.results_dir, "results.csv")
        if os.path.exists(output_path):
            raw_results = self._load_existing_results(output_path)
            # Keeps only the last entry for any problem_id
            unique_results_map = {r['problem_id']: r for r in raw_results}
            results = list(unique_results_map.values())
            
            self.logger.info(f"Loaded {len(results)} unique results from existing CSV.")

            if len(results) >= self.config.expected_num_samples:
                self.logger.info(
                    f"Dataset {self.dataset_name} is already fully processed "
                    f"({len(results)}/{self.config.expected_num_samples}). Exiting pipeline."
                )
                self.model.stop()
                sys.exit(2) 

        entries = [e for e in os.scandir(self.dataset_dir) if e.is_dir()]
        entries.sort(key=lambda entry: entry.name)
        
        processed_ids = {r['problem_id'] for r in results}

        if restart_problem_id:
            self.logger.info(f"Restarting from problem ID: {restart_problem_id}")
            entries = [e for e in entries if e.name >= restart_problem_id]
            results = [r for r in results if r['problem_id'] < restart_problem_id]
        elif processed_ids:
            last_id = max(processed_ids)
            entries = [e for e in entries if e.name > last_id]
            self.logger.info(f"Auto-resuming from entry after {last_id}")

        for problem_entry in entries:
            try:
                problem_id = problem_entry.name
                response, _, problem_descriptions = self._execute_problem(problem_id)

                if problem_descriptions and self.descriptions_path:
                    all_descriptions_data[problem_id] = problem_descriptions
                    self.save_descriptions_to_json(self.descriptions_path, all_descriptions_data)

                problem_id_str = str(problem_id).zfill(3)

                result = {
                    "problem_id": problem_id_str,
                    "answer": get_field(response, "answer", "") if response else "",
                    "confidence": get_field(response, "confidence", "") if response else "",
                    "rationale": get_field(response, "rationale", "") if response else "",
                }

                results.append(result)
                self.save_raw_answers_to_csv(results)
                time.sleep(0.3)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error processing {problem_entry.name}: {error_msg}")
                
                fatal_errors = ["Request timed out"]
                
                if any(msg in error_msg for msg in fatal_errors):
                    self.logger.critical("Fatal error encountered. Terminating pipeline to prevent further failures.")
                    if results:
                        self.save_raw_answers_to_csv(results)
                    self.model.stop()
                    sys.exit(1)
                continue

        if results:
            unique_results_map = {r["problem_id"]: r for r in results}
            results = sorted(unique_results_map.values(), key=lambda x: x["problem_id"])
            self.save_raw_answers_to_csv(results)

        if all_descriptions_data and self.descriptions_path:
            self.save_descriptions_to_json(self.descriptions_path, all_descriptions_data)

        self.logger.info(f"Run completed for {self.dataset_name}.")

    def get_prompt(self, prompt_type: str) -> str:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(current_dir))

            if prompt_type.startswith(("problem_description", "sample_answers")):
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
            
            if not os.path.exists(prompt_path):
                self.logger.warning(f"Prompt file not found: {prompt_path}")
                return ""

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
        sample_answer_prompt: Optional[str] = None,
        describe_prompt: Optional[str] = None,
    ) -> None:
        """Save dataset, strategy, model, and config info into a metadata file."""

        # FIX: CLEAN SERIALIZATION FOR DATACLASS
        if is_dataclass(self.config):
            config_data = asdict(self.config)
        else:
            # Fallback for non-dataclass objects
            try:
                config_data = vars(self.config)
            except Exception:
                config_data = str(self.config)

        metadata = {
            "dataset": self.dataset_name,
            "strategy": self.strategy_name,
            "model": self.model.get_model_name(),
            "param_set_number": self.param_set_number,
            "config": config_data,
            "problem_description_prompt": problem_description_prompt,
            "sample_answer_prompt": sample_answer_prompt,
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

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"Saved {len(results)} results to {output_path}")

    # FUNCTIONS FOR GETTING IMAGES AND PANELS

    def get_choice_panel(self, problem_id: str) -> Optional[str]:
        if hasattr(self.config, "category") and self.config.category != "standard":
            self.logger.warning(
                f"get_choice_panel called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return None
        image_path = f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/choice_panel.png"
        return image_path

    def get_choice_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        if not self.verify_choice_index(image_index):
            return ""
        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/choices/{image_index}.png"
        )
        return image_path

    def get_question_panel(self, problem_id: str) -> str:
        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/question_panel.png"
        )
        return image_path

    def get_question_image(self, problem_id: str) -> str:
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
        image_path = (
            f"{self.data_dir}/{self.dataset_name}/problems/{problem_id}/classification_panel.png"
        )
        return image_path

    def get_list_of_choice_images(
        self, problem_id: str, image_indices: List[Union[str, int]]
    ) -> List[str]:
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
                if not isinstance(image_index, str):
                    return False
            elif self.config.category == "BP":
                valid_indices = [i for i in range(self.config.num_choices)]
                if not isinstance(image_index, int):
                    return False
            else:
                return False

            if image_index not in valid_indices:
                return False
        except AttributeError as e:
            self.logger.exception(f"Config is missing an attribute: {e}")
            return False
        return True

    def save_descriptions_to_json(
        self, descriptions_path: str, all_descriptions_data: dict
    ):
        try:
            directory = os.path.dirname(descriptions_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(descriptions_path, "w", encoding="utf-8") as f:
                json.dump(all_descriptions_data, f, indent=4)
            self.logger.info(f"Saved descriptions to {descriptions_path}")
        except Exception as e:
            self.logger.error(f"Error in save_descriptions_to_json: {e}")