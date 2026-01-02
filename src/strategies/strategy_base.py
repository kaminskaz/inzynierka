import logging
from abc import ABC, abstractmethod
import sys
from typing import Any, List, Union, Optional, Dict
import os
import csv
import json
from dataclasses import asdict, is_dataclass
import time
from src.technical.exceptions import PipelineCriticalError

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
        self.problem_description_prompt = self.get_prompt(f"problem_description", self.prompt_number)
        self.sample_answer_prompt = self.get_prompt(f"sample_answer", self.prompt_number)
        self.question_prompt = self.get_prompt(f"question", self.prompt_number)
        self.main_prompt = f"{self.problem_description_prompt}\n{self.question_prompt}"
        self.descriptions_prompt = None
        self.example_prompt = self.get_prompt(f"example", self.prompt_number)

        self.descriptions_path: Optional[str] = None

        self.logger.info(f"Initialized strategy for dataset: '{self.dataset_name}'")

    @abstractmethod
    def _execute_problem(self, problem_id: str) -> list[Optional[ResponseSchema], str, Optional[Dict[str, str]]]:  # type: ignore
        pass

    def _load_existing_results(self, csv_path: str) -> List[dict]:
        """Helper to read existing CSV results into a list of dicts."""
        existing_data = []
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Verify that 'problem_id' exists in the header
                if reader.fieldnames and "problem_id" in reader.fieldnames:
                    existing_data = list(reader)
                else:
                    self.logger.warning(f"CSV at {csv_path} is missing 'problem_id' header.")
            self.logger.info(f"Loaded {len(existing_data)} existing results from {csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to load existing results from {csv_path}: {e}")
        return existing_data

    def run(self, restart_problem_id: str = "") -> None:
        """
        Main execution loop with robust handling for empty results, 
        missing files, and persistent descriptions.
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
            unique_results_map = {
                r['problem_id']: r 
                for r in raw_results 
                if r.get('problem_id') is not None
            }
            results = list(unique_results_map.values())
            self.logger.info(f"Loaded {len(results)} unique results from existing CSV.")

        descriptions_path = os.path.join(self.results_dir, "descriptions.json")
        if self.descriptions_path and os.path.exists(descriptions_path):
            try:
                with open(self.descriptions_path, 'r', encoding='utf-8') as f:
                    all_descriptions_data = json.load(f)
                self.logger.info(f"Loaded {len(all_descriptions_data)} existing descriptions from JSON.")
            except Exception as e:
                self.logger.warning(f"Failed to load existing descriptions: {e}. Starting fresh.")
                all_descriptions_data = {}

        if len(results) >= self.config.expected_num_samples:
            self.logger.info(
                f"Dataset {self.dataset_name} is already fully processed "
                f"({len(results)}/{self.config.expected_num_samples}). Exiting pipeline."
            )
            #self.model.stop()
            return

        entries = [e for e in os.scandir(self.dataset_dir) if e.is_dir()]
        entries.sort(key=lambda entry: entry.name)
        
        processed_ids = {r['problem_id'] for r in results if 'problem_id' in r}

        if restart_problem_id:
            self.logger.info(f"Manual restart requested from problem ID: {restart_problem_id}")
            entries = [e for e in entries if e.name >= restart_problem_id]
            results = [r for r in results if r.get('problem_id', '') < restart_problem_id]
            all_descriptions_data = {
                k: v for k, v in all_descriptions_data.items() if k < restart_problem_id
            }
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

                num_digits = len(str(self.config.expected_num_samples))
                problem_id_str = str(problem_id).zfill(num_digits)

                result = {
                    "problem_id": problem_id_str,
                    "answer": get_field(response, "answer", "") if response else "",
                    "confidence": get_field(response, "confidence", "") if response else "",
                    "rationale": get_field(response, "rationale", "") if response else "",
                    "raw_response": response if response else "Response is None",
                }

                results.append(result)
                self.save_raw_answers_to_csv(results)
                time.sleep(0.1)

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error processing {problem_entry.name}: {error_msg}")
                
                fatal_errors = ["Request timed out"]
                if any(msg in error_msg for msg in fatal_errors):
                    self.logger.critical("Fatal error encountered. Raising exception for restart.")
                    if results:
                        self.save_raw_answers_to_csv(results)
                        if all_descriptions_data and self.descriptions_path:
                                self.save_descriptions_to_json(self.descriptions_path, all_descriptions_data)
                    raise PipelineCriticalError(f"Critical failure: {error_msg}")
                continue

        if results:
            unique_results_map = {r["problem_id"]: r for r in results if "problem_id" in r}
            results = sorted(unique_results_map.values(), key=lambda x: x["problem_id"])
            self.save_raw_answers_to_csv(results)

        if all_descriptions_data and self.descriptions_path:
            self.save_descriptions_to_json(self.descriptions_path, all_descriptions_data)

        self.logger.info(f"Run completed for {self.dataset_name}.")

    def get_prompt(self, prompt_type: str, prompt_number: int) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(current_dir))

        if prompt_type.startswith(("problem_description", "sample_answer")):
            prompt_path = os.path.join(repo_root, "prompts", self.dataset_name, f"{prompt_type}_{prompt_number}.txt")
        else:
            prompt_path = os.path.join(
                repo_root,
                "prompts",
                self.dataset_name,
                self.strategy_name,
                f"{prompt_type}_{prompt_number}.txt",
            )

        if not os.path.exists(prompt_path):
            if prompt_number != 1:
                self.logger.warning(f"Prompt {prompt_number} not found. Falling back to version 1.")
                
                if prompt_type.startswith(("problem_description", "sample_answer")):
                    prompt_path = os.path.join(repo_root, "prompts", self.dataset_name, f"{prompt_type}_1.txt")
                else:
                    prompt_path = os.path.join(repo_root, "prompts", self.dataset_name, self.strategy_name, f"{prompt_type}_1.txt")
                
                if not os.path.exists(prompt_path):
                    raise ValueError(f"Neither prompt {prompt_number} nor default prompt 1 exists at {prompt_path}")
            else:
                raise ValueError(f"Primary prompt file not found: {prompt_path}")

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, IOError) as e:
            self.logger.exception(f"Error reading prompt file at {prompt_path}")
            raise ValueError(f"Error reading prompt file: {e}") from e
        
    def save_metadata(self, question_prompt: str, problem_description_prompt: str, 
                      sample_answer_prompt: Optional[str] = None, 
                      describe_prompt: Optional[str] = None) -> None:
        if is_dataclass(self.config):
            config_data = asdict(self.config)
        else:
            try:
                config_data = vars(self.config)
            except Exception:
                config_data = str(self.config)

        metadata = {
            "dataset": self.dataset_name,
            "strategy": self.strategy_name,
            "model": self.model.get_model_name(),
            "param_set_number": self.param_set_number,
            "prompt_number": self.prompt_number,
            "config": config_data,
            "problem_description_prompt": problem_description_prompt,
            "sample_answer_prompt": sample_answer_prompt,
            "question_prompt": question_prompt,
            "describe_prompt": describe_prompt,
            "example_prompt": self.example_prompt,
        }
        try:
            metadata_path = os.path.join(self.results_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            self.logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            self.logger.exception(f"Failed to save metadata: {e}")

    def save_raw_answers_to_csv(self, results: List[dict]) -> None:
        """Saves results to CSV. Correctly handles empty result sets by using predefined headers."""
        if not results:
            self.logger.warning("No results to save.")
            return

        output_path = os.path.join(self.results_dir, "results.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fieldnames = ["problem_id", "answer", "confidence", "rationale", "raw_response"]

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            cleaned_results = [{k: r.get(k, "") for k in fieldnames} for r in results]
            writer.writerows(cleaned_results)

        self.logger.info(f"Saved {len(results)} results to {output_path}")

    def save_descriptions_to_json(self, descriptions_path: str, all_descriptions_data: dict):
        try:
            os.makedirs(os.path.dirname(descriptions_path), exist_ok=True)
            with open(descriptions_path, "w", encoding="utf-8") as f:
                json.dump(all_descriptions_data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving descriptions: {e}")

    # --- Image Handling Methods ---
    def _build_image_path(self, problem_id: str, *subpaths: str) -> str:
        """Centralizes path construction to make the code easy to maintain."""
        return os.path.join(self.data_dir, self.dataset_name, "problems", problem_id, *subpaths)
    
    def get_choice_panel(self, problem_id: str) -> Optional[str]:
        if getattr(self.config, "category", None) != "standard":
            return None
        return self._build_image_path(problem_id, "choice_panel.png")

    def get_choice_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        if not self.verify_choice_index(image_index):
            return ""
        return self._build_image_path(problem_id, "choices", f"{image_index}.png")

    def get_question_panel(self, problem_id: str) -> str:
        return self._build_image_path(problem_id, "question_panel.png")

    def get_question_image(self, problem_id: str) -> str:
        if getattr(self.config, "category", None) != "standard":
            return ""
        return self._build_image_path(problem_id, "question.png")

    def get_blackout_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        if getattr(self.config, "category", None) != "choice_only":
            return ""
        if not self.verify_choice_index(image_index):
            return ""
        return self._build_image_path(problem_id, "blackout", f"{image_index}.png")

    def get_classification_panel(self, problem_id: str) -> str:
        return self._build_image_path(problem_id, "classification_panel.png")

    def verify_choice_index(self, image_index: Union[str, int]) -> bool:
        if not hasattr(self.config, "category"):
            return False
        try:
            if self.config.category in ["standard", "choice_only"]:
                valid_indices = [chr(i) for i in range(ord("A"), ord("A") + self.config.num_choices)]
                return isinstance(image_index, str) and image_index in valid_indices
            elif self.config.category == "BP":
                valid_indices = list(range(self.config.num_choices))
                return isinstance(image_index, int) and image_index in valid_indices
            return False
        except Exception:
            return False