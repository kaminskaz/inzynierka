import logging
from abc import ABC, abstractmethod
import re
from string import Template
from typing import Any, List, Union, Optional, Dict
import os
import csv
import json
import pandas as pd

from src.technical.response_schema import GeneralEnsembleSchema
from src.technical.utils import get_dataset_config, get_results_directory, get_ensemble_directory
from src.tests.strategy_test import run_single_experiment


class EnsembleBase(ABC):
    def __init__(self, dataset_name: str, members_configuration: List[List[str]], skip_missing: bool = True, type_name: str = "", prompt_number: int = 1):
        self.logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.config: Dict[str, Any] = {}
        self.prompt_number = prompt_number
        self.skip_missing = skip_missing
        self.members_configuration = members_configuration
        self.answers = pd.DataFrame()
        self.dataset_config = get_dataset_config(dataset_name)
        self.seed = 42
        self.type_name = type_name
        self.ensemble_directory = None
        self.exists = False
        self.config["ensemble_model"] = ""
        self.config["dataset"] = self.dataset_name
        self.config["dataset_category"] = self.dataset_config.get("category", "")
        self.config["task_type"] = self.dataset_config.get("task_type", "")

        self._build_ensemble()
        
        self.config["main_prompt"] = self._get_filled_prompt()

    def get_results_dir(self, dataset_name: str, strategy: str, model_name: str, version: str = '1',) -> str:
        base_dir = get_results_directory(dataset_name, strategy, model_name, version)
        if not os.path.exists(base_dir):
            self.logger.warning(f"Directory {base_dir} does not exist.")
            return ""
        return base_dir

    def load_data_from_results_path(
        self, 
        dataset_name: str, 
        strategy: str, 
        model_name: str, 
        version: str = "1"
    ) -> tuple[pd.DataFrame, dict]:
        
        results_dir = self.get_results_dir(dataset_name, strategy, model_name, version)
        path_to_csv = os.path.join(results_dir, "results.csv")
        path_to_metadata = os.path.join(results_dir, "metadata.json")

        try:
            results_df = pd.read_csv(path_to_csv, dtype={"problem_id": str}, encoding="utf-8")
            num_digits = len(str(self.dataset_config.expected_num_samples))
            results_df["problem_id"] = results_df["problem_id"].apply(lambda x: str(x).zfill(num_digits))

            with open(path_to_metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.logger.info(
                f"Loaded results from {path_to_csv} with {len(results_df)} entries."
            )
            self.logger.info(f"Loaded metadata from {path_to_metadata}.")
            return results_df, metadata

        except FileNotFoundError as e:
            self.logger.error(f"Missing file in {results_dir}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading results or metadata from {results_dir}: {e}")

        return pd.DataFrame(), {}

    def _build_ensemble(self) -> pd.DataFrame:
        
        ensemble_df = pd.DataFrame()
        valid_member_idx = 0 

        for idx, mem in enumerate(self.members_configuration):
            strategy, model_name, version = mem

            if self.dataset_name == 'bp' and strategy == 'classification':
                self.logger.info(
                    f"Skipping configuration {idx}: 'classification' strategy is not allowed "
                    f" for dataset '{self.dataset_name}'."
                )
                continue

            df, meta = self.load_data_from_results_path(self.dataset_name, strategy, model_name, version)
            
            if meta is None:
                self.logger.warning(f"""No metadata found for member {idx} with strategy {strategy}, model {model_name}, version {version}.
                                    Defaulting to version '1'.""")
                df, meta = self.load_data_from_results_path(self.dataset_name, strategy, model_name, "1")
                if meta is None:
                    self.logger.error(f"Version 1 not found.")
                    meta = {}

            if df.empty:
                if self.skip_missing:
                    self.logger.warning(f"skip_missing set to 'true': Skipping member with strategy {strategy}, model {model_name}.")
                    continue
                else:
                    self.logger.warning(f"""No data loaded for member {idx} with strategy {strategy}, model {model_name}.  
                                        You can check available configurations in the results/{self.dataset_name} folder""")
                    raise ValueError(f"""Missing ensemble member configurations and skip_missing set to 'false':
                                     Please run the missing configuration and restart the process.""")

            self.config[f"member_{valid_member_idx}"] = meta

            df = df.copy()
            df["member_idx"] = valid_member_idx
            ensemble_df = pd.concat([ensemble_df, df], ignore_index=True)

            valid_member_idx += 1

        self.answers = ensemble_df
        existing_version = self.check_if_ensemble_exists()
        if existing_version:
            self.logger.info(f"Ensemble configuration already exists as version {existing_version}.")
            self.exists = True
            return
        return 

    def evaluate(self) -> None:
        if self.exists:
            self.logger.info("Ensemble already exists. Skipping evaluation.")
            return
        
        self.ensemble_directory = get_ensemble_directory(self.dataset_name, self.type_name, create_dir=True)
        self.save_config_to_json(self.ensemble_directory)

        results = []
        problem_ids = self.answers["problem_id"].unique()

        for problem_id in problem_ids:
            final_answer = self.evaluate_single_problem(problem_id)
            results.append({
                "problem_id": problem_id,
                "answer": final_answer
            })
            results_df = pd.DataFrame(results)
            self.save_results_to_csv(results_df, self.ensemble_directory)

        results_df = pd.DataFrame(results)
        self.save_results_to_csv(results_df, self.ensemble_directory)
        
    @abstractmethod
    def evaluate_single_problem(self):
        pass

    def save_results_to_csv(self, results_df: pd.DataFrame, results_dir: str) -> None:
        path_to_csv = os.path.join(results_dir, "ensemble_results.csv")
        results_df.to_csv(path_to_csv, index=False, mode='w')
        self.logger.info(f"Ensemble results saved to {path_to_csv}.")

    def save_config_to_json(self, results_dir: str) -> None:
        path_to_json = os.path.join(results_dir, "ensemble_config.json")
        with open(path_to_json, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
        self.logger.info(f"Ensemble configuration saved to {path_to_json}.")

    def check_if_ensemble_exists(self) -> Optional[str]:
        base_results_dir = os.path.join("results", "ensembles", self.dataset_name, self.type_name)
        if not os.path.exists(base_results_dir):
            return None
        prefix = f"ensemble_"
        version_pattern = re.compile(rf"^{re.escape(prefix)}ver(\d+)$")
        for entry in os.scandir(base_results_dir):
            if entry.is_dir():
                match = version_pattern.match(entry.name)
                if match:
                    version = match.group(1)
                    path_to_json = os.path.join(entry.path, "ensemble_config.json")
                    try:
                        with open(path_to_json, "r", encoding="utf-8") as f:
                            existing_config = json.load(f)
                        if self.check_if_configs_equal(existing_config):
                            return version
                    except Exception as e:
                        self.logger.error(f"Error reading ensemble config from {path_to_json}: {e}")
        return None

    def check_if_configs_equal(self, other_config: dict) -> bool:
        # checks if the meta values in self.config match the ones in other_config, ignoring the 'member_{idx}' keys.
        self_metas = [m for m in self.config.values() if m != {}]
        other_metas = [v for v in other_config.values() if v != {}]

        for meta in self_metas:
            if meta in other_metas:
                other_metas.remove(meta)
            else:
                return False
        return len(other_metas) == 0
    
    def get_ensemble_prompt_path(self, prompt_number: int = 1):
        prompt_path = os.path.join("prompts", "ensemble", f"ensemble_{self.type_name}_{prompt_number}.txt")
        
        if not os.path.exists(prompt_path):
            if prompt_number != 1:
                self.logger.warning(f"Prompt {prompt_number} not found for {self.type_name}. Falling back to prompt 1.")
                prompt_path = os.path.join("prompts", "ensemble", f"ensemble_{self.type_name}_1.txt")
                
                if not os.path.exists(prompt_path):
                    error_msg = f"Prompt file not found: {prompt_path}. Default prompt 1 also missing."
                    raise ValueError(error_msg)
            else:
                error_msg = f"Primary prompt 1 not found: {prompt_path}."
                raise ValueError(error_msg)

        return prompt_path
    
    def _get_filled_prompt(self, all_answers: str = None) -> str:
        prompt_path = self.get_ensemble_prompt_path(self.dataset_name, self.prompt_number)
        with open(prompt_path, "r", encoding="utf-8") as f:
            main_prompt = f.read()
        
        first_member = next(v for k, v in self.config.items() if k.startswith("member_"))
        sample_answer = first_member.get("sample_answer_prompt", "")
        problem_description = first_member.get("problem_description_prompt", "")
        task_type = self.dataset_config.get("task_type", "")
        if task_type == "close-ended":
            example_prompt_path = os.path.join("prompts", "ensemble", "close_ended_example.txt")
        else:
            example_prompt_path = os.path.join("prompts", "ensemble", "open_ended_example.txt")
        
        if not os.path.exists(example_prompt_path):
            raise ValueError(f"Example prompt file not found: {example_prompt_path}")
        with open(example_prompt_path, "r", encoding="utf-8") as f:
            example = f.read()

        if all_answers is None:
            all_answers = "The ensemble members' answers provided here."

        template = Template(main_prompt)
        main_prompt_filled = template.substitute(
            problem_description=problem_description,
            all_answers=all_answers,
            sample_answer=sample_answer,
            example=example
        )

        return main_prompt_filled