import logging
from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional, Dict
import os
import csv
import json
import pandas as pd

from code.preprocessing.processor_config import ProcessorConfig
from code.models.vllm import VLLM
from code.technical.response_schema import GeneralEnsembleSchema
from code.technical.utils import get_dataset_config
from run_single_experiment import run_single_experiment
from code.models.llm_judge import LLMJudge

class EnsembleBase(ABC):
    def __init__(self, dataset: str, members_configuration: List[List[str]], run_missing: bool = True):
        self.logger = logging.getLogger(__name__)
        self.dataset = dataset
        self.config: Dict[str, Any] = {}
        self.run_missing = run_missing
        self.members_configuration = members_configuration
        self.answers = pd.DataFrame()
        self.dataset_config = get_dataset_config(dataset)
        self.seed = 42

        self._build_ensemble()
        self.llm = LLMJudge()

    def get_results_dir(self, dataset: str, strategy: str, model: str, version: str = '1',) -> str:
        base_dir = f"results/{dataset}_{strategy}_{model}_{version}"
        if not os.path.exists(base_dir):
            self.logger.warning(f"Directory {base_dir} does not exist.")
            return ""
        return base_dir

    def load_data_from_results_path(
        self, 
        dataset: str, 
        strategy: str, 
        model: str, 
        version: str = "1"
    ) -> tuple[pd.DataFrame, dict]:
        
        results_dir = self.get_results_dir(dataset, strategy, model, version)
        path_to_csv = os.path.join(results_dir, "results.csv")
        path_to_metadata = os.path.join(results_dir, "metadata.json")

        try:
                results_df = pd.read_csv(path_to_csv)

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
        for idx, mem in enumerate(self.members_configuration):
            strategy, model, version = mem

            if self.dataset == 'bp' and strategy == 'classification':
                self.logger.info(
                    f"Skipping member {idx}: 'classification' strategy is not allowed"
                    f"for dataset '{self.dataset}'."
                )
                continue

            df, meta = self.load_data_from_results_path(self.dataset, strategy, model, version)

            if meta is None:
                self.logger.warning(f"No metadata found for member {idx} with strategy {strategy}, model {model}.")
                meta = {}

            if df.empty:
                self.logger.warning(f"No data loaded for member {idx} with strategy {strategy}, model {model}.")
                if self.run_missing:
                    self.logger.info(f"Running new for member {idx} with strategy {strategy}, model {model}.")
                    run_single_experiment(dataset_name=self.dataset, 
                                          strategy_name=strategy,
                                          model_name=model)
                    df, meta = self.load_data_from_results_path(self.dataset, strategy, model, version)
            
            self.config[f"member_{idx}"] = meta

            df = df.copy()
            df["member_idx"] = idx
            ensemble_df = pd.concat([ensemble_df, df], ignore_index=True)

        self.answers = ensemble_df
        return ensemble_df

    def evaluate(self):
        results = []
        problem_ids = self.answers["problem_id"].unique()

        for problem_id in problem_ids:
            final_answer = self.evaluate_single_problem(problem_id)
            results.append({
                "problem_id": problem_id,
                "ensemble_answer": final_answer
            })

        results_df = pd.DataFrame(results)
        return results_df
        
    @abstractmethod
    def evaluate_single_problem(self):
        pass
            
        