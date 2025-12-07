from abc import ABC, abstractmethod
from typing import List, Optional, Type
import logging
from pydantic import BaseModel
import pandas as pd
import os
from code.technical.utils import make_dir_for_results, shorten_model_name, get_dataset_config
from pathlib import Path

logger = logging.getLogger(__name__)

class EvaluationBase(ABC):

    @abstractmethod
    def evaluate_single_answer(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    def run_evaluation(
            self, 
            dataset_name,
            model_name,
            strategy_name,
            version,
            evaluation_output_path = "evaluation_results", 
            prompt=None,
            concat = True, 
            output_all_results_concat_path = None
        ):
        results_dir = make_dir_for_results(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_name=model_name,
            version=version
        )
        print(f"Results directory: {results_dir}")
        if output_all_results_concat_path  is None:
            default_dir = results_dir.split("results")[0] + "results"
            output_all_results_concat_path = os.path.join(default_dir, "all_results_concat.csv")

        d_category = get_dataset_config(dataset_name).category

        if d_category == "bp":
            self.evaluate(
                strategy_name=strategy_name,
                dataset_name=dataset_name,
                model_name=model_name,
                version=version,
                prompt=prompt,
                evaluation_output_path=evaluation_output_path
            )
            
        elif d_category == "standard" or d_category == "choice_only":
            self.evaluate(
                strategy_name=strategy_name,
                dataset_name=dataset_name,
                model_name=model_name,
                version=version,
                evaluation_output_path=evaluation_output_path
            )
        
        if concat:
            csv_path = f"{results_dir}/{evaluation_output_path}.csv"

            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

            if os.path.exists(csv_path):
                concat_df = pd.read_csv(csv_path)
            else:
                concat_df = pd.DataFrame()

            self.append_to_all_results_concat(
                dataset_name,
                model_name,
                strategy_name,
                version,
                concat_df,
                output_all_results_concat_path
            )

    @abstractmethod
    def calculate_metrics(self, *args, **kwargs) -> dict:
        pass

    def check_completeness(self, df, descriptions = None) -> dict:
        summary = {}

        summary["row_ids_with_any_missing"] = df.index[df.isna().any(axis=1)].tolist()
        summary["row_ids_fully_missing"] = df.index[df.isna().all(axis=1)].tolist()

        summary["missing_count_per_column"] = df.isna().sum().to_dict()
        summary["missing_ratio_per_column"] = df.isna().mean().to_dict()

        if descriptions:
            json_summary = {}

            max_inner_count = max(len(inner) for inner in descriptions.values())
            expected_keys = {str(i) for i in range(max_inner_count)}

            incomplete_ids = {}
            for outer_id, inner_dict in descriptions.items():
                missing_inner = sorted(list(expected_keys - set(inner_dict.keys())))
                if missing_inner:
                    incomplete_ids[outer_id] = missing_inner

            json_summary["problem_ids_with_missing_descriptions"] = incomplete_ids
            json_summary["num_problem_ids_with_missing_descriptions"] = len(incomplete_ids)
            json_summary["num_descriptions_expected"] = sorted(list(expected_keys))

            summary["descriptions_completeness"] = json_summary

        total_cells = df.size
        missing_cells = int(df.isna().sum().sum())
        summary["completeness_ratio"] = float(1 - (missing_cells / total_cells if total_cells > 0 else 0))

        return summary
    
    def append_to_all_results_concat(
            self, 
            dataset_name,
            model_name,
            strategy_name,
            version, 
            df, 
            all_results_concat_path
        ):
        df["dataset_name"]  = dataset_name
        df["model_name"]    = shorten_model_name(model_name)
        df["strategy_name"] = strategy_name
        df["version"]       = version
        if os.path.exists(all_results_concat_path):
            existing_df  = pd.read_csv(all_results_concat_path)
            combined_df  = pd.concat([existing_df, df ], ignore_index=True)
        else:
            combined_df  = df

        meta_cols = [
            "reasoning",
            "dataset_name",
            "model_name",
            "strategy_name",
            "version"
        ]

        for col in meta_cols:
            if col not in combined_df.columns:
                combined_df[col] = ""


        other_cols = [c for c in combined_df.columns if c not in meta_cols]
        final_order = meta_cols + other_cols

        combined_df = combined_df[final_order]
        combined_df = combined_df.drop_duplicates(subset=["problem_id", "dataset_name", "model_name", "strategy_name", "version"], keep='last')

        combined_df.to_csv(all_results_concat_path, index=False)

    def get_evaluation_paths(
            self,
            strategy_name: str,
            dataset_name: str,
            model_name: str,
            version: str
        ):
        
        results_dir = make_dir_for_results(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_name=model_name,
            version=version
        )
        answers_path = f"{results_dir}/results.csv"
        key_path = f"data/{dataset_name}/jsons/{dataset_name}_solutions.json"

        if not results_dir or not os.path.exists(results_dir):
            logger.error("Results directory is not provided or does not exist.")
            results_dir = None
        
        if not answers_path or not os.path.exists(answers_path):
            logger.error("Answers path is not provided or does not exist.")
            answers_path = None 

        if not key_path or not os.path.exists(key_path):
            logger.error("Key path is not provided or does not exist.")
            key_path = None

        return results_dir, answers_path, key_path