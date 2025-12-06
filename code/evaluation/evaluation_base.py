from abc import ABC, abstractmethod
from typing import List, Optional, Type
from pydantic import BaseModel
import pandas as pd
import os

from code.technical.utils import get_dataset_config
from code.evaluation.evaluation_judge import EvaluationWithJudge 
from code.evaluation.evaluation_basic import EvaluationBasic

class EvaluationBase(ABC):

    @abstractmethod
    def evaluate_single_answer(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def run_evaluation(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_metrics(self, *args, **kwargs) -> dict:
        pass

    def run_multiple_evaluations(
        self,
        strategy_name: List[str],
        dataset_name: List[str],
        model_name: List[str],
        version: List[str],
        judge_prompt: str = "",
        evaluation_output_path: str = "evaluation_results"
    ):
        evaluator_judge = EvaluationWithJudge()
        evaluator_simple = EvaluationBasic()

        for d_name in dataset_name:
            d_category = get_dataset_config(d_name).category 
            for s_name, m_name, ver in zip(strategy_name, dataset_name, model_name, version):
                if d_category == "standard":
                    evaluator = evaluator_simple
                    evaluator.evaluate(
                        dataset_name=d_name,
                        model_name=m_name,
                        strategy_name=s_name,
                        version=ver,
                        evaluation_output_path=evaluation_output_path,
                    )
                else:
                    evaluator = evaluator_judge
                    evaluator.evaluate(
                        dataset_name=d_name,
                        model_name=m_name,
                        strategy_name=s_name,
                        version=ver,
                        prompt=judge_prompt,
                        evaluation_output_path=evaluation_output_path,
                    )

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
        df["model_name"]    = model_name
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
    