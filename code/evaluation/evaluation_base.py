from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel
import pandas as pd
import os

class EvaluationBase(ABC):

    @abstractmethod
    def evaluate_single_answer(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def evaluate(self, answers_path, key_path, output_path):
        pass

    @abstractmethod
    def run_evaluation(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_metrics(self, *args, **kwargs) -> dict:
        pass

    def check_completeness(self, df) -> dict:
        summary = {}

        summary["row_ids_with_any_missing"] = df.index[df.isna().any(axis=1)].tolist()
        summary["row_ids_fully_missing"] = df.index[df.isna().all(axis=1)].tolist()

        summary["missing_count_per_column"] = df.isna().sum().to_dict()
        summary["missing_ratio_per_column"] = df.isna().mean().to_dict()

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
            "reasoning"
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
        combined_df = combined_df.drop_duplicates(subset=["dataset_name", "model_name", "strategy_name", "version"], keep='last')

        combined_df.to_csv(all_results_concat_path, index=False)
    