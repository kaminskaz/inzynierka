from abc import ABC, abstractmethod
import json
import string
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
            prompt = None,
            model_object = None,
            concat = True, 
            output_all_results_concat_path = None
        ):

        results_dir = make_dir_for_results(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_name=model_name,
            version=version,
            create_dir=False
        )

        if output_all_results_concat_path is None:
            default_dir = results_dir.split("results")[0] + "results"
            output_all_results_concat_path = os.path.join(default_dir, "all_results_concat.csv")

        d_category = get_dataset_config(dataset_name).category

        results_dir, answers_path, key_path = self.get_evaluation_paths(
            strategy_name,
            dataset_name,
            model_name,
            version
        )

        if not results_dir or not answers_path or not key_path:
            return

        answers_df = pd.read_csv(answers_path, dtype={"problem_id": str}, encoding="utf-8")
        # for now fixed width of 3 for problem ids (e.g., 001, 002, ..., 010, etc.)
        answers_df["problem_id"] = answers_df["problem_id"].apply(lambda x: str(x).zfill(3))

        metadata_path = os.path.join(results_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        if metadata["strategy"] == "descriptive" or metadata["strategy"] == "contrastive":
            with open(os.path.join(results_dir, "descriptions.json"), "r") as f:
                descriptions = json.load(f)
        else:
            descriptions = None

        summary_answers = self.check_completeness(answers_df, metadata, descriptions)
        logger.info(f"Answers DataFrame Completeness Summary: {summary_answers}")

        output_df = answers_df.copy()
        output_df["score"] = ""

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        key_df = pd.DataFrame({
            "problem_id": list(key_dict.keys()),
            "answer": list(key_dict.values())  
        })
        summary_key = self.check_completeness(key_df, metadata)
        logger.info(f"Key DataFrame Completeness Summary: {summary_key}")

        if d_category == "BP":
            self.evaluate(
                answers_df=answers_df,
                key_dict=key_dict,
                output_df=output_df,
                prompt=prompt,
                model_object=model_object
            )
            
        elif d_category == "standard" or d_category == "choice_only":
            self.evaluate(
                answers_df=answers_df,
                key_dict=key_dict,
                output_df=output_df,
            )
        
        output_summaries_path = f"{results_dir}/{evaluation_output_path}_summary.json"
        with open(output_summaries_path, "w") as summary_file:
            json.dump({
                "answers_completeness": summary_answers,
                "key_completeness": summary_key
            }, summary_file, indent=4)
        logger.info(f"Summaries saved to {output_summaries_path}")

        metrics = self.calculate_metrics(output_df)
        metrics_path = f"{results_dir}/{evaluation_output_path}_metrics.json"
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

        output_path = f"{results_dir}/{evaluation_output_path}.csv"
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
 
        if concat:
            if os.path.exists(output_all_results_concat_path):
                concat_df = pd.read_csv(output_all_results_concat_path)
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

    def check_completeness(self, df, metadata, descriptions = None) -> dict:
        summary = {}

        summary["row_ids_with_any_missing"] = df.index[df.isna().any(axis=1)].tolist()
        summary["row_ids_fully_missing"] = df.index[df.isna().all(axis=1)].tolist()

        summary["missing_count_per_column"] = df.isna().sum().to_dict()
        summary["missing_ratio_per_column"] = df.isna().mean().to_dict()

        expected_num_samples = metadata["config"]["expected_num_samples"]
        summary["expected_num_samples"] = expected_num_samples

        # num_digits = len(str(expected_num_samples - 1)) 
        # for now fixed for 3 digits !
        expected_ids = {str(i).zfill(3) for i in range(expected_num_samples)}
        actual_ids = set(df["problem_id"].tolist())

        missing_ids = sorted(expected_ids - actual_ids)
        summary["missing_problem_ids"] = missing_ids
        summary["num_missing_problem_ids"] = len(missing_ids)

        if descriptions:
            json_summary = {}

            max_inner_count = max(len(inner) for inner in descriptions.values())

            incomplete_ids = {}
            for outer_id, inner_dict in descriptions.items():
                missing_inner = sorted(
                [k for k, v in inner_dict.items() if v is None or v.strip() == ""]
            )
            if missing_inner:
                incomplete_ids[outer_id] = missing_inner

            json_summary["problem_ids_with_missing_descriptions"] = incomplete_ids
            json_summary["num_problem_ids_with_missing_descriptions"] = len(incomplete_ids)

            summary["descriptions_completeness"] = json_summary

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
            version=version,
            create_dir=False
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