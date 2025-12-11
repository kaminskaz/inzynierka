from abc import ABC, abstractmethod
import json
import string
from typing import List, Optional, Type
import logging
from pydantic import BaseModel
import pandas as pd
import os
from code.technical.utils import get_results_directory, shorten_model_name, get_dataset_config, get_ensemble_directory
from pathlib import Path

logger = logging.getLogger(__name__)

class EvaluationBase(ABC):
    def __init__(self, model_object: Optional[Type] = None, model_name: Optional[str] = None):
        self.judge = None
        self.model_object = model_object
        self.model_name = model_name

    @abstractmethod
    def evaluate_single_answer(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    def run_evaluation(
            self, 
            dataset_name,
            strategy_name: Optional[str] = None,
            model_name: Optional[str] = None,
            version: Optional[str] = None,
            evaluation_output_path = "evaluation_results", 
            prompt = None,
            model_object = None,
            concat = True, 
            output_all_results_concat_path = None,
            ensemble: bool = False,
            type_name: Optional[str] = None
        ):

        if ensemble:
            if type_name is None:
                raise ValueError("type_name is required when ensemble=True")
            
            results_dir = get_ensemble_directory(
                dataset_name=dataset_name,
                type_name=type_name,
                version=version,
                create_dir=False
            )
            if os.path.exists(os.path.join(results_dir, "ensemble_config.json")):
                with open(os.path.join(results_dir, "ensemble_config.json"), "r") as f:
                    metadata = json.load(f)
                    model_name =  metadata.get("ensemble_model", None)
            else:
                logger.warning("Ensemble config file not found, model_name will be set to None.")
                model_name = None

        else:
            if strategy_name is None:
                raise ValueError("strategy_name is required when ensemble=False")
            if model_name is None:
                raise ValueError("model_name is required when ensemble=False")
            
            results_dir = get_results_directory(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_name=model_name,
            version=version,
            create_dir=False
            )

        if output_all_results_concat_path is None:
            default_dir = results_dir.split("results")[0] + "results"
            output_all_results_concat_path = os.path.join(default_dir, "all_results_concat.csv")

        answers_path, key_path = self.get_evaluation_paths(
            dataset_name=dataset_name,
            results_dir=results_dir,
            ensemble=ensemble
        )
        
        if not results_dir or not answers_path or not key_path:
            return

        answers_df = pd.read_csv(answers_path, dtype={"problem_id": str}, encoding="utf-8")
        # for now fixed width of 3 for problem ids (e.g., 001, 002, ..., 010, etc.)
        answers_df["problem_id"] = answers_df["problem_id"].apply(lambda x: str(x).zfill(3))

        metadata_path = os.path.join(results_dir, "metadata.json")

        descriptions = None
        if not ensemble:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            if metadata["strategy"] == "descriptive" or metadata["strategy"] == "contrastive":
                with open(os.path.join(results_dir, "descriptions.json"), "r") as f:
                    descriptions = json.load(f)

        expected_num_samples = get_dataset_config(dataset_name).expected_num_samples
        summary_answers = self.check_completeness(answers_df, expected_num_samples, descriptions)
        logger.info(f"Answers DataFrame Completeness Summary: {summary_answers}")

        output_df = answers_df.copy()
        output_df["score"] = ""

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        key_df = pd.DataFrame({
            "problem_id": list(key_dict.keys()),
            "answer": list(key_dict.values())  
        })

        summary_key = self.check_completeness(key_df, expected_num_samples)
        logger.info(f"Key DataFrame Completeness Summary: {summary_key}")

        self.evaluate(
            answers_df=answers_df,
            key_dict=key_dict,
            output_df=output_df,
            prompt=prompt,
            model_object=model_object
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
            self.append_to_all_results_concat(
                results_df=output_df,
                all_results_concat_path=output_all_results_concat_path,
                dataset_name=dataset_name,
                model_name=model_name,
                strategy_name=strategy_name,
                version=version,
                ensemble=ensemble,
            )

    @abstractmethod
    def calculate_metrics(self, *args, **kwargs) -> dict:
        pass

    def check_completeness(self, df, expected_num_samples, descriptions = None) -> dict:
        summary = {}

        summary["row_ids_with_any_missing"] = df.index[df.isna().any(axis=1)].tolist()
        summary["row_ids_fully_missing"] = df.index[df.isna().all(axis=1)].tolist()

        summary["missing_count_per_column"] = df.isna().sum().to_dict()
        summary["missing_ratio_per_column"] = df.isna().mean().to_dict()

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
            results_df: pd.DataFrame,
            all_results_concat_path: str,
            dataset_name: str,
            model_name: str = None,
            strategy_name: Optional[str] = None,
            version: Optional[str]= None,
            type_name: Optional[str]= None,
            ensemble: bool = False
        ):
        
        results_df["dataset_name"]  = dataset_name
        results_df["model_name"] = model_name
        results_df["strategy_name"] = strategy_name
        results_df["version"] = version
        results_df["type_name"] = type_name

        if ensemble:
            results_df["ensemble"] = "Ensemble"
        else:
            results_df["ensemble"] = "Single Model"

        if os.path.exists(all_results_concat_path):
            existing_df  = pd.read_csv(all_results_concat_path)
            combined_df  = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            combined_df  = results_df

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
        final_order = other_cols + meta_cols

        combined_df = combined_df[final_order]
        combined_df = combined_df.drop_duplicates(subset=["problem_id", "dataset_name", "model_name", "strategy_name", "version"], keep='last')

        combined_df.to_csv(all_results_concat_path, index=False)

    def get_evaluation_paths(
            self,
            dataset_name: str,
            results_dir: str,
            ensemble: bool = False
        ):

        if ensemble:
            answers_path = f"{results_dir}/ensemble_results.csv"
        else:
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

        return answers_path, key_path