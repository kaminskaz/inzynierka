from pathlib import Path
from typing import Any
import pandas as pd
import json
import logging
import os

from src.evaluation.evaluation_base import EvaluationBase
from src.models.llm_judge import LLMJudge
from src.technical.response_schema import BongardEvaluationSchema
from src.technical.utils import shorten_model_name


class EvaluationWithJudge(EvaluationBase):
    def __init__(
            self, 
            judge_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
            judge_model_object: Any = None,
            judge_param_set_number: int = None,
            prompt: str = None,
            prompt_number: int = 1,
            prompt_path: str = None
        ):
        self.logger = logging.getLogger(__name__)
        self.prompt_number = prompt_number
        self.judge_model_name = judge_model_name
        self.judge_param_set_number = judge_param_set_number

        if prompt is not None:
            self.prompt = prompt
            
        elif prompt_path is not None:
            if not os.path.exists(prompt_path):
                error_msg = f"Prompt file not found: {prompt_path}. Check if prompt path is correct."
                raise ValueError(error_msg)
            try:
                with open(prompt_path, "r") as file:
                    self.prompt = file.read()
                    
            except (OSError, IOError) as e:
                error_msg = f"Error reading prompt file at {prompt_path}: {e}."
                self.logger.exception(error_msg)
                raise ValueError(error_msg) from e
        else:
            prompt_path = os.path.join("prompts", "evaluation", f"evaluation_bongard_{self.prompt_number}.txt")
            
            if not os.path.exists(prompt_path):
                self.logger.warning(f"Prompt {self.prompt_number} not found. Attempting to default to prompt 1.")
                prompt_path = os.path.join("prompts", "evaluation", "evaluation_bongard_1.txt")
                
                if not os.path.exists(prompt_path):
                    error_msg = f"Prompt file not found: {prompt_path}. Default prompt 1 also missing."
                    raise ValueError(error_msg)

            try:
                with open(prompt_path, "r") as file:
                    self.prompt = file.read()
            except (OSError, IOError) as e:
                error_msg = f"Error reading prompt file at {prompt_path}: {e}."
                self.logger.exception(error_msg)
                raise ValueError(error_msg) from e

        if judge_model_object is not None:
            self.judge_model_object = judge_model_object
            self.judge_model_name = judge_model_object.get_model_name()

        else:
            self.logger.info(f"Initializing judge model: {self.judge_model_name}")
            self.judge_model_object = LLMJudge(
                model_name=self.judge_model_name,
                param_set_number=self.judge_param_set_number
            )

    def evaluate_single_answer(
        self,
        answer: str,
        key: str,
        response_schema: BongardEvaluationSchema,
    ):
        return self.judge_model_object.evaluate_similarity(
            prompt=self.prompt, 
            answer=answer, 
            key=key, 
            response_schema=response_schema
        )

    def evaluate(
        self, 
        output_df: pd.DataFrame, 
        key_dict: dict,
        dataset_category: str,
        stop_after_evaluation: bool = False
    ):
    
        for index, row in output_df.iterrows():
            answer = row.get("answer")
            id_ = str(row["problem_id"])

            if id_ not in key_dict:
                self.logger.info(f"ID {id_} not found in key file.")
                output_df.at[index, "score"] = "Problem id not found in key"
                output_df.at[index, "key"] = "Key missing"
                continue

            if dataset_category.lower() == "bp":
                left_rule, right_rule = key_dict[id_]
                key = f"{left_rule} vs. {right_rule}"
            else:
                key = key_dict[id_]

            if answer is None or pd.isna(answer) or answer.strip() == "":
                output_df.at[index, "score"] = "No answer provided"
                output_df.at[index, "key"] = key
                continue

            score, judge_rationale = self.evaluate_single_answer(
                answer=answer,
                key=key,
                response_schema=BongardEvaluationSchema,
            )

            output_df.at[index, "key"] = key

            if score is None:
                output_df.at[index, "score"] = "LLM evaluation failed"
                continue

            if judge_rationale is None:
                output_df.at[index, "judge_rationale"] = "LLM reasoning missing"
            output_df.at[index, "score"] = score
            output_df.at[index, "judge_rationale"] = judge_rationale
        output_df["judge_model_name"] = shorten_model_name(self.judge_model_name)
        output_df["judge_model_param_set"] = (self.judge_param_set_number 
                                              if self.judge_param_set_number is not None 
                                              else 1)
        
        if stop_after_evaluation:
            self.judge_model_object.stop()
            

    def calculate_metrics(self, evaluated_df):
        total = len(evaluated_df)

        if "score" in evaluated_df.columns:
            correct = (evaluated_df["score"] == "Right").sum()
            accuracy = correct / total if total > 0 else 0.0
        else:
            accuracy = 0.0

        if "score" in evaluated_df.columns:
            grouped = evaluated_df.groupby("score")
            bin_counts = grouped.size().to_dict()
        else:
            bin_counts = {}

        if "confidence" in evaluated_df.columns and "score" in evaluated_df.columns:
            bin_counts = grouped["confidence"].size().to_dict()
            avg_confidence = grouped["confidence"].mean().to_dict()
            median_confidence = grouped["confidence"].median().to_dict()
        else:
            avg_confidence = {}
            median_confidence = {}

        return {
            "total": total,
            "bin_counts": bin_counts,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "median_confidence": median_confidence
        }

        
