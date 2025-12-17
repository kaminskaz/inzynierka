from pathlib import Path
from typing import Any
import pandas as pd
import json
import logging
import os

from code.evaluation.evaluation_base import EvaluationBase
from code.models.llm_judge import LLMJudge
from code.technical.response_schema import BongardEvaluationSchema
from code.technical.utils import get_results_directory

logger = logging.getLogger(__name__)


class EvaluationWithJudge(EvaluationBase):
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 model_object: Any = None,
                 param_set_number: int = None,
                 prompt: str = None,
                 prompt_number: int = 1):
        self.prompt_number = prompt_number
        try:
            if prompt is not None:
                self.prompt = prompt
            else:
                prompt_path = os.path.join("prompts", "evaluation", f"evaluation_bongard_{self.prompt_number}.txt")
                with open(prompt_path, "r") as file:
                    self.prompt = file.read()
        except Exception as e:
            logger.error(f"Failed to read prompt file: {e}")

        if model_object is not None:
            self.judge = model_object

        else:
            self.judge = LLMJudge(
                model_name=model_name,
                param_set_number=param_set_number
            )

    def evaluate_single_answer(
        self,
        prompt: str,
        answer: str,
        key: str,
        model: LLMJudge,
        response_schema: BongardEvaluationSchema,
    ):
        return model.evaluate_similarity(
            prompt=prompt, 
            answer=answer, 
            key=key, 
            response_schema=response_schema
        )

    def evaluate(
            self, 
            answers_df: pd.DataFrame, 
            key_dict: dict,
            output_df: pd.DataFrame,
            prompt: str = None,
            model_object: Any = None
        ):
        
        for index, row in answers_df.iterrows():
            answer = row.get("answer") or row.get("ensemble_answer")
            id_ = str(row["problem_id"])

            if id_ not in key_dict:
                logger.info(f"ID {id_} not found in key file.")
                output_df.at[index, "score"] = "Problem id not found in key"
                output_df.at[index, "key"] = "Key missing"
                continue

            left_rule, right_rule = key_dict[id_]
            key = f"{left_rule} vs. {right_rule}"

            if answer is None or pd.isna(answer) or answer.strip() == "":
                output_df.at[index, "score"] = "No answer provided"
                output_df.at[index, "key"] = key
                continue

            score, reasoning = self.evaluate_single_answer(
                prompt=prompt if prompt else self.prompt,
                answer=answer,
                key=key,
                model=model_object if model_object else self.judge,
                response_schema=BongardEvaluationSchema,
            )

            output_df.at[index, "key"] = key

            if score is None:
                output_df.at[index, "score"] = "LLM evaluation failed"
                continue

            if reasoning is None:
                output_df.at[index, "reasoning"] = "LLM reasoning missing"

            output_df.at[index, "score"] = score
            output_df.at[index, "reasoning"] = reasoning
            

    def calculate_metrics(self, evaluated_df):
        total = len(evaluated_df)
        correct = len(evaluated_df[evaluated_df["score"] == "Right"])
        accuracy = correct / total if total > 0 else 0.0

        if {"score", "confidence"}.issubset(evaluated_df.columns):
            bin_counts = evaluated_df.groupby("score")["confidence"].size().to_dict()
            avg_confidence = evaluated_df.groupby("score")["confidence"].mean().to_dict()
            median_confidence = evaluated_df.groupby("score")["confidence"].median().to_dict()
        else:
            bin_counts = {}
            avg_confidence = {}
            median_confidence = {}

        return {
            "total": total,
            "bin_counts": bin_counts,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "median_confidence": median_confidence
        }
        
