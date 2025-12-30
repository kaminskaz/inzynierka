from typing import Any
import pandas as pd
import json
import logging
import os
from pathlib import Path

from src.evaluation.evaluation_base import EvaluationBase

logger = logging.getLogger(__name__)


class EvaluationBasic(EvaluationBase):
    def evaluate_single_answer(self, answer: str, key: str) -> float:
        score = "Right" if answer == key else "Wrong"
        return score

    def evaluate(
            self, 
            output_df: pd.DataFrame, 
            key_dict: dict,
            dataset_category: str = None
        ):
        
        for index, row in output_df.iterrows():
            answer = row.get("answer")
            id_ = str(row["problem_id"])


            if id_ not in key_dict:
                logger.info(f"ID {id_} not found in key file.")
                output_df.at[index, "score"] = "Problem id not found in key"
                output_df.at[index, "key"] = "Key missing"
                continue

            key = key_dict[id_].strip().upper()

            if answer is None or pd.isna(answer) or answer.strip() == "":
                output_df.at[index, "score"] = "No answer provided"
                output_df.at[index, "key"] = key
                continue

            score = self.evaluate_single_answer(answer, key)
            output_df.at[index, "score"] = score
            output_df.at[index, "key"] = key

    def calculate_metrics(self, evaluated_df):
        total = len(evaluated_df)
        correct = len(evaluated_df[evaluated_df["score"] == "Right"])
        accuracy = correct / total if total > 0 else 0.0

        avg_confidence = evaluated_df.groupby("score")["confidence"].mean().to_dict() if "confidence" in evaluated_df.columns else {}
        median_confidence = evaluated_df.groupby("score")["confidence"].median().to_dict() if "confidence" in evaluated_df.columns else {}

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "median_confidence": median_confidence
        }
        