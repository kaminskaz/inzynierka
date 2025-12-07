import pandas as pd
import json
import logging
import os
from pathlib import Path

from code.evaluation.evaluation_base import EvaluationBase

logger = logging.getLogger(__name__)


class EvaluationBasic(EvaluationBase):
    def evaluate_single_answer(self, answer: str, key: str) -> float:
        score = "Right" if answer == key else "Wrong"
        return score

    def evaluate(
            self, 
            strategy_name: str,
            dataset_name: str,
            model_name: str,
            version: str,
            evaluation_output_path: str = "evaluation_results"
        ):
        
        results_dir, answers_path, key_path = self.get_evaluation_paths(
            strategy_name,
            dataset_name,
            model_name,
            version
        )

        if not results_dir or not answers_path or not key_path:
            return

        answers_df = pd.read_csv(answers_path, dtype={"problem_id": str}, encoding="utf-8")

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

        key_df = df = pd.DataFrame({
            "problem_id": list(key_dict.keys()),
            "answer": list(key_dict.values())  
        })
        summary_key = self.check_completeness(key_df, metadata)
        logger.info(f"Key DataFrame Completeness Summary: {summary_key}")

        for index, row in answers_df.iterrows():
            answer = row["answer"]
            id_ = str(row["problem_id"])

            if answer is None or pd.isna(answer) or answer.strip() == "":
                output_df.at[index, "score"] = "No answer provided"
                continue

            if id_ not in key_dict:
                logger.info(f"ID {id_} not found in key file.")
                output_df.at[index, "score"] = "Problem id not found in key"
                continue

            key = key_dict[id_].strip().upper()

            score = self.evaluate_single_answer(answer, key)
            output_df.at[index, "score"] = score
            output_df.at[index, "key"] = key

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
        