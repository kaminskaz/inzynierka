import pandas as pd
import json
import logging
import os

from code.evaluation.evaluation_base import EvaluationBase
from code.models.llm_judge import LLMJudge
from code.technical.response_schema import BongardEvaluationSchema

logger = logging.getLogger(__name__)


class EvaluationWithJudge(EvaluationBase):
    def __init__(self):
        super().__init__()
        try:
            # na razie na prompcie dla bongarda tylko jeśli się nie poda prompta z zewnątrz
            with open("prompts/evaluation/evaluation_bongard_main.txt", "r") as file:
                self.prompt = file.read()
        except Exception as e:
            logger.error(f"Failed to read prompt file: {e}")

    def evaluate_single_answer(
        self,
        prompt: str,
        answer: str,
        key: str,
        model: LLMJudge,
        response_schema: BongardEvaluationSchema,
    ) -> float:
        return model.evaluate_similarity(
            prompt=prompt, 
            answer=answer, 
            key=key, 
            response_schema=response_schema
        )

    def evaluate(
            self, 
            results_dir: str,
            answers_path: str, 
            key_path: str, 
            prompt: str = "",
            evaluation_output_path: str = "evaluation_results",
        ):
        if not answers_path or not os.path.exists(answers_path):
            logger.error("Answers path is not provided or does not exist.")
            return

        answers_df = pd.read_csv(answers_path, dtype={"problem_id": str})
        output_df = answers_df.copy()
        output_df["score"] = ""

        summary_answers = self.check_completeness(answers_df)
        logger.info(f"Answers DataFrame Completeness Summary: {summary_answers}")

        output_df = answers_df.copy()
        output_df["score"] = ""

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        summary_key = self.check_completeness(pd.Series(key_dict).to_frame())
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

            left_rule, right_rule = key_dict[id_].strip()
            key = f"{left_rule} vs. {right_rule}"

            score, reasoning = self.evaluate_single_answer(
                prompt=prompt if prompt else self.prompt,
                answer=answer,
                key=key,
                judge=LLMJudge(),
                schema=BongardEvaluationSchema,
            )

            output_df.at[index, "score"] = score
            output_df.at[index, "reasoning"] = reasoning

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

    def run_evaluation(
            self, 
            dataset_name,
            model_name,
            strategy_name,
            version,
            results_dir, 
            answers_path, 
            key_path, 
            evaluation_output_path = "evaluation_results", 
            concat = True, 
            output_all_results_concat_path = None
        ):
        if output_all_results_concat_path  is None:
            default_dir = results_dir.split("results")[0] + "results"
            output_all_results_concat_path  = default_dir / "all_results_concat.csv"

        self.evaluate(
            results_dir, 
            answers_path, 
            key_path, 
            evaluation_output_path)
        if concat:
            self.append_to_all_results_concat(
                dataset_name,
                model_name,
                strategy_name,
                version,
                pd.read_csv(f"{results_dir}/{evaluation_output_path}.csv"),
                output_all_results_concat_path
            )

    def calculate_metrics(self, evaluated_df):
        total = len(evaluated_df)
        correct = len(evaluated_df[evaluated_df["score"] == "Right"])
        accuracy = correct / total if total > 0 else 0.0

        bin_counts = evaluated_df.groupby("score")["confidence"].size().to_dict() if "score" in evaluated_df.columns else {}

        avg_confidence = evaluated_df.groupby("score")["confidence"].mean().to_dict() if "confidence" in evaluated_df.columns else {}
        median_confidence = evaluated_df.groupby("score")["confidence"].median().to_dict() if "confidence" in evaluated_df.columns else {}

        return {
            "total": total,
            "bin_counts": bin_counts,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "median_confidence": median_confidence
        }
        
