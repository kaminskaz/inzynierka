import pandas as pd
import json
import logging

from code.evaluation.evaluationbase import EvaluationBase

logger = logging.getLogger(__name__)


class EvaluationBasic(EvaluationBase):

    def evaluate_single_answer(self, answer: str, key: str) -> float:
        score = 1.0 if answer == key else 0.0
        return score

    def evaluate(self, answers_path, key_path, output_dir):
        answers_df = pd.read_csv(answers_path)
        output_df = answers_df.copy()
        output_df["score"] = 0.0

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        for index, row in answers_df.iterrows():
            answer = row["answer"]
            id_ = str(row["id"])

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        for index, row in answers_df.iterrows():
            answer = str(row["answer"]).strip().upper()
            id_ = str(row["id"]).zfill(3)

            if id_ not in key_dict:
                logger.info(f"ID {id_} not found in key file.")
                continue

            key = key_dict[id_].strip().upper()

            score = self.evaluate_single_answer(answer, key)
            output_df.at[index, "score"] = score

        output_path = f"{output_dir}/evaluation_results.csv"
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
