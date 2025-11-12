import pandas as pd
import json
import logging

from code.evaluation.evaluationbase import EvaluationBase
from code.models.llm_judge import LLMJudge
from code.technical.response_schema import SimilarityResponseSchema

logger = logging.getLogger(__name__)


class EvaluationWithJudge(EvaluationBase):

    def evaluate_single_answer(
        self,
        answer: str,
        key: str,
        model: LLMJudge,
        response_schema: SimilarityResponseSchema,
    ) -> float:
        return model.evaluate_similarity(answer, key, response_schema)

    def evaluate(self, answers_path, key_path, output_dir):
        answers_df = pd.read_csv(answers_path)
        output_df = answers_df.copy()
        output_df["score"] = 0.0

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        for index, row in answers_df.iterrows():
            answer = row["answer"]
            id_ = str(row["id"])

            if id_ not in key_dict:
                logger.info(f"ID {id_} not found in key file.")
                continue

            left_rule, right_rule = key_dict[id_]
            key = f"Left side rule: {left_rule}\nRight side rule: {right_rule}"

            score = self.evaluate_single_answer(
                answer=answer,
                key=key,
                judge=LLMJudge(),
                schema=SimilarityResponseSchema(),
            )

            output_df.at[index, "score"] = score

        output_path = f"{output_dir}/evaluation_results.csv"
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
