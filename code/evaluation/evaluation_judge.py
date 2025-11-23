import pandas as pd
import json
import logging

from code.evaluation.evaluation_base import EvaluationBase
from code.models.llm_judge import LLMJudge
from code.technical.response_schema import SimilarityResponseSchema

logger = logging.getLogger(__name__)


class EvaluationWithJudge(EvaluationBase):
    def __init__(self):
        super().__init__()
        try:
            with open("prompts/evaluation/evaluation_main.txt", "r") as file:
                self.prompt = file.read()
        except Exception as e:
            logger.error(f"Failed to read prompt file: {e}")

    def evaluate_single_answer(
        self,
        prompt: str,
        answer: str,
        key: str,
        model: LLMJudge,
        response_schema: SimilarityResponseSchema,
    ) -> float:
        return model.evaluate_similarity(
            prompt=prompt, 
            answer=answer, 
            key=key, 
            response_schema=response_schema
        )

    def evaluate(
            self, 
            answers_path: str, 
            key_path: str, 
            output_dir: str,
            prompt: str
        ):

        answers_df = pd.read_csv(answers_path)
        output_df = answers_df.copy()
        output_df["score"] = ""

        with open(key_path, "r") as f:
            key_dict = json.load(f)

        for index, row in answers_df.iterrows():
            left_rule_answer = row["left_side_rule"]
            right_rule_answer = row["right_side_rule"]
            id_ = str(row["id"])
            answer = f"Left side rule: {left_rule_answer}\nRight side rule: {right_rule_answer}"

            if id_ not in key_dict:
                logger.info(f"ID {id_} not found in key file.")
                continue

            left_rule, right_rule = key_dict[id_]
            key = f"Left side rule: {left_rule}\nRight side rule: {right_rule}"

            score = self.evaluate_single_answer(
                prompt=prompt,
                answer=answer,
                key=key,
                judge=LLMJudge(),
                schema=SimilarityResponseSchema,
            )

            output_df.at[index, "score"] = score

        output_path = f"{output_dir}/evaluation_results.csv"
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
