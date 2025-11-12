import pandas as pd

from code.evaluation.evaluationbase import EvaluationBase
from code.models.llm_judge import LLMJudge
from code.technical.response_schema import SimilarityResponseSchema


class EvaluationBP(EvaluationBase):

   def evaluate_single_answer(self, answer: str, key: str, model: LLMJudge, response_schema: SimilarityResponseSchema) -> float:
       return model.evaluate_similarity(answer, key, response_schema)

   def evaluate(self, answers_path, key_path, output_path):
       answers_df = pd.read_csv(answers_path)
       key_df = pd.read_csv(key_path)

