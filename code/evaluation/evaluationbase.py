from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel

from code.models.llm_judge import LLMJudge

class EvaluationBase(ABC):

    @abstractmethod
    def evaluate_single_answer(self, answer: str, key: str, model: Optional[LLMJudge], response_schema: Optional[Type[BaseModel]]) -> float:
        pass

    @abstractmethod
    def evaluate(self, answers_path, key_path, output_path):
        pass

