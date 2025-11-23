from abc import ABC, abstractmethod
from typing import Optional, Type
from pydantic import BaseModel

from code.models.llm_judge import LLMJudge


class EvaluationBase(ABC):

    @abstractmethod
    def evaluate_single_answer(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def evaluate(self, answers_path, key_path, output_path):
        pass
