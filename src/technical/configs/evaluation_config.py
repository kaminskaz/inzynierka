from dataclasses import dataclass
from typing import Optional

from src.models.llm_judge import LLMJudge

@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    dataset_name: str
    version: str

    strategy_name: Optional[str] = None
    model_name: Optional[str] = None
    ensemble: bool = False
    type_name: Optional[str] = None

    evaluation_output_path: str = "evaluation_results"
    concat: bool = True
    output_all_results_concat_path: str = "all_results_concat"

    judge_model_name: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.3"
    judge_model_object: Optional[LLMJudge] = None
    prompt_number: int = 1
