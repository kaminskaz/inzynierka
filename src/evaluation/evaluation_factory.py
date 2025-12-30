import logging
from typing import Any, Dict, Optional, Type, Tuple

from src.evaluation.evaluation_basic import EvaluationBasic
from src.evaluation.evaluation_judge import EvaluationWithJudge
from src.evaluation.evaluation_base import EvaluationBase
from src.models.llm_judge import LLMJudge

class EvaluationFactory():
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
        self.logger.info(
            f"EnsembleFactory initialized."
        )

    def which_evaluator(
        self,
        dataset_name: str,
        strategy_name: Optional[str] = None,
        ensemble: bool = False,
        type_name: Optional[str] = None
    ) -> Optional[Type[EvaluationBase]]:
        
        if dataset_name.lower() == "bp":
            if not ensemble and strategy_name is None:
                raise ValueError("strategy_name must be provided for non-ensemble evaluations.")
            if strategy_name is not None and strategy_name.lower() == "classification":
                return EvaluationBasic   
            else:
                return EvaluationWithJudge
      
        elif ensemble:
            if type_name is None:
                raise ValueError("type_name must be provided for ensemble evaluations.")
            if type_name.lower() in ["reasoning", "reasoning_with_image"]:
                return EvaluationWithJudge
        return EvaluationBasic

    def create_evaluator(
        self,
        dataset_name: str,
        ensemble: bool = False,
        strategy_name: Optional[str] = None,
        type_name: Optional[str] = None,
        judge_model_object: Optional[LLMJudge] = None,
        judge_model_name: Optional[str] = 'mistralai/Mistral-7B-Instruct-v0.3',
        prompt_number: Optional[int] = 1,
        prompt: Optional[str] = None,
        param_set_number: Optional[int] = None
    ) -> EvaluationBase:
        
        if ensemble and type_name is None:
            raise ValueError("type_name must be provided for ensemble evaluations.")
        if not ensemble and strategy_name is None:
            raise ValueError("strategy_name must be provided for non-ensemble evaluations.")
        
        evaluator_cls = self.which_evaluator(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            ensemble=ensemble,
            type_name=type_name
        )

        evaluator = evaluator_cls(judge_model_object=judge_model_object, judge_model_name=judge_model_name)

        if isinstance(evaluator, EvaluationWithJudge) and prompt_number is not None:
            evaluator.prompt_number = prompt_number
        if isinstance(evaluator, EvaluationWithJudge) and prompt is not None:
            evaluator.prompt = prompt
        if isinstance(evaluator, EvaluationWithJudge) and param_set_number is not None:
            evaluator.param_set_number = param_set_number
            
        return evaluator