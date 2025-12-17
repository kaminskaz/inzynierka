import logging
from typing import Any, Dict, Optional, Type, Tuple

from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.evaluation.evaluation_base import EvaluationBase

class EvaluationFactory():
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
        self.logger.info(
            f"EnsembleFactory initialized."
        )

    def which_evaluator(
        self,
        dataset_name: str,
        ensemble: bool = False,
        type_name: Optional[str] = None
    ) -> Optional[Type[EvaluationBase]]:
        
        if dataset_name.lower() == "bp":
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
        type_name: Optional[str] = None,
        model_object: Optional[Any] = None,
        model_name: Optional[str] = 'mistralai/Mistral-7B-Instruct-v0.3',
        prompt_number: Optional[int] = 1
    ) -> EvaluationBase:
        
        if ensemble and type_name is None:
            raise ValueError("type_name must be provided for ensemble evaluations.")
        
        evaluator_cls = self.which_evaluator(
            dataset_name=dataset_name,
            ensemble=ensemble,
            type_name=type_name
        )

        # bez sensu może ale nie wiem jak to sensowniej tutaj dodać
        evaluator = evaluator_cls(model_object=model_object, model_name=model_name)

        if isinstance(evaluator, EvaluationWithJudge) and prompt_number is not None:
            evaluator.prompt_number = prompt_number
            
        return evaluator