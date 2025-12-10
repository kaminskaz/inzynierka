import logging
from typing import Any, Dict, Optional, Type, Tuple

from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.evaluation.evaluation_base import EvaluationBase

class EvaluationFactory():
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.evaluator_map: dict[Tuple[Optional[str], Optional[bool], Optional[str]], Type[EvaluationBase]] = {
            # single model evaluations
            ("bp", False, None): EvaluationWithJudge,               
            ("raven", False, None): EvaluationBasic,                
            ("cvr", False, None): EvaluationBasic, 
            ("marsvqa", False, None): EvaluationBasic,                
            # ensemble evaluations
            ("bp", True, "majority"): EvaluationWithJudge,               
            ("raven", True, "majority"): EvaluationBasic,                
            ("cvr", True, "majority"): EvaluationBasic,
            ("marsvqa", True, "majority"): EvaluationBasic,
            ("bp", True, "confidence"): EvaluationWithJudge,               
            ("raven", True, "confidence"): EvaluationBasic,                
            ("cvr", True, "confidence"): EvaluationBasic,
            ("marsvqa", True, "confidence"): EvaluationBasic,
            ("bp", True, "reasoning"): EvaluationWithJudge,               
            ("raven", True, "reasoning"): EvaluationWithJudge,                        
            ("cvr", True, "reasoning"): EvaluationWithJudge,   
            ("marsvqa", True, "reasoning"): EvaluationWithJudge,   
            ("bp", True, "reasoning_with_image"): EvaluationWithJudge,               
            ("raven", True, "reasoning_with_image"): EvaluationWithJudge,                        
            ("cvr", True, "reasoning_with_image"): EvaluationWithJudge, 
            ("marsvqa", True, "reasoning_with_image"): EvaluationWithJudge,          

        }
    
        self.logger.info(
            f"EnsembleFactory initialized. {len(self.evaluator_map)} ensembles available."
        )

    def create_evaluator(
        self,
        dataset_name: str,
        ensemble: bool = False,
        type_name: Optional[str] = None,
        model_object: Optional[Any] = None,
        model_name: Optional[str] = None
    ) -> EvaluationBase:
        
        if ensemble and type_name is None:
            raise ValueError("type_name must be provided for ensemble evaluations.")
        
        key = (dataset_name.lower(), ensemble, type_name.lower() if type_name else None)
        evaluator_cls = self.evaluator_map.get(key)

        if not evaluator_cls:
            self.logger.warning(f"No evaluator found for {key}, using default EvaluationBasic")
            evaluator_cls = EvaluationBasic

        return evaluator_cls(model_object=model_object, model_name=model_name)