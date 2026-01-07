import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.ensemble.ensemble_base import EnsembleBase

from src.ensemble.majority_ensemble import MajorityEnsemble
from src.ensemble.confidence_ensemble import ConfidenceEnsemble
from src.ensemble.reasoning_ensemble import ReasoningEnsemble
from src.ensemble.reasoning_ensemble_with_image import ReasoningEnsembleWithImage
from src.models.llm_judge import LLMJudge

class EnsembleFactory:
    """
    Factory to create and configure a specific ensemble based on its name.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Map ensembles names (as used in the command line) to their classes
        self.ensemble_map: Dict[str, type[EnsembleBase]] = {
            "majority": MajorityEnsemble,
            "confidence": ConfidenceEnsemble,
            "reasoning": ReasoningEnsemble,
            "reasoning_with_image": ReasoningEnsembleWithImage,
            # Add more ensembles here as they are created
        }
        self.logger.info(
            f"EnsembleFactory initialized. {len(self.ensemble_map)} ensembles available."
        )


    def create_ensemble(
        self, 
        dataset_name: str, 
        members_configuration: List[List[str]], 
        skip_missing: bool = True, 
        judge_model: Optional[LLMJudge] = None, 
        type_name: str = "majority", 
        prompt_number: int = 1,
        seed: int = 42,
        version: int = None
    ) -> EnsembleBase:
        """
        Method to create, configure, and return an ensemble instance.
        This is called by `run_single_experiment.py`.

        Args:
            dataset_name (str): The name of the dataset.
            members_configuration (List[List[str]]): Configuration of ensemble members. (strategy, model, version)
            skip_missing (bool): Whether to run missing experiments.
            judge_model (Optional[LLMJudge]): The instantiated judge model object (e.g., LLM, VLLM).
        """
        self.logger.info(
            f"Attempting to create ensemble for dataset: '{dataset_name}' using members: '{members_configuration}'"
        )

        ensemble_class = self.ensemble_map.get(type_name.lower())
        if not ensemble_class:
            self.logger.error(
                f"Unknown ensemble type: '{type_name}'. Available: {list(self.ensemble_map.keys())}"
            )
            raise ValueError(f"Unknown ensemble type: '{type_name}'")

        # initialize and return the ensemble
        try:
            ensemble_instance = ensemble_class(
                dataset_name=dataset_name,
                members_configuration=members_configuration,
                skip_missing=skip_missing,
                judge_model=judge_model, 
                type_name=type_name,
                prompt_number=prompt_number,
                version=version,
                seed=seed
            )
            self.logger.info(f"Successfully created: {ensemble_class.__name__}")
            return ensemble_instance

        except Exception as e:
            self.logger.error(f"Failed to instantiate {ensemble_class.__name__}: {e}")
            raise
