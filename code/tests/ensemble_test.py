import argparse
import json
import sys
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from code.models.llm_judge import LLMJudge
from code.strategies.strategy_factory import StrategyFactory
from code.ensemble.ensemble_factory import EnsembleFactory
from code.models.vllm import VLLM
from code.technical.utils import get_dataset_config, make_dir_for_results
from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge


logger = logging.getLogger(__name__)

def json_list(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("members_configuration must be valid JSON")

def run_single_ensemble(
        dataset_name: str,
        members_configuration: List[List[str]],
        type_name: str,
        vllm_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None, 
        model_object: Optional[Any] = None
    ) -> None:
    """
    Initializes and runs a single experiment strategy.
    """
    logger.info(f"Creating ensemble '{type_name}' for dataset '{dataset_name}' with members: {members_configuration}')")
    try:
        ensemble_factory = EnsembleFactory()

        if not model_object:
            if type_name == "reasoning_with_image" and vllm_model_name:
                logger.info(f"Initializing VLLM model '{vllm_model_name}' for reasoning with image ensemble.")
                model = VLLM(
                    model_name=vllm_model_name
                )

            elif (get_dataset_config(dataset_name).category == "BP" and llm_model_name) or (type_name == "reasoning" and llm_model_name):
                logger.info(f"Initializing LLM model '{llm_model_name}' for ensemble.")
                
                model = LLMJudge(
                    model_name=llm_model_name
                )
            else:
                model = None
        else:
            model = model_object
        
        ensemble = ensemble_factory.create_ensemble(
            dataset_name=dataset_name,
            members_configuration=members_configuration,
            skip_missing=True,
            judge_model=model,
            type_name=type_name
        )
        
        logger.info("Ensemble created successfully. Running ensemble...")
        ensemble.evaluate()
        logger.info(f"Ensemble run complete for {dataset_name} / {type_name}.")
        
        if model:
            model.stop()

    except ImportError as e:
        logger.error(f"Failed to create ensemble. Does '{type_name}' exist and is it importable? Error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
        sys.exit(1)
    

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single experiment')

    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='Name of the dataset to use (same as in dataset_config.json)')
    parser.add_argument('--members_configuration', type=json_list, required=True, 
                        help='Configuration string/file for the ensemble members')
    parser.add_argument('--ensemble_type', type=str, required=True, 
                        help='The type of ensemble method to apply')
    parser.add_argument('--vllm_model_name', type=str, required=False, 
                        help='Name of the VLLM model (e.g., OpenGVLab/InternVL3-8B)')
    parser.add_argument('--llm_model_name', type=str, required=False, 
                        help='Name of the LLM model (e.g., mistralai/Mistral-7B-Instruct-v0.3)')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature setting for the model')
    parser.add_argument('--max_tokens', type=int, default=2048, 
                        help='Maximum context tokens')
    parser.add_argument('--max_output_tokens', type=int, default=1536, 
                        help='Maximum output tokens')
    parser.add_argument('--limit_mm_per_prompt', type=int, default=2, 
                        help='Limit of multimodal inputs per prompt')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable DEBUG logging level')
    parser.add_argument('--local_testing', action='store_true',
                        help='Enable local CPU testing mode')
    parser.add_argument('--custom_args', nargs=argparse.REMAINDER, default=[], 
                        help='List of custom arguments passed to the underlying model')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_single_ensemble(
        dataset_name=args.dataset_name,
        members_configuration=args.members_configuration,
        type_name=args.ensemble_type,
        vllm_model_name=args.vllm_model_name if args.vllm_model_name else None,
        llm_model_name=args.llm_model_name if args.llm_model_name else None,
    )