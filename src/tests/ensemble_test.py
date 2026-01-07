import argparse
import json
import sys
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from src.models.llm_judge import LLMJudge
from src.strategies.strategy_factory import StrategyFactory
from src.ensemble.ensemble_factory import EnsembleFactory
from src.models.vllm import VLLM
from src.technical.utils import get_dataset_config, get_results_directory
from src.evaluation.evaluation_basic import EvaluationBasic
from src.evaluation.evaluation_judge import EvaluationWithJudge
import os
import gc
import time
import torch
from src.base import FullPipeline
from src.models.vllm import VLLM
from src.technical.exceptions import PipelineCriticalError


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
        model_object: Optional[VLLM] = None,
        prompt_number: Optional[int] = 1
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
            type_name=type_name,
            prompt_number=prompt_number
        )
        
        logger.info("Ensemble created successfully. Running ensemble...")
        ensemble.evaluate()
        logger.info(f"Ensemble run complete for {dataset_name} / {type_name}.")
        
        if model:
            model.stop()

    except ImportError as e:
        logger.error(f"Failed to create ensemble. Does '{type_name}' exist and is it importable? Error: {e}", exc_info=True)
        if model_object is None:
            model.stop()
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
        if model_object is None:
            model.stop()
        raise e
    
def run_all_ensembles(dataset):
    pipeline = FullPipeline()

    ### top overall:
    top_members_overall = [['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['direct', 'OpenGVLab/InternVL3-8B', '1']]

    ### strategy ensembles:
    direct_ens = [['direct', 'OpenGVLab/InternVL3-8B', '1'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'llava-hf/llava-v1.6-mistral-7b-hf', '1']]
    descr_ens = [['descriptive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['descriptive', 'llava-hf/llava-v1.6-mistral-7b-hf', '3'], ['descriptive', 'OpenGVLab/InternVL3-8B', '1']]
    contrast_ens = [['contrastive', 'OpenGVLab/InternVL3-8B', '3'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['contrastive', 'llava-hf/llava-v1.6-mistral-7b-hf', '3']]
    classif_ens = [['classification', 'OpenGVLab/InternVL3-8B', '1'], ['classification', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['classification', 'llava-hf/llava-v1.6-mistral-7b-hf', '3']]

    ### model_ensembles
    qwen_ens = [['classification', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['descriptive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3']]
    llava_ens = [['classification', 'llava-hf/llava-v1.6-mistral-7b-hf', '3'], ['contrastive', 'llava-hf/llava-v1.6-mistral-7b-hf', '3'], ['descriptive', 'llava-hf/llava-v1.6-mistral-7b-hf', '3'], ['direct', 'llava-hf/llava-v1.6-mistral-7b-hf', '1']]
    intern_ens = [['classification', 'OpenGVLab/InternVL3-8B', '1'], ['contrastive', 'OpenGVLab/InternVL3-8B', '3'], ['descriptive', 'OpenGVLab/InternVL3-8B', '1'], ['direct', 'OpenGVLab/InternVL3-8B', '1']]

    if dataset == "bp":
        top_members_dataset = [['direct', 'OpenGVLab/InternVL3-8B', '1'], ['direct', 'OpenGVLab/InternVL3-8B', '3'], ['descriptive', 'OpenGVLab/InternVL3-8B', '1'], ['descriptive', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '1']]

    elif dataset == "cvr":
        top_members_dataset = [['classification', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'OpenGVLab/InternVL3-8B', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['classification', 'OpenGVLab/InternVL3-8B', '3']]

    elif dataset == "raven":
        top_members_dataset = [['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['contrastive', 'OpenGVLab/InternVL3-8B', '3']]

    elif dataset == "marsvqa":
        top_members_dataset = [['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['descriptive', 'OpenGVLab/InternVL3-8B', '1']]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    configurations = [top_members_overall, top_members_dataset, qwen_ens, llava_ens, intern_ens, direct_ens, descr_ens, contrast_ens]
    if dataset != "bp":
        configurations.append(classif_ens)

    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
    type_names = ["majority","confidence","reasoning","reasoning_with_image"]

    for config in configurations:
        model = VLLM(model_name)
        for type_name in type_names:
            pipeline.run_ensemble(dataset_name=dataset, members_configuration=config, type_name=type_name, model_object=model)
        model.stop()

def seed_and_version_test(dataset):
    pipeline = FullPipeline()
    ### top overall:
    top_members_overall = [['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '3'], ['direct', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['contrastive', 'Qwen/Qwen2.5-VL-7B-Instruct', '1'], ['direct', 'OpenGVLab/InternVL3-8B', '1']]
    configurations = [top_members_overall]

    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
    type_names = ["majority"]
    versions = [100, 101, 102]
    seeds = [100, 100, 101]

    for ver, seed in zip(versions, seeds):
        for config in configurations:
            if dataset == "bp":
                model = VLLM(model_name)
            else:
                model = None
            for type_name in type_names:
                pipeline.run_ensemble(dataset_name=dataset, members_configuration=config, type_name=type_name, model_object=model, version=ver, seed=seed)
            if model:
                model.stop()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single ensemble experiment')
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help='Name of the dataset to use (same as in dataset_config.json)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    dataset = args.dataset_name
    
    #run_all_ensembles(dataset)
    seed_and_version_test(dataset)