import argparse
import sys
import logging
import os
import re
from pathlib import Path
from itertools import product
from typing import Any, List, Optional

from code.strategies.strategy_factory import StrategyFactory
from code.models.vllm import VLLM
from code.technical.utils import get_dataset_config, make_dir_for_results
from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge


logger = logging.getLogger(__name__)

def run_multiple_evaluations(
        strategy_names: List[str],
        dataset_names: List[str],
        model_names: List[str],
        versions: List[str],
        judge_prompt: str = None,
        evaluation_output_path: str = "evaluation_results"
    ):
        # evaluator_judge = EvaluationWithJudge()
        evaluator_simple = EvaluationBasic()

        for d_name in dataset_names:
            d_category = get_dataset_config(d_name).category 
            for s_name, m_name, ver in product(strategy_names, model_names, versions):
                print(f"Evaluating dataset: {d_name} of category {d_category} for strategies: {s_name}")
                if d_category == "standard" or d_category == "choice_only":
                    evaluator = evaluator_simple
                    evaluator.run_evaluation(
                        dataset_name=d_name,
                        model_name=m_name,
                        strategy_name=s_name,
                        version=ver,
                        evaluation_output_path=evaluation_output_path,
                    )
                else:
                    # evaluator = evaluator_judge
                    # evaluator.run_evaluation(
                    #     dataset_name=d_name,
                    #     model_name=m_name,
                    #     strategy_name=s_name,
                    #     version=ver,
                    #     prompt=judge_prompt,
                    #     evaluation_output_path=evaluation_output_path,
                    # )
                    pass


def _load_model(
        model_name: str, 
        temperature: float = 1.0, 
        max_tokens: int = 2048, 
        max_output_tokens: int = 1536, 
        limit_mm_per_prompt: int = 2,
        custom_args: list = [],
        local_testing: bool = False
    ) -> Any:
    """
    Loads a VLLM model based on the provided model name and parameters.
    Currently supports VLLM models via VLLMFactory.
    """
    logger.info(f"Attempting to load model: '{model_name}'")

    try:
        vllm_model = VLLM(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            limit_mm_per_prompt=limit_mm_per_prompt,
            custom_args=custom_args,
            cpu_local_testing=local_testing
        )

        if vllm_model:
            return vllm_model
        else:
            return None

    except TimeoutError as e:
        logger.critical(
            f"Failed to start VLLM server for model '{model_name}'. "
            f"Pipeline execution cannot continue. Error: {e}"
        )
        return None

    except Exception as e:
        logger.error(f"An unexpected error occurred during VLLM setup for '{model_name}'. Error: {e}")
        return None


def check_data_preprocessed(dataset_name: str) -> bool:
    """
    Checks if the specified dataset appears to be preprocessed and in the
    standardized format required by the strategies.
    
    This checks for the existence of:
    - data/<dataset_name>/
    - data/<dataset_name>/problems/
    - data/<dataset_name>/jsons/
    - At least one .json file in the jsons/ directory.
    """
    logger.info(f"Checking for preprocessed data for dataset: {dataset_name}...")
    base_data_path = Path("data") / dataset_name
    problems_path = base_data_path / "problems"
    jsons_path = base_data_path / "jsons"

    if not base_data_path.exists():
        logger.error(f"Data directory not found: {base_data_path}")
        return False
    
    if not problems_path.exists():
        logger.error(f"Standardized 'problems' directory not found: {problems_path}")
        return False

    if not jsons_path.exists():
        logger.error(f"Standardized 'jsons' directory not found: {jsons_path}")
        return False
    
    if not any(jsons_path.glob("*.json")):
            logger.error(f"No JSON metadata files found in: {jsons_path}")
            return False

    logger.info(f"Found preprocessed data at: {base_data_path}")
    return True

def run_strategy_tests(model_name: str, 
        temperature: float, 
        max_tokens: int, 
        max_output_tokens: int, 
        limit_mm_per_prompt: int,
        custom_args: list = [],
        local_testing: bool = False
    ):
    
    model = _load_model(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            limit_mm_per_prompt=limit_mm_per_prompt,
            custom_args=custom_args,
            local_testing=local_testing
        )
    
    datasets = ['bp','cvr','raven','marsvqa']
    strategies = ['direct','descriptive','contrastive','classification']

    for d in datasets:
        for s in strategies:
            run_single_experiment(dataset_name=d, strategy_name=s, model_object=model)

    model.stop()


def run_single_experiment(
        dataset_name: str,
        strategy_name: str, 
        model_name: Optional[str] = None, 
        temperature: float=0.5, 
        max_tokens: int=2048, 
        max_output_tokens: int=1024, 
        limit_mm_per_prompt: int=2,
        custom_args: list = [],
        local_testing: bool = False,
        model_object: Optional[VLLM] = None
    ) -> None:
    """
    Initializes and runs a single experiment strategy.
    """
    logger.info(f"Creating strategy '{strategy_name}' for dataset '{dataset_name}' with model '{model_name}'")
    try:
        results_dir = make_dir_for_results(dataset_name, strategy_name, model_name)

        strategy_factory = StrategyFactory()

        if not model_object:
            model = _load_model(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_output_tokens=max_output_tokens,
                limit_mm_per_prompt=limit_mm_per_prompt,
                custom_args=custom_args,
                local_testing=local_testing
            )
        else:
            model = model_object
        
        strategy = strategy_factory.create_strategy(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_object=model,
            results_dir=results_dir
        )
        
        logger.info("Strategy created successfully. Running experiment...")
        strategy.run()
        logger.info(f"Experiment run complete for {dataset_name} / {strategy_name}.")

        model.stop()

    except ImportError as e:
        logger.error(f"Failed to create strategy. Does '{strategy_name}' exist and is it importable? Error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
        sys.exit(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single experiment')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use (same as in dataset_config.json)')
    parser.add_argument('--strategy', type=str, required=True, help='Name of the strategy to run')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for the experiment')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature setting for the model (if applicable)')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for the model (if applicable)')
    parser.add_argument('--max_output_tokens', type=int, default=1536, help='Maximum output tokens for the model (if applicable)')
    parser.add_argument('--limit_mm_per_prompt', type=int, default=2, help='Limit of multimodal inputs per prompt (if applicable)')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG logging level')
    parser.add_argument('--local_testing', help='Enable local CPU testing mode for VLLM models with limited resources')
    parser.add_argument('--custom_args', nargs=argparse.REMAINDER, default=[], help='List of custom arguments for the model (if applicable)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    if not check_data_preprocessed(args.dataset_name):
        logger.error(f"Data for '{args.dataset_name}' is not preprocessed or is missing.")
        logger.error("Please run the data preprocessing pipeline first.")
        sys.exit(1)


    # run_strategy_tests(
    #     model_name=args.model_name,
    #     temperature=args.temperature,
    #     max_tokens=args.max_tokens,
    #     max_output_tokens=args.max_output_tokens,
    #     limit_mm_per_prompt=args.limit_mm_per_prompt,
    #     custom_args=args.custom_args,
    #     local_testing=args.local_testing)

    run_single_experiment(
        dataset_name=args.dataset_name,
        strategy_name=args.strategy,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_output_tokens=args.max_output_tokens,
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        custom_args=args.custom_args,
        local_testing=args.local_testing
        )