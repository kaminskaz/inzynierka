import argparse
import sys
import logging
from pathlib import Path
from typing import Any, Optional

from code.strategies.strategy_factory import StrategyFactory
from code.models.vllm import VLLM
from code.technical.utils import get_results_directory

logger = logging.getLogger(__name__)


def _load_model(
        model_name: str, 
        param_set_number: Optional[int] = None
    ) -> Any:
    """
    Loads a VLLM model based on the provided model name and parameters.
    Currently supports VLLM models via VLLMFactory.
    """
    logger.info(f"Attempting to load model: '{model_name}'")

    try:
        vllm_model = VLLM(
            model_name=model_name,
            param_set_number=param_set_number
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

def run_strategy_tests(model_name: str):
    
    model = _load_model(
            model_name=model_name,
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
        model_object: Optional[VLLM] = None,
        restart_problem_id: Optional[str] = None,
        restart_version: Optional[str] = None,
        param_set_number: Optional[int] = None,
        prompt_number: Optional[int]=1,
    ) -> None:
    """
    Initializes and runs a single experiment strategy.
    """
    logger.info(f"Creating strategy '{strategy_name}' for dataset '{dataset_name}' with model '{model_name}'")
    try:
        if restart_problem_id is not None:
            target_version = restart_version if restart_version else "latest"
            
            results_dir = get_results_directory(
                dataset_name=dataset_name, 
                strategy_name=strategy_name, 
                model_name=model_name, 
                version=target_version, 
                create_dir=False
            )
        else:
            results_dir = get_results_directory(dataset_name, strategy_name, model_name)

        strategy_factory = StrategyFactory()

        if not model_object:
            model = _load_model(
                model_name=model_name,
                param_set_number=param_set_number
            )
        else:
            model = model_object
        
        strategy = strategy_factory.create_strategy(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_object=model,
            results_dir=results_dir,
            param_set_number=param_set_number,
            prompt_number=prompt_number
        )
        
        logger.info("Strategy created successfully. Running experiment...")
        strategy.run(restart_problem_id=restart_problem_id)
        logger.info(f"Experiment run complete for {dataset_name} / {strategy_name}.")

        model.stop()

    except ImportError as e:
        logger.error(f"Failed to create strategy. Does '{strategy_name}' exist and is it importable? Error: {e}", exc_info=True)
        model.stop()
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
        model.stop()
        sys.exit(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single experiment')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use (same as in dataset_config.json)')
    parser.add_argument('--strategy', type=str, required=True, help='Name of the strategy to run')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for the experiment')
    parser.add_argument('--restart_problem_id', type=str, default=None, help='Problem ID to restart from (if applicable)')
    parser.add_argument('--restart_version', type=str, default=None, help='Version of the model-strategy-dataset combination to be restarted (if applicable)')
    parser.add_argument('--param_set_number', type=int, default=1, help='Parameter set number to use for the experiment (if applicable)')
    parser.add_argument('--prompt_number', type=int, default=1, help='Prompt number to use')
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

    run_single_experiment(
        dataset_name=args.dataset_name,
        strategy_name=args.strategy,
        model_name=args.model_name,
        restart_problem_id=args.restart_problem_id,
        restart_version = args.restart_version,
        param_set_number = args.param_set_number,
        prompt_number= args.prompt_number
        )