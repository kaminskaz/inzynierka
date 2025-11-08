import argparse
import sys
import logging
from pathlib import Path
from code.strategies.strategyfactory import StrategyFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def run_single_experiment(dataset_name: str, strategy_name: str, model_name: str) -> None:
    """
    Initializes and runs a single experiment strategy.
    """
    logger.info(f"Creating strategy '{strategy_name}' for dataset '{dataset_name}' with model '{model_name}'")
    try:
        strategy_factory = StrategyFactory()
        
        strategy = strategy_factory.create_strategy(
            dataset_name=dataset_name,
            strategy_name=strategy_name,
            model_name=model_name 
        )
        
        logger.info("Strategy created successfully. Running experiment...")
        strategy.run()
        logger.info(f"Experiment run complete for {dataset_name} / {strategy_name}.")

        strategy_factory._stop_model()

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
    args = parser.parse_args()

    if not check_data_preprocessed(args.dataset_name):
        logger.error(f"Data for '{args.dataset_name}' is not preprocessed or is missing.")
        logger.error("Please run the data preprocessing pipeline first.")
        sys.exit(1)

    run_single_experiment(
        dataset_name=args.dataset_name,
        strategy_name=args.strategy,
        model_name=args.model,
        
    )