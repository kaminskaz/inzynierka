import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from code.strategies.strategybase import StrategyBase

from code.strategies.classificationstrategy import ClassificationStrategy
from code.strategies.contrastivestrategy import ContrastiveStrategy
from code.strategies.directstrategy import DirectStrategy
from code.strategies.descriptivestrategy import DescriptiveStrategy
from code.preprocessing.processorconfig import ProcessorConfig


class StrategyFactory:
    """
    Factory to create and configure a specific strategy based on its name.
    """
    
    def __init__(self, config_path: str = "code/preprocessing/dataset_config_test.json"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = Path(config_path)
        
        self.dataset_configs_raw: Dict[str, Any] = self._load_all_dataset_configs()
        
        # Map strategy names (as used in the command line) to their classes
        self.strategy_map: Dict[str, type[StrategyBase]] = {
            "classification": ClassificationStrategy,
            "contrastive": ContrastiveStrategy,
            "direct": DirectStrategy,
            "descriptive": DescriptiveStrategy,
            # Add more strategies here as they are created
        }
        self.logger.info(f"StrategyFactory initialized. {len(self.strategy_map)} strategies available.")

    def _load_all_dataset_configs(self) -> Dict[str, Any]:
        """Loads the main dataset_config.json file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load dataset config file at {self.config_path}: {e}")
            raise
            
    def _get_dataset_config(self, dataset_name: str) -> Optional[ProcessorConfig]:
        """Gets the ProcessorConfig for a specific dataset."""
        raw_config = self.dataset_configs_raw.get(dataset_name)
        if not raw_config:
            self.logger.error(f"No config found for dataset: '{dataset_name}' in {self.config_path}")
            return None
        
        try:
            return ProcessorConfig.from_dict(raw_config)
        except Exception as e:
            self.logger.error(f"Error creating ProcessorConfig for {dataset_name}: {e}")
            return None


    def create_strategy(self, dataset_name: str, strategy_name: str, model_object: Any, results_dir: str) -> StrategyBase:
        """
        Method to create, configure, and return a strategy instance.
        This is called by `run_single_experiment.py`.

        Args:
            dataset_name (str): The name of the dataset.
            strategy_name (str): The name of the strategy to use.
            model_object (Any): The instantiated model object (e.g., VLLM).
            results_dir (str): The path to the directory for saving results.
        """
        self.logger.info(f"Attempting to create strategy: '{strategy_name}' for dataset: '{dataset_name}' using model: '{model_object.get_model_name()}'")

        # get the Strategy Class
        strategy_class = self.strategy_map.get(strategy_name.lower())
        if not strategy_class:
            self.logger.error(f"Unknown strategy: '{strategy_name}'. Available: {list(self.strategy_map.keys())}")
            raise ValueError(f"Unknown strategy: '{strategy_name}'")
            
        # get the Dataset Config
        dataset_config = self._get_dataset_config(dataset_name)
        if not dataset_config:
            raise ValueError(f"Failed to load config for dataset: '{dataset_name}'")
        
        # initialize and return the strategy
        try:
            strategy_instance = strategy_class(
                dataset_name=dataset_name,
                model=model_object,
                dataset_config=dataset_config,
                results_dir=results_dir,
                strategy_name=strategy_name.lower()
            )
            self.logger.info(f"Successfully created: {strategy_class.__name__}")
            return strategy_instance
        
        except Exception as e:
            self.logger.error(f"Failed to instantiate {strategy_class.__name__}: {e}")
            raise