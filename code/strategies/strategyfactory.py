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
import PIL.Image # Used for the placeholder model object


class StrategyFactory:
    """
    Factory to create and configure a specific strategy based on its name.
    """
    
    def __init__(self, config_path: str = "code/preprocessing/dataset_config.json"):
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

    def _load_model(self, model_name: str) -> PIL.Image.Image:
        """
        --- PLACEHOLDER MODEL LOADING LOGIC ---
        This function should contain the logic to load the model using vllm
        
        NOTE: Your `StrategyBase` __init__ currently expects a `Any`
        object for the model. This is likely a placeholder. When you
        update your model logic, you should pass the *actual model object* here.
        
        """
        self.logger.warning(
            f"--- Using PLACEHOLDER model loader for '{model_name}'. ---"
        )
        # TODO: Karolina - Add your model loading logic here.
        pass

    def create_strategy(self, dataset_name: str, strategy_name: str, model_name: str) -> StrategyBase:
        """
        Method to create, configure, and return a strategy instance.
        This is called by `run_single_experiment.py`.
        """
        self.logger.info(f"Attempting to create strategy: '{strategy_name}' for dataset: '{dataset_name}' using model: '{model_name}'")
        
        # get the Strategy Class
        strategy_class = self.strategy_map.get(strategy_name.lower())
        if not strategy_class:
            self.logger.error(f"Unknown strategy: '{strategy_name}'. Available: {list(self.strategy_map.keys())}")
            raise ValueError(f"Unknown strategy: '{strategy_name}'")
            
        # get the Dataset Config
        dataset_config = self._get_dataset_config(dataset_name)
        if not dataset_config:
            raise ValueError(f"Failed to load config for dataset: '{dataset_name}'")
            
        # load the Model (using the placeholder)
        model_object = self._load_model(model_name)
        
        # initialize and return the strategy
        try:
            strategy_instance = strategy_class(
                dataset_name=dataset_name,
                model=model_object,
                dataset_config=dataset_config
            )
            self.logger.info(f"Successfully created: {strategy_class.__name__}")
            return strategy_instance
        except Exception as e:
            self.logger.error(f"Failed to instantiate {strategy_class.__name__}: {e}")
            raise