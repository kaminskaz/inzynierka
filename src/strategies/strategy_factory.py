import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.strategies.strategy_base import StrategyBase

from src.strategies.classification_strategy import ClassificationStrategy
from src.strategies.contrastive_strategy import ContrastiveStrategy
from src.strategies.direct_strategy import DirectStrategy
from src.strategies.descriptive_strategy import DescriptiveStrategy
from src.technical.utils import get_dataset_config

class StrategyFactory:
    """
    Factory to create and configure a specific strategy based on its name.
    """
    def __init__(
        self,
        config_path: str = os.path.join("src", "technical", "configs", "dataset_config.json")
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = Path(config_path)

        # Map strategy names (as used in the command line) to their classes
        self.strategy_map: Dict[str, type[StrategyBase]] = {
            "classification": ClassificationStrategy,
            "contrastive": ContrastiveStrategy,
            "direct": DirectStrategy,
            "descriptive": DescriptiveStrategy,
            # Add more strategies here as they are created
        }
        self.logger.info(
            f"StrategyFactory initialized. {len(self.strategy_map)} strategies available."
        )


    def create_strategy(
        self, dataset_name: str, strategy_name: str, model_object: Any, results_dir: str, param_set_number: Optional[int] = None, prompt_number: Optional[int] = 1
    ) -> StrategyBase:
        """
        Method to create, configure, and return a strategy instance.
        This is called by `run_single_experiment.py`.

        Args:
            dataset_name (str): The name of the dataset.
            strategy_name (str): The name of the strategy to use.
            model_object (Any): The instantiated model object (e.g., VLLM).
            results_dir (str): The path to the directory for saving results.
        """
        self.logger.info(
            f"Attempting to create strategy: '{strategy_name}' for dataset: '{dataset_name}' using model: '{model_object.get_model_name()}'"
        )

        # get the Strategy Class
        strategy_class = self.strategy_map.get(strategy_name.lower())
        if not strategy_class:
            self.logger.error(
                f"Unknown strategy: '{strategy_name}'. Available: {list(self.strategy_map.keys())}"
            )
            raise ValueError(f"Unknown strategy: '{strategy_name}'")

        # get the Dataset Config
        dataset_config = get_dataset_config(dataset_name, self.config_path)
        if not dataset_config:
            raise ValueError(f"Failed to load config for dataset: '{dataset_name}'")

        # initialize and return the strategy
        try:
            strategy_instance = strategy_class(
                dataset_name=dataset_name,
                model=model_object,
                dataset_config=dataset_config,
                results_dir=results_dir,
                strategy_name=strategy_name.lower(),
                param_set_number=param_set_number,
                prompt_number=prompt_number
            )
            self.logger.info(f"Successfully created: {strategy_class.__name__}")
            return strategy_instance

        except Exception as e:
            self.logger.error(f"Failed to instantiate {strategy_class.__name__}: {e}")
            raise
