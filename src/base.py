import argparse
import logging
import os
import sys
from typing import Any, List, Optional
from src.ensemble.ensemble_factory import EnsembleFactory
from src.evaluation.evaluation_factory import EvaluationFactory
from src.models.llm_judge import LLMJudge
from src.models.vllm import VLLM
from src.preprocessing.data_module import DataModule
from src.strategies.strategy_factory import StrategyFactory
from src.technical.utils import get_results_directory, get_dataset_config
from src.technical.configs.evaluation_config import EvaluationConfig


class FullPipeline: 
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def prepare_data(
            self, 
            config_path: str = os.path.join("src", "technical", "configs", "dataset_config.json"),
            download: bool = False):    
        data_module = DataModule(
            config_path=config_path,
            load_from_hf=download
        )
        
        data_module.run()
    
    def run_experiment(
            self,
            dataset_name: str,
            strategy_name: str, 
            model_name: str, 
            model_object: Optional[VLLM] = None,
            restart_problem_id: Optional[str] = None,
            restart_version: Optional[str] = None,
            param_set_number: Optional[int] = None,
            prompt_number: Optional[int]=1,
        ) -> None:
        """
        Initializes and runs a single experiment strategy.
        """
        self.logger.info(f"Creating strategy '{strategy_name}' for dataset '{dataset_name}' with model '{model_name}'")
        
        try:
            target_version = restart_version if (restart_version and restart_version.strip()) else "latest"
                
            results_dir = get_results_directory(
                dataset_name=dataset_name, 
                strategy_name=strategy_name, 
                model_name=model_name, 
                version=target_version, 
                create_dir=True
            )

            strategy_factory = StrategyFactory()

            model = model_object 
            if not model:
                model = self._load_model(
                    model_name=model_name,
                    param_set_number=param_set_number
                )
            
            if model is None:
                raise RuntimeError(f"Failed to initialize model: {model_name}")
            
            strategy = strategy_factory.create_strategy(
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                model_object=model,
                results_dir=results_dir,
                param_set_number=param_set_number,
                prompt_number=prompt_number
            )
            
            self.logger.info("Strategy created successfully. Running experiment...")
            strategy.run(restart_problem_id=restart_problem_id)
            self.logger.info(f"Experiment run complete for {dataset_name} / {strategy_name}.")

            model.stop()

        except Exception as e:
            self.logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
            if model is not None and hasattr(model, 'stop'):
                model.stop()
            sys.exit(3)


    def run_ensemble(
        self,
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
        self.logger.info(f"Creating ensemble '{type_name}' for dataset '{dataset_name}' with members: {members_configuration}')")
        try:
            ensemble_factory = EnsembleFactory()

            if not model_object:
                if type_name == "reasoning_with_image" and vllm_model_name:
                    self.logger.info(f"Initializing VLLM model '{vllm_model_name}' for reasoning with image ensemble.")
                    model = VLLM(
                        model_name=vllm_model_name
                    )

                elif (get_dataset_config(dataset_name).category == "BP" and llm_model_name) or (type_name == "reasoning" and llm_model_name):
                    self.logger.info(f"Initializing LLM model '{llm_model_name}' for ensemble.")
                    
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
            
            self.logger.info("Ensemble created successfully. Running ensemble...")
            ensemble.evaluate()
            self.logger.info(f"Ensemble run complete for {dataset_name} / {type_name}.")
            
            if model:
                model.stop()

        except ImportError as e:
            self.logger.error(f"Failed to create ensemble. Does '{type_name}' exist and is it importable? Error: {e}", exc_info=True)
            model.stop()
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
            model.stop()
            sys.exit(1)


    def run_evaluation(
            self, 
            config: EvaluationConfig
        ):

        eval_factory = EvaluationFactory()
        
        evaluator = eval_factory.create_evaluator(
            dataset_name=config.dataset_name,
            ensemble=config.ensemble,
            type_name=config.type_name,
            judge_model_object=config.judge_model_object,
            judge_model_name=config.judge_model_name,
            prompt_number=config.prompt_number
        )
        evaluator.run_evaluation(
            dataset_name=config.dataset_name,
            version=config.version, 
            strategy_name=config.strategy_name, 
            model_name=config.model_name, 
            ensemble=config.ensemble,
            type_name=config.type_name,
            evaluation_output_path=config.evaluation_output_path,
            concat=config.concat,
            output_all_results_concat_path=config.output_all_results_concat_path
        )

    def run_evaluations(self, configs: List[EvaluationConfig]):
        for config in configs:
            self.run_evaluation(config)

    def visualise(self):
        pass

    def _load_model(
        self,
        model_name: str, 
        param_set_number: Optional[int] = None
    ) -> Any:

        self.logger.info(f"Attempting to load model: '{model_name}'")

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
            self.logger.critical(
                f"Failed to start VLLM server for model '{model_name}'. "
                f"Pipeline execution cannot continue. Error: {e}"
            )
            return None

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during VLLM setup for '{model_name}'. Error: {e}")
            return None
        
    def check_data_preprocessed(self, dataset_name: str) -> bool:
        """
        Checks if the specified dataset appears to be preprocessed and in the
        standardized format required by the strategies.
        
        This checks for the existence of:
        - data/<dataset_name>/
        - data/<dataset_name>/problems/
        - data/<dataset_name>/jsons/
        - At least one .json file in the jsons/ directory.
        """
        self.logger.info(f"Checking for preprocessed data for dataset: {dataset_name}...")
        base_data_path = os.path.join("data", dataset_name)
        problems_path = os.path.join(base_data_path, "problems")
        jsons_path = os.path.join(base_data_path, "jsons")

        if not os.path.exists(base_data_path):
            self.logger.error(f"Data directory not found: {base_data_path}")
            return False
        
        if not os.path.exists(problems_path):
            self.logger.error(f"Standardized 'problems' directory not found: {problems_path}")
            return False

        if not os.path.exists(jsons_path):
            self.logger.error(f"Standardized 'jsons' directory not found: {jsons_path}")
            return False
        
        if not any(fname.endswith(".json") for fname in os.listdir(jsons_path)):
                self.logger.error(f"No JSON metadata files found in: {jsons_path}")
                return False

        self.logger.info(f"Found preprocessed data at: {base_data_path}")
        return True
