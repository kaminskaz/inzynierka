import argparse
import logging
import os
import sys, subprocess
from typing import Any, List, Optional
from src.ensemble.ensemble_factory import EnsembleFactory
from src.evaluation.evaluation_base import EvaluationBase
from src.evaluation.evaluation_factory import EvaluationFactory
from src.models.llm_judge import LLMJudge
from src.models.vllm import VLLM
from src.preprocessing.data_module import DataModule
from src.strategies.strategy_factory import StrategyFactory
from src.technical.utils import get_results_directory, get_dataset_config, set_all_seeds, get_eval_config_from_path
from src.technical.configs.evaluation_config import EvaluationConfig
from src.visualisation.visualiser import StreamlitVisualiser
import pathlib


class FullPipeline: 
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._proc = None

    def prepare_data(
        self, 
        config_path: str = os.path.join("src", "technical", "configs", "dataset_config.json"),
        download: bool = False
    ) -> None:    
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
        seed: Optional[int] = 42
    ) -> None:
        if seed:
            set_all_seeds(seed)
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

            if model_object is None:
                model.stop()

        except Exception as e:
            self.logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
            if model is not None and hasattr(model, 'stop'):
                model.stop()
            raise e


    def run_ensemble(
        self,
        dataset_name: str,
        members_configuration: List[List[str]],
        type_name: str,
        vllm_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None, 
        model_object: Optional[VLLM] = None,
        prompt_number: Optional[int] = 1, 
        version: Optional[int] = None,
        seed: Optional[int] = 42
    ) -> None:
        if seed:
            set_all_seeds(seed)
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
                prompt_number=prompt_number,
                version=version,
                seed=seed
            )
            
            self.logger.info("Ensemble created successfully. Running ensemble...")
            ensemble.evaluate()
            self.logger.info(f"Ensemble run complete for {dataset_name} / {type_name}.")
            
            if model and model_object is None:
                model.stop()

        except ImportError as e:
            self.logger.error(f"Failed to create ensemble. Does '{type_name}' exist and is it importable? Error: {e}", exc_info=True)
            if model_object is None:
                model.stop()
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"An error occurred during the experiment run: {e}", exc_info=True)
            if model_object is None:
                model.stop()
            raise e

    def run_missing_evaluations_in_directory(
        self,
        path: str, 
        judge_model_name: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.3",
        param_set_number: Optional[int] = 1,
        judge_model_object: Optional[LLMJudge] = None,
        prompt_number: int = 1,
        seed: Optional[int] = 42
    ):

        root_path = pathlib.Path(path)
        if not str(root_path).startswith("results"):
            self.logger.info(f"Skipping: Path '{path}' must start with 'results/'.")
            return

        if not root_path.exists() or not root_path.is_dir():
            self.logger.info(f"Skipping: Path '{path}' does not exist or is not a directory.")
            return

        self.logger.info(f"Scanning for missing evaluations in: {root_path}")

        # .rglob('*') finds all files and directories recursively
        for subdir in root_path.rglob('*'):
        
            if subdir.is_dir() and ("ver" in subdir.name):

                subfolders = [f for f in subdir.iterdir() if f.is_dir()]
                if len(subfolders) > 0:
                    continue

                eval_file = subdir / "evaluation_results.csv"
                if eval_file.exists():
                    self.logger.info(f"Found existing results in {subdir}. Skipping.")
                    continue

                is_ensemble = "ensemble" in str(subdir).lower()
                
                try:
                    config = self.get_eval_config_from_path(
                        path=str(subdir),
                        ensemble=is_ensemble,
                        judge_model_name=judge_model_name,
                        judge_model_object=judge_model_object,
                        prompt_number=prompt_number,
                        param_set_number=param_set_number
                    )
        
                    self.logger.info(f"Running evaluation for: {subdir}")
                    self.run_evaluation(config=config, seed=seed)
                    
                except ValueError as e:
                    self.logger.info(f"Could not parse config for {subdir}: {e}")

    def run_evaluation(
        self, 
        config: EvaluationConfig,
        evaluator: Optional[EvaluationBase] = None,
        seed: Optional[int] = 42
    ) -> None:
        if seed:
            set_all_seeds

        if evaluator is not None:
            self.logger.info("Using provided evaluator instance.")
            stop_after_evaluation = False
        else:
            self.logger.info("Creating evaluator instance from configuration.")
            stop_after_evaluation = True
            eval_factory = EvaluationFactory()
            
            evaluator = eval_factory.create_evaluator(
                dataset_name=config.dataset_name,
                ensemble=config.ensemble,
                strategy_name=config.strategy_name,
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

        if stop_after_evaluation and evaluator.judge_model_object is not None:
            evaluator.judge_model_object.stop()

    def run_evaluations(
        self, 
        configs: List[EvaluationConfig]
    ) -> None:
        for config in configs:
            self.run_evaluation(config)

    def visualise(
        self, 
        csv_path: str = os.path.join("results", "all_results_concat.csv")
    ) -> None:
        visualiser_path = os.path.join("src", "visualisation", "visualiser.py")
        self._proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", visualiser_path, "--", csv_path])
        self.logger.info(f"Streamlit visualiser started with PID: {self._proc.pid}")

    def stop_visualiser(self):
        if self._proc and self._proc.poll() is None:
            self.logger.info(f"Terminating Streamlit visualiser with PID: {self._proc.pid}")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Streamlit visualiser did not terminate in time. Killing process.")
                self._proc.kill()
            self.logger.info("Streamlit visualiser terminated.")

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
        
    def check_data_preprocessed(
            self,
            dataset_name: str
        ) -> bool:
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
