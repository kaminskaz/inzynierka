import logging
from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional
import PIL.Image
import os
import csv
import json

from code.preprocessing.processorconfig import ProcessorConfig 
from code.models.vllm import VLLM

class StrategyBase(ABC):
    def __init__(self, dataset_name: str, model: VLLM, dataset_config: ProcessorConfig, results_dir: str):
        self.dataset_name: str = dataset_name
        self.model: VLLM = model
        self.config: ProcessorConfig = dataset_config
        self.strategy_name: str = self.__class__.__name__
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized strategy for dataset: '{self.dataset_name}'")
        self.results_dir = results_dir

    @abstractmethod
    def run_single_problem(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def get_prompt(self, prompt_type: str) -> str:
        try:
            repo_root = os.path.dirname(os.path.abspath(__file__))  

            if prompt_type == "problem_description_main":
                prompt_path = os.path.join(
                    repo_root,
                    "prompts",
                    self.dataset_name,
                    f"{prompt_type}.txt"
                )
            else:
                prompt_path = os.path.join(
                    repo_root,
                    "prompts",
                    self.dataset_name,
                    self.strategy_name,
                    f"{prompt_type}.txt"
                )
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            return prompt
        
        except Exception as e:
            self.logger.exception(
                f"Error reading prompt for strategy '{self.strategy_name}' and dataset '{self.dataset_name}': {e}"
            )
            return ""

    def save_metadata(self, question_prompt:str, problem_description_prompt:str, describe_prompt:Optional[str]=None) -> None:
        """Save dataset, strategy, model, and config info into a metadata file."""
        try:
            metadata_path = os.path.join(self.results_dir, "metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Strategy: {self.strategy_name}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Config: {self.config}\n")
                f.write(f"Problem description prompt: {problem_description_prompt}\n")
                f.write(f"Question prompt: {question_prompt}\n")
                f.write(f"Describe prompt: {describe_prompt}\n")
            self.logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            self.logger.exception(f"Failed to save metadata: {e}")

    def save_raw_answers_to_csv(self, results: List[dict]) -> None:
        if not results:
            self.logger.warning("No results to save.")
            return
        
        output_path = os.path.join(self.results_dir, "results.csv")
        fieldnames = list(results[0].keys())

        write_header = not output_path.exists()
        with open(output_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"Saved {len(results)} results to {output_path}")

    # FUNCTIONS FOR GETTING IMAGES AND PANELS

    def get_choice_panel(self, problem_id: str) -> Optional[PIL.Image.Image]:
        # applicable only to "standard" datasets
        if hasattr(self.config, 'category') and self.config.category != 'standard':
            self.logger.warning(
                f"get_choice_panel called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return None
        
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/choice_panel.png"
        return image_path

    def get_choice_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        # image index is either string A-D, A-H or integer 0-11 (for BPs)
        # applicable to all datasets
        if not self.verify_choice_index(image_index):
            return ""

        image_path = f"data/{self.dataset_name}/problems/{problem_id}/choices/{image_index}.png"
        return image_path

    def get_question_panel(self, problem_id: str) -> str:
        # applicable to all datasets
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/question_panel.png"
        return image_path

    def get_question_image(self, problem_id: str) -> str:
        # applicable only to "standard" datasets
        if hasattr(self.config, 'category') and self.config.category != 'standard':
            self.logger.warning(
                f"get_question_image called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return ""

        image_path = f"data/{self.dataset_name}/problems/{problem_id}/question.png"
        return image_path

    def get_blackout_image(self, problem_id: str, image_index: Union[str, int]) -> str:
        # applicable only to "choice_only" datasets
        if hasattr(self.config, 'category') and self.config.category != 'choice_only':
            self.logger.warning(
                f"get_blackout_image called for non-choice_only dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return ""

        if not self.verify_choice_index(image_index):
            return ""

        image_path = f"data/{self.dataset_name}/problems/{problem_id}/blackout/{image_index}.png"
        return image_path

    def get_classification_panel(self, problem_id: str) -> str:
        # applicable to all datasets
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/classification_panel.png"
        return image_path

    def get_list_of_choice_images(self, problem_id: str, image_indices: List[Union[str, int]]) -> List[str]:
        # applicable to all datasets, used for BPs
        # verify if all indices are the same format
        if not all(isinstance(index, (str, int)) for index in image_indices):
            self.logger.error(
                f"Image indices list contained mixed types: {image_indices}. "
                f"All indices must be str or int. Returning empty list."
            )
            return []
            
        images = []
        for index in image_indices:
            images.append(self.get_choice_image(problem_id, index))
        return images
    
    def verify_choice_index(self, image_index: Union[str, int]) -> bool:
        if not hasattr(self.config, 'category'):
             self.logger.error("Config object has no 'category' attribute. Cannot verify index.")
             return False

        try:
            if self.config.category == 'standard' or self.config.category == 'choice_only':
                valid_indices = [chr(i) for i in range(ord('A'), ord('A') + self.config.num_choices)]
                # Check type consistency
                if not isinstance(image_index, str):
                    self.logger.warning(f"Index '{image_index}' is not a string, but category is '{self.config.category}'.")
                    return False
            elif self.config.category == 'BP':
                valid_indices = [i for i in range(self.config.num_choices)]
                # Check type consistency
                if not isinstance(image_index, int):
                    self.logger.warning(f"Index '{image_index}' is not an int, but category is 'BP'.")
                    return False
            else:
                self.logger.error(f"Unknown dataset category '{self.config.category}'.")
                return False
            
            if image_index not in valid_indices:
                self.logger.warning(
                    f"image_index '{image_index}' is not in the list of valid indices "
                    f"for dataset category '{self.config.category}'. Valid: {valid_indices}"
                )
                return False
                
        except AttributeError as e:
            self.logger.exception(f"Config is missing an attribute: {e}")
            return False
            
        return True

    def save_descriptions_to_json(self, descriptions_path: str, all_descriptions_data: dict):
        """
        Saves the collected descriptions dictionary to a JSON file.

        Args:
            descriptions_path (str): The full file path to save the JSON to.
            all_descriptions_data (dict): The dictionary containing problem IDs
                                          and their corresponding descriptions.
        """
        try:
            # Ensure the directory exists
            directory = os.path.dirname(descriptions_path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            # Write the data to the JSON file
            with open(descriptions_path, 'w', encoding='utf-8') as f:
                json.dump(all_descriptions_data, f, indent=4)
                
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to create directory or write to file {descriptions_path}: {e}")
        except TypeError as e:
            self.logger.error(f"Error serializing descriptions to JSON: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in save_descriptions_to_json: {e}")