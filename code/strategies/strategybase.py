import logging
from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional
import PIL.Image

from code.preprocessing.processorconfig import ProcessorConfig 

class StrategyBase(ABC):
    def __init__(self, dataset_name: str, model: Any, dataset_config: ProcessorConfig):
        self.dataset_name: str = dataset_name
        self.model: Any = model
        self.config: ProcessorConfig = dataset_config
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized strategy for dataset: '{self.dataset_name}'")

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def save_metadata(self) -> None:
        pass

    @abstractmethod
    def save_raw_answers_to_csv(self) -> None:
        pass

    @abstractmethod
    def ask_model(self) -> Any:
        # functions from model module and prompt formatter here
        pass

    # FUNCTIONS FOR GETTING IMAGES AND PANELS

    def get_image_from_path(self, image_path: str) -> Optional[PIL.Image.Image]:
        try:
            image = PIL.Image.open(image_path)
            return image
        except Exception as e:
            self.logger.exception(f"Error loading image from {image_path}")
            return None

    def get_choice_panel(self, problem_id: str) -> Optional[PIL.Image.Image]:
        # applicable only to "standard" datasets
        if hasattr(self.config, 'category') and self.config.category != 'standard':
            self.logger.warning(
                f"get_choice_panel called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return None
        
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/choice_panel.png"
        return self.get_image_from_path(image_path)

    def get_choice_image(self, problem_id: str, image_index: Union[str, int]) -> Optional[PIL.Image.Image]:
        # image index is either string A-D, A-H or integer 0-11 (for BPs)
        # applicable to all datasets
        if not self.verify_choice_index(image_index):
            return None
        
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/choices/{image_index}.png"
        return self.get_image_from_path(image_path)
    
    def get_question_panel(self, problem_id: str) -> Optional[PIL.Image.Image]:
        # applicable to all datasets
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/question_panel.png"
        return self.get_image_from_path(image_path)
    
    def get_question_image(self, problem_id: str) -> Optional[PIL.Image.Image]:
        # applicable only to "standard" datasets
        if hasattr(self.config, 'category') and self.config.category != 'standard':
            self.logger.warning(
                f"get_question_image called for non-standard dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return None

        image_path = f"data/{self.dataset_name}/problems/{problem_id}/question.png"
        return self.get_image_from_path(image_path)

    def get_blackout_image(self, problem_id: str, image_index: Union[str, int]) -> Optional[PIL.Image.Image]:
        # applicable only to "choice_only" datasets
        if hasattr(self.config, 'category') and self.config.category != 'choice_only':
            self.logger.warning(
                f"get_blackout_image called for non-choice_only dataset "
                f"(category: '{self.config.category}'). Returning None."
            )
            return None
        
        if not self.verify_choice_index(image_index):
            return None

        image_path = f"data/{self.dataset_name}/problems/{problem_id}/blackout/{image_index}.png"
        return self.get_image_from_path(image_path)

    def get_classification_panel(self, problem_id: str) -> Optional[PIL.Image.Image]:
        # applicable to all datasets
        image_path = f"data/{self.dataset_name}/problems/{problem_id}/classification_panel.png"
        return self.get_image_from_path(image_path)

    def get_list_of_choice_images(self, problem_id: str, image_indices: List[Union[str, int]]) -> List[Optional[PIL.Image.Image]]:
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