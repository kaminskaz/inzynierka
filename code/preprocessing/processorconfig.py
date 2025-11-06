from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ProcessorConfig:
    """Configuration for a dataset processor."""
    image_format: str
    category: str
    data_folder: str
    num_choices: int
    choice_images_folder: str
    regex_choice_number: str
    shuffle: bool
    hf_repo_id: Optional[str] = None
    hf_repo_type: Optional[str] = "dataset"
    question_images_folder: Optional[str] = None
    answer_images_folder: Optional[str] = None
    annotations_folder: Optional[str] = None
    regex_answer_number: Optional[str] = None
    solutions_folder: Optional[str] = None
    true_idx: Optional[int] = None
    expected_num_samples: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessorConfig':
        """Create config from dictionary."""
        return cls(**config_dict)