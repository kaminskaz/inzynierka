import os
import json
import random
import re
import string
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image
from src.technical.configs.dataset_config import DatasetConfig


class BaseProcessor(ABC):
    """Abstract base class for all dataset processors."""

    def __init__(self, config: DatasetConfig, output_base_path: str = "data"):
        self.config = config
        self.raw_data_path = Path(config.data_folder)
        self.output_base_path = Path(output_base_path)
        self.dataset_name = self.raw_data_path.name
        self.logger = logging.getLogger(f"DataProcessor.{self.dataset_name}")
        random.seed(42)  # For reproducible shuffling

    @abstractmethod
    def process(self) -> None:
        """Process all problems in the dataset."""
        pass

    @abstractmethod
    def load_choice_images(self, problem_id: str) -> List[Optional[Image.Image]]:
        """Load all choice images for a problem."""
        pass

    def is_already_processed(self, problem_id: str) -> bool:
        """Check if a problem has already been processed."""
        output_path = os.path.join(self.output_base_path, self.dataset_name, "problems", problem_id, "question_panel.png")
        return os.path.exists(output_path)

    def get_output_dir(self, subfolder: str) -> Path:
        """Get output directory path."""
        path = os.path.join(self.output_base_path, self.dataset_name, subfolder)
        os.makedirs(path, exist_ok=True)
        return path

    def save_sheet(
        self,
        problem_id: str,
        sheet: Image.Image,
        switched: bool = False,
        choice_panel: bool = False,
        classification_panel: bool = False,
    ) -> None:
        """Save the generated question panel image."""
        # Directory structure: data/<dataset_name>/<problem_id>/
        output_dir = os.path.join(self.output_base_path, self.dataset_name, "problems", problem_id)
        os.makedirs(output_dir, exist_ok=True)
        if classification_panel:
            save_path = os.path.join(output_dir, "classification_panel.png")
        elif switched and not choice_panel:
            save_path = os.path.join(output_dir, "question_panel_switched.png")
        elif choice_panel:
            save_path = os.path.join(output_dir, "choice_panel.png")
        else:
            save_path = os.path.join(output_dir, "question_panel.png")
        sheet.save(save_path)
        self.logger.debug(
            f"Saved question panel for problem {problem_id} to {save_path}"
        )

    def save_json(self, data: Dict, filename: str) -> None:
        """Save dataset-level JSON metadata inside jsons/ folder."""
        json_dir = os.path.join(self.output_base_path, self.dataset_name, "jsons")
        os.makedirs(json_dir, exist_ok=True)

        file_path = os.path.join(json_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.logger.info(f"Saved JSON metadata to {file_path}")

    def load_image_by_pattern(
        self, directory: Path, pattern: str
    ) -> Optional[Image.Image]:
        """Load image matching a pattern in the directory."""
        try:
            if not directory.exists():
                return None

            for fname in os.listdir(directory):
                if pattern in fname or fname.endswith(pattern):
                    return Image.open(os.path.join(directory, fname)).convert("RGB")

            return None
        except Exception as e:
            self.logger.error(
                f"Error loading image from {directory} with pattern {pattern}: {e}"
            )
            return None

    def standardize_problem_id(self, problem_id: str) -> str:
        """Standardize problem ID to fixed number of digits."""
        expected = self.config.expected_num_samples
        if expected is None:
                raise ValueError("expected_num_samples is not set in config")
        if not isinstance(expected, int):
                raise ValueError("expected_num_samples must be an integer")
        digits = len(str(self.config.expected_num_samples))
        return str(problem_id).zfill(digits)

    def evaluate_regex(self, regex_template: str, image_index: int) -> str:
        """Evaluate regex template with image_index."""
        # Replace {image_index} and {image_index+1} patterns
        result = regex_template.replace("{image_index}", str(image_index))
        result = re.sub(
            r"\{image_index\+(\d+)\}",
            lambda m: str(image_index + int(m.group(1))),
            result,
        )
        return result

    def save_refactored_images(
        self,
        problem_id_standardized: str,
        choice_images: List[Image.Image],
        letters: bool,
        question_image: Optional[Image.Image] = None,
    ) -> None:
        """Save labeled choice and question images into data/<dataset>/problems/<problem_id>/"""

        base_dir = os.path.join(self.output_base_path, self.dataset_name, "problems", problem_id_standardized)
        os.makedirs(base_dir, exist_ok=True)

        choices_dir = os.path.join(base_dir, "choices")
        os.makedirs(choices_dir, exist_ok=True)

        for i, img in enumerate(choice_images):
            if img is None:
                continue
            label = string.ascii_uppercase[i] if letters else str(i)
            out_path = os.path.join(choices_dir, f"{label}.{self.config.image_format.lstrip('.')}")
            img.save(out_path)
            self.logger.debug(f"Saved choice {label} to {out_path}")

        if question_image is not None:
            q_path = os.path.join(base_dir, f"question.{self.config.image_format.lstrip('.')}")
            question_image.save(q_path)
            self.logger.debug(f"Saved question image to {q_path}")
