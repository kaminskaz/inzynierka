import os
import re
import json
import string
from typing import List, Optional, Dict, Any
from PIL import Image
from code.preprocessing_new.baseprocessor import BaseProcessor
from code.preprocessing_new.processorconfig import ProcessorConfig
from PIL import Image, ImageDraw, ImageFont
import random

class StandardProcessor(BaseProcessor):
    """Processor for standard visual reasoning datasets with choice images."""
    
    def __init__(self, config: ProcessorConfig, sheet_maker, output_base_path: str = "data"):
        super().__init__(config, output_base_path)
        self.sheet_maker = sheet_maker
        self.answers_dict = {}
        self.shuffle_orders = {}
        self.annotations_dict = {}
        random.seed(42)  # For reproducible shuffling
    
    def process(self) -> None:
        """Process all problems in the dataset."""
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        problems = [p for p in os.listdir(self.raw_data_path) 
                   if (self.raw_data_path / p).is_dir()]
        
        self.logger.info(f"Found {len(problems)} problems to process")
        
        for problem_id in problems:
            problem_path = self.raw_data_path / problem_id
            
            if not problem_path.is_dir():
                continue
            
            # Standardize problem ID first
            problem_id_standardized = self.standardize_problem_id(problem_id)
            
            # Check if already processed
            if self.is_already_processed(problem_id_standardized):
                self.logger.debug(f"Problem {problem_id_standardized} already processed, skipping")
                skipped_count += 1
                continue
            
            try:
                # Load images
                choice_images = self.load_choice_images(problem_id)
                if None in choice_images:
                    self.logger.warning(f"Skipping problem {problem_id} due to missing choice images")
                    error_count += 1
                    continue
                
                question_image = self.load_question_image(problem_id)

                # Save to refactored/
                self.save_refactored_images(problem_id_standardized, choice_images, letters=True, question_image=question_image)
                
                # Determine answer and shuffle settings
                answer_info = self.get_answer_info(problem_id)
                
                # Generate sheet
                sheet, answer_label, shuffle_order = self.sheet_maker.generate_question_sheet_from_images(
                    choice_images,
                    question_image=question_image,
                    shuffle_answers=self.config.shuffle,
                    true_answer_index=answer_info['true_idx']
                )
                
                # Generate blackout versions for CVR dataset
                if "cvr" in str(self.config.data_folder).lower():
                    self.generate_blackout_sheets(problem_id_standardized, choice_images, question_image)

                if "raven" in str(self.config.data_folder).lower() or "marsvqa" in str(self.config.data_folder).lower():
                    self.sheet_maker.generate_question_filled(question_image, choice_images, self.config.data_folder, problem_id_standardized, self.output_base_path, crop_px=2)

                # Save sheet
                self.save_sheet(problem_id_standardized, sheet)
                
                # Store metadata
                if answer_label is not None:
                    self.answers_dict[problem_id_standardized] = answer_label
                else:
                    if answer_info.get('true_idx') is not None:
                        idx = answer_info['true_idx']
                        label = string.ascii_uppercase[idx] if 0 <= idx < len(string.ascii_uppercase) else str(idx)
                        self.answers_dict[problem_id_standardized] = label
                if shuffle_order:
                    self.shuffle_orders[problem_id_standardized] = shuffle_order
                
                # Load and store annotations if available
                annotations = self.load_annotations(problem_id, shuffle_order)
                if annotations:
                    self.annotations_dict[problem_id_standardized] = annotations
                
                processed_count += 1
                self.logger.debug(f"Successfully processed problem {problem_id_standardized}")
                
            except Exception as e:
                self.logger.error(f"Error processing problem {problem_id}: {e}", exc_info=True)
                error_count += 1
        
        # Log summary
        self.logger.info(
            f"Processing complete: {processed_count} processed, "
            f"{skipped_count} skipped (already processed), "
            f"{error_count} errors"
        )
        
        # Save all metadata
        if processed_count > 0:
            self.save_metadata()
        else:
            self.logger.info("No new problems processed, skipping metadata save")
    
    def load_choice_images(self, problem_id: str) -> List[Optional[Image.Image]]:
        """Load choice images for a problem."""
        images = []
        choice_dir = self.raw_data_path / problem_id / self.config.choice_images_folder.lstrip('/')
        
        for i in range(self.config.num_choices):
            pattern = self.evaluate_regex(self.config.regex_choice_number, i)
            
            # Try loading by exact filename first
            exact_path = choice_dir / pattern
            if exact_path.exists():
                try:
                    images.append(Image.open(exact_path).convert("RGB"))
                    continue
                except Exception as e:
                    self.logger.error(f"Error loading {exact_path}: {e}")
            
            # Try pattern matching
            image = self.load_image_by_pattern(choice_dir, pattern)
            images.append(image)
        
        return images
    
    def load_question_image(self, problem_id: str) -> Optional[Image.Image]:
        """Load question image if available."""
        if not self.config.question_images_folder:
            return None
        
        question_dir = self.raw_data_path / problem_id / self.config.question_images_folder.lstrip('/')
        return self.load_image_by_pattern(question_dir, self.config.image_format)
    
    def get_answer_info(self, problem_id: str) -> Dict[str, Any]:
        """Get answer information for a problem."""
        if self.config.shuffle is False and self.config.true_idx is None:
            return self.load_answer_from_image(problem_id)
        return {'true_idx': self.config.true_idx}
    
    def load_answer_from_image(self, problem_id: str) -> Dict[str, Any]:
        """Load answer from answer image file."""
        answer_dir = self.raw_data_path / problem_id / self.config.answer_images_folder.lstrip('/')
        
        try:
            for fname in os.listdir(answer_dir):
                if fname.lower().endswith(self.config.image_format):
                    num_str = os.path.splitext(fname)[0]
                    if num_str.isdigit():
                        return {'true_idx': int(num_str)}
            return {'true_idx': None}
        except Exception as e:
            self.logger.error(f"Error loading answer image: {e}")
            return {'true_idx': None}
    
    def load_annotations(self, problem_id: str, shuffle_order: Optional[List[int]] = None) -> Optional[Dict[str, str]]:
        """Load and process annotations."""
        if not self.config.annotations_folder:
            return None
        
        annot_path = self.raw_data_path / problem_id / self.config.annotations_folder.lstrip('/')
        
        if not annot_path.exists():
            return None
        
        try:
            with open(annot_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # Process based on format
            if isinstance(annotations, dict):
                processed = self._process_dict_annotations(annotations)
            elif isinstance(annotations, list):
                processed = annotations
            else:
                return None
            
            # Apply shuffle if provided
            if shuffle_order:
                processed = [processed[i] for i in shuffle_order]
            
            # Add letter prefixes
            return {
                string.ascii_uppercase[i]: desc 
                for i, desc in enumerate(processed)
            }
        
        except Exception as e:
            self.logger.error(f"Error reading annotations: {e}")
            return None
    
    def _process_dict_annotations(self, annotations: Dict) -> List[str]:
        """Process dictionary-format annotations."""
        # Try to extract numeric indices from filenames
        items = []
        for fname, desc in annotations.items():
            # Try to find number pattern
            match = re.search(r'[_T]?(\d+)', fname)
            if match:
                index = int(match.group(1))
                # Adjust for 1-based indexing
                if index > 0:
                    index -= 1
                items.append((index, desc))
            else:
                items.append((len(items), desc))
        
        items.sort(key=lambda x: x[0])
        return [desc for _, desc in items]
    
    def save_metadata(self) -> None:
        """Save all metadata to JSON files."""
        dataset_name = self.dataset_name
        
        if self.answers_dict:
            self.save_json(self.answers_dict, f"{dataset_name}_solutions.json")
        
        if self.shuffle_orders:
            self.save_json(self.shuffle_orders, f"{dataset_name}_shuffle_orders.json")
        
        if self.annotations_dict:
            self.save_json(self.annotations_dict, f"{dataset_name}_annotations.json")

    def generate_blackout_sheets(self, problem_id_standardized: str, choice_images: list, question_image: Image.Image | None) -> None:
        """Generate and save blackout sheets (A–D) for each answer position."""
        blackout_dir = self.output_base_path / "cvr" / "problems" / problem_id_standardized / "blackout"
        blackout_dir.mkdir(parents=True, exist_ok=True)

        num_choices = len(choice_images)
        for i in range(num_choices):
            try:
                sheet, _, _ = self.sheet_maker.generate_question_sheet_from_images(
                    choice_images,
                    question_image=question_image,
                    shuffle_answers=False,  # no shuffle for blackout
                    blackout=i,              # black out the i-th choice
                )
                label = string.ascii_uppercase[i]
                out_path = blackout_dir / f"{label}.png"
                sheet.save(out_path)
                self.logger.debug(f"Saved blackout sheet for {problem_id_standardized} ({label}) at {out_path}")
            except Exception as e:
                self.logger.error(f"Failed to generate blackout sheet {label} for {problem_id_standardized}: {e}")

