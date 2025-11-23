import os
import re
import json
import string
from typing import List, Optional, Dict, Any
from PIL import Image
from code.preprocessing.base_processor import BaseProcessor
from code.preprocessing.processor_config import ProcessorConfig
from PIL import Image, ImageDraw, ImageFont
import random


class StandardProcessor(BaseProcessor):
    """Processor for standard visual reasoning datasets with choice images."""

    def __init__(
        self, config: ProcessorConfig, sheet_maker, output_base_path: str = "data"
    ):
        super().__init__(config, output_base_path)
        self.sheet_maker = sheet_maker
        # Load existing metadata first to make processing additive
        self.answers_dict = self.load_existing_json(
            f"{self.dataset_name}_solutions.json"
        )
        self.shuffle_orders = self.load_existing_json(
            f"{self.dataset_name}_shuffle_orders.json"
        )
        self.annotations_dict = self.load_existing_json(
            f"{self.dataset_name}_annotations.json"
        )

        # Log counts of loaded metadata
        self.logger.info(f"Loaded {len(self.answers_dict)} existing solutions")
        self.logger.info(f"Loaded {len(self.annotations_dict)} existing annotations")

        random.seed(42)  # For reproducible shuffling

    def load_existing_json(self, filename: str) -> Dict[str, Any]:
        """Loads an existing JSON metadata file if it exists."""
        json_path = self.output_base_path / self.dataset_name / "jsons" / filename
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(
                    f"Could not load existing metadata {filename}: {e}. Starting fresh."
                )
                return {}
        return {}

    def process(self) -> None:
        """Process all problems in the dataset."""
        processed_count = 0
        skipped_count = 0
        error_count = 0

        problems = [
            p
            for p in os.listdir(self.raw_data_path)
            if (self.raw_data_path / p).is_dir()
        ]

        self.logger.info(f"Found {len(problems)} problems to process")

        # Keep track of problems processed in *this* run
        newly_processed_problem_ids = set()

        for problem_id in problems:
            problem_path = self.raw_data_path / problem_id

            if not problem_path.is_dir():
                continue

            # Standardize problem ID first
            problem_id_standardized = self.standardize_problem_id(problem_id)

            # Check if already processed
            if self.is_already_processed(problem_id_standardized):
                self.logger.debug(
                    f"Problem {problem_id_standardized} already processed, skipping image generation"
                )

                # Check if metadata exists. If not, re-generate it.
                if problem_id_standardized not in self.answers_dict or (
                    self.config.annotations_folder
                    and problem_id_standardized not in self.annotations_dict
                ):
                    self.logger.debug(
                        f"Missing metadata for {problem_id_standardized}. Re-populating..."
                    )
                else:
                    skipped_count += 1
                    continue  # Skip fully if images AND metadata exist

            try:
                # Load images
                choice_images = self.load_choice_images(problem_id)
                if None in choice_images:
                    self.logger.warning(
                        f"Skipping problem {problem_id} due to missing choice images"
                    )
                    error_count += 1
                    continue

                question_image = self.load_question_image(problem_id)

                # Determine answer and shuffle settings
                answer_info = self.get_answer_info(problem_id)

                # Determine shuffle order (load if it exists, else generate)
                shuffle_order = self.shuffle_orders.get(problem_id_standardized)
                true_answer_index = answer_info["true_idx"]

                # --- NEW SHUFFLE LOGIC: Run this *before* any image generation ---
                if self.config.shuffle:
                    if shuffle_order:
                        # Use existing shuffle order
                        self.logger.debug(
                            f"Using existing shuffle order for {problem_id_standardized}"
                        )
                        shuffled_choices = [choice_images[i] for i in shuffle_order]
                        if true_answer_index is not None:
                            answer_label = string.ascii_uppercase[
                                shuffle_order.index(true_answer_index)
                            ]
                        else:
                            answer_label = None
                    else:
                        # Generate new shuffle order
                        self.logger.debug(
                            f"Generating new shuffle order for {problem_id_standardized}"
                        )
                        indices = list(range(len(choice_images)))
                        random.shuffle(indices)
                        shuffle_order = indices
                        shuffled_choices = [choice_images[i] for i in shuffle_order]
                        if true_answer_index is not None:
                            answer_label = string.ascii_uppercase[
                                shuffle_order.index(true_answer_index)
                            ]
                        else:
                            answer_label = None
                else:
                    # No shuffle
                    shuffle_order = None  # Explicitly set to None
                    shuffled_choices = choice_images
                    if true_answer_index is not None:
                        answer_label = string.ascii_uppercase[true_answer_index]
                    else:
                        answer_label = None
                # --- END OF SHUFFLE LOGIC ---

                # Generate sheet (only if not already processed)
                if not self.is_already_processed(problem_id_standardized):
                    # --- FIX 1: Save refactored images *using shuffled_choices* ---
                    self.save_refactored_images(
                        problem_id_standardized,
                        shuffled_choices,
                        letters=True,
                        question_image=question_image,
                    )

                    sheet, _, _ = self.sheet_maker.generate_question_sheet_from_images(
                        shuffled_choices,  # Use the (potentially) shuffled choices
                        question_image=question_image,
                        shuffle_answers=False,  # Shuffling is now done manually above
                        true_answer_index=None,  # Not needed, we derived label already
                    )
                    # Save sheet
                    self.save_sheet(problem_id_standardized, sheet)

                    # --- FIX 2: Generate blackout versions *using shuffled_choices* ---
                    if self.config.category == "choice_only":
                        tmp_images = self.generate_blackout_sheets(
                            problem_id_standardized, shuffled_choices, question_image
                        )

                    # --- FIX 3: Generate filled versions *using shuffled_choices* ---
                    if self.config.category == "standard":
                        tmp_images = self.sheet_maker.generate_question_filled(
                            question_image,
                            shuffled_choices,
                            self.config.data_folder,
                            problem_id_standardized,
                            self.output_base_path,
                            crop_px=2,
                        )
                        choice_panel, _, _ = (
                            self.sheet_maker.generate_question_sheet_from_images(
                                shuffled_choices,
                                shuffle_answers=False,
                                true_answer_index=None,
                            )
                        )
                        self.save_sheet(
                            problem_id_standardized, choice_panel, choice_panel=True
                        )

                    # Save blackout/filled images for classification setting
                    classification_panel = (
                        self.sheet_maker.generate_question_sheet_from_images(
                            tmp_images, shuffle_answers=False, true_answer_index=None
                        )[0]
                    )
                    self.save_sheet(
                        problem_id_standardized,
                        classification_panel,
                        classification_panel=True,
                    )

                # Store metadata (runs even if images were skipped, to populate metadata)
                if answer_label is not None:
                    self.answers_dict[problem_id_standardized] = answer_label
                else:
                    if answer_info.get("true_idx") is not None:
                        idx = answer_info["true_idx"]
                        label = (
                            string.ascii_uppercase[idx]
                            if 0 <= idx < len(string.ascii_uppercase)
                            else str(idx)
                        )
                        self.answers_dict[problem_id_standardized] = label

                if shuffle_order:
                    self.shuffle_orders[problem_id_standardized] = shuffle_order

                # Load and store annotations if available
                # This function correctly uses the shuffle_order to remap annotations
                annotations = self.load_annotations(problem_id, shuffle_order)
                if annotations:
                    self.annotations_dict[problem_id_standardized] = annotations

                processed_count += 1
                newly_processed_problem_ids.add(problem_id_standardized)
                self.logger.debug(
                    f"Successfully processed/updated metadata for problem {problem_id_standardized}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error processing problem {problem_id}: {e}", exc_info=True
                )
                error_count += 1

        # Log summary
        self.logger.info(
            f"Processing complete: {processed_count} processed/updated, "
            f"{skipped_count} skipped (images and metadata OK), "
            f"{error_count} errors"
        )

        # Save all metadata
        if processed_count > 0:
            self.logger.info(
                f"Saving metadata for {len(self.answers_dict)} total problems..."
            )
            self.save_metadata()
        else:
            self.logger.info("No new problems processed, skipping metadata save")

    def load_choice_images(self, problem_id: str) -> List[Optional[Image.Image]]:
        """Load choice images for a problem."""
        images = []
        choice_dir = (
            self.raw_data_path
            / problem_id
            / self.config.choice_images_folder.lstrip("/")
        )

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

        question_dir = (
            self.raw_data_path
            / problem_id
            / self.config.question_images_folder.lstrip("/")
        )
        return self.load_image_by_pattern(question_dir, self.config.image_format)

    def get_answer_info(self, problem_id: str) -> Dict[str, Any]:
        """Get answer information for a problem."""
        if self.config.shuffle is False and self.config.true_idx is None:
            return self.load_answer_from_image(problem_id)
        return {"true_idx": self.config.true_idx}

    def load_answer_from_image(self, problem_id: str) -> Dict[str, Any]:
        """Load answer from answer image file."""
        answer_dir = (
            self.raw_data_path
            / problem_id
            / self.config.answer_images_folder.lstrip("/")
        )

        try:
            for fname in os.listdir(answer_dir):
                if fname.lower().endswith(self.config.image_format):
                    num_str = os.path.splitext(fname)[0]
                    if num_str.isdigit():
                        return {"true_idx": int(num_str)}
            return {"true_idx": None}
        except Exception as e:
            self.logger.error(f"Error loading answer image: {e}")
            return {"true_idx": None}

    def load_annotations(
        self, problem_id: str, shuffle_order: Optional[List[int]] = None
    ) -> Optional[Dict[str, str]]:
        """Load and process annotations."""
        if not self.config.annotations_folder:
            return None

        annot_path = (
            self.raw_data_path / problem_id / self.config.annotations_folder.lstrip("/")
        )

        if not annot_path.exists():
            return None

        try:
            with open(annot_path, "r", encoding="utf-8") as f:
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
            return {string.ascii_uppercase[i]: desc for i, desc in enumerate(processed)}

        except Exception as e:
            self.logger.error(f"Error reading annotations: {e}")
            return None

    def _process_dict_annotations(self, annotations: Dict) -> List[str]:
        """Process dictionary-format annotations."""
        # Try to extract numeric indices from filenames
        items = []
        for fname, desc in annotations.items():
            # Try to find number pattern
            match = re.search(r"[_T]?(\d+)", fname)
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
        dataset__name = self.dataset_name

        if self.answers_dict:
            self.save_json(self.answers_dict, f"{self.dataset_name}_solutions.json")

        if self.shuffle_orders:
            self.save_json(
                self.shuffle_orders, f"{self.dataset_name}_shuffle_orders.json"
            )

        if self.annotations_dict:
            self.save_json(
                self.annotations_dict, f"{self.dataset_name}_annotations.json"
            )

    def generate_blackout_sheets(
        self,
        problem_id_standardized: str,
        choice_images: list,
        question_image: Image.Image | None,
    ) -> list[Image.Image]:
        """
        Generate and save blackout sheets (A–D) for each answer position.
        Assumes choice_images is the *final* list (i.e., already shuffled).
        """
        blackout_image_list = []
        blackout_dir = (
            self.output_base_path
            / "cvr"
            / "problems"
            / problem_id_standardized
            / "blackout"
        )
        blackout_dir.mkdir(parents=True, exist_ok=True)

        num_choices = len(choice_images)
        for i in range(num_choices):
            try:
                sheet, _, _ = self.sheet_maker.generate_question_sheet_from_images(
                    choice_images,
                    question_image=question_image,
                    shuffle_answers=False,  # no shuffle, list is already in final order
                    blackout=i,  # black out the i-th choice
                    no_label=True,  # do not label choices
                )
                label = string.ascii_uppercase[i]
                out_path = blackout_dir / f"{label}.png"
                sheet.save(out_path)
                blackout_image_list.append(sheet)
                self.logger.debug(
                    f"Saved blackout sheet for {problem_id_standardized} ({label}) at {out_path}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to generate blackout sheet {label} for {problem_id_standardized}: {e}"
                )

        return blackout_image_list
