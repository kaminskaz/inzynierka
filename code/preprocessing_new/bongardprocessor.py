import os
import json
from typing import List, Optional, Dict, Any
from PIL import Image, ImageDraw
from code.preprocessing_new.baseprocessor import BaseProcessor
from code.preprocessing_new.processorconfig import ProcessorConfig


class BongardProcessor(BaseProcessor):
    """Specialized processor for Bongard Problems."""
    
    def process(self) -> None:
        """Process all Bongard problems."""
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
            
            # Standardize ID first
            problem_id_standardized = self.standardize_problem_id(problem_id)
            
            # Check if already processed
            if self.is_already_processed(problem_id_standardized):
                self.logger.debug(f"Problem {problem_id_standardized} already processed, skipping")
                skipped_count += 1
                continue
            
            try:
                images = self.load_choice_images(problem_id)
                
                if None in images:
                    self.logger.warning(f"Skipping problem {problem_id} due to missing images")
                    error_count += 1
                    continue

                # Save to refactored/
                self.save_refactored_images(problem_id_standardized, images, letters=False)

                # Generate sheet using custom layout
                sheets = self.generate_bongard_sheet(images)

                # Save both sheets
                self.save_sheet(problem_id_standardized, sheets["normal"])
                self.save_sheet(problem_id_standardized, sheets["switched"], switched=True)
                
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
        
        # Process solutions
        if processed_count > 0 or not self.get_output_dir("jsons").joinpath("bp_solutions.json").exists():
            self.process_solutions()
        else:
            self.logger.info("No new problems processed and solutions already exist, skipping solutions processing")
    
    def load_choice_images(self, problem_id: str) -> List[Optional[Image.Image]]:
        """Load all 12 images for a Bongard problem."""
        images = []
        problem_path = self.raw_data_path / problem_id
        
        for i in range(self.config.num_choices):
            image_path = problem_path / f"{i}.png"
            try:
                images.append(Image.open(image_path).convert("RGB"))
            except Exception as e:
                self.logger.error(f"Error loading image {image_path}: {e}")
                images.append(None)
        
        return images
    
    def generate_bongard_sheet(self, images: List[Image.Image], spacing: int = 10, margin: int = 20, border_thickness: int = 2, 
        space_between_panels: int = 40) -> Dict[str, Image.Image]:
        """Generate two Bongard problem sheets: normal and with 5↔11 switched."""
        if len(images) != 12:
            raise ValueError("Exactly 12 images are required for Bongard problems.")
        
        def _make_sheet(imgs: List[Image.Image]) -> Image.Image:
            """Internal helper to build one sheet."""
            # Resize images to uniform width
            max_width = max(img.width for img in imgs)
            resized_images = []
            for img in imgs:
                if img.width != max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height))
                resized_images.append(img)
            
            # --- START LAYOUT FIX ---
            
            col_width = max_width
            
            # Calculate max height for each of the 3 rows
            max_row_heights = [
                max(resized_images[0].height, resized_images[3].height, resized_images[6].height, resized_images[9].height),  # Row 0
                max(resized_images[1].height, resized_images[4].height, resized_images[7].height, resized_images[10].height), # Row 1
                max(resized_images[2].height, resized_images[5].height, resized_images[8].height, resized_images[11].height)  # Row 2
            ]
            
            # Calculate total width
            total_width = col_width * 4 + spacing * 3 + space_between_panels + 2 * margin
            
            # Calculate new total height
            total_height = sum(max_row_heights) + spacing * 2 + 2 * margin
            
            # Create sheet
            sheet = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(sheet)
            
            # Define x-coordinates for the 4 columns
            x_coords = [
                margin,
                margin + col_width + spacing,
                margin + col_width * 2 + spacing * 2 + space_between_panels,
                margin + col_width * 3 + spacing * 3 + space_between_panels
            ]
            
            # Define y-coordinates for the 3 rows
            y_coords = [
                margin,
                margin + max_row_heights[0] + spacing,
                margin + max_row_heights[0] + max_row_heights[1] + spacing * 2
            ]
            
            # Map of image indices for the [row][col] grid layout
            image_grid_indices = [
                [0, 3, 6, 9],   # Row 0
                [1, 4, 7, 10],  # Row 1
                [2, 5, 8, 11]   # Row 2
            ]

            for r_idx in range(3):
                for c_idx in range(4):
                    img_idx = image_grid_indices[r_idx][c_idx]
                    img = resized_images[img_idx]
                    x_pos = x_coords[c_idx]
                    y_pos = y_coords[r_idx]
                    
                    # Paste image (aligns to top-left of its grid cell)
                    sheet.paste(img, (x_pos, y_pos))
                    
                    # Draw border around the pasted image
                    draw.rectangle(
                        [x_pos, y_pos, x_pos + img.width - 1, y_pos + img.height - 1],
                        outline="black", 
                        width=border_thickness
                    )
            
            # --- END LAYOUT FIX ---
            
            return sheet
        
        # Normal sheet
        normal_sheet = _make_sheet(images)
        
        # Switched version (swap index 5 and 11)
        switched_images = images.copy()
        switched_images[5], switched_images[11] = switched_images[11], switched_images[5]
        switched_sheet = _make_sheet(switched_images)
        
        return {
            "normal": normal_sheet,
            "switched": switched_sheet
        }

    
    def process_solutions(self) -> None:
        """Process and renumber solutions from raw data."""
        raw_solutions_path = self.raw_data_path / "bp_solutions.json"
        
        if not raw_solutions_path.exists():
            self.logger.warning(f"Solutions file not found: {raw_solutions_path}")
            return
        
        try:
            with open(raw_solutions_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            # Renumber with zero-padding
            renumbered_data = {
                f"{int(k):03d}": answers_data[k] 
                for k in sorted(answers_data.keys(), key=lambda x: int(x))
            }
            
            self.save_json(renumbered_data, "bp_solutions.json")
            
        except Exception as e:
            self.logger.error(f"Error processing solutions: {e}", exc_info=True)