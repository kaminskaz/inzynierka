import os
import string
import json
from PIL import Image
from code.preprocessing.vcog.VCOGsheetmaker import VCOGSheetMaker


class RAVENProcessor:
    def __init__(self):
        self.raw_data_folder_path = os.path.join("data_raw", "vcog-bench", "raven")
        self.output_data_folder_path = "data"
        self.sheetmaker = VCOGSheetMaker()
        self.answers_dict = {}
        self.annotations_dict = {}

    def process(self):
        for problem_id in os.listdir(self.raw_data_folder_path):
            problem_path = os.path.join(self.raw_data_folder_path, problem_id)
            if not os.path.isdir(problem_path):
                continue

            # Load 8 choice images
            images = [self.load_image(problem_id, i) for i in range(8)]
            if None in images:
                print(f"Skipping problem {problem_id} due to missing images.")
                continue

            # Load the question image
            question_image = self.load_question_image(problem_id)
            if question_image is None:
                print(f"Skipping problem {problem_id} due to missing question image.")
                continue

            # Load the answer image
            answer_image, answer = self.load_answer_image(problem_id)
            if answer_image is None:
                print(f"Skipping problem {problem_id} due to missing answer image.")
                continue

            # Generate sheet — no shuffling, true answer index known (last or specific one)
            sheet, _, _ = self.sheetmaker.generate_question_sheet_from_images(
                images, question_image=question_image, shuffle_answers=False, true_answer_index=None
            )

            # Standardize problem_id to 3 digits
            problem_id = problem_id.zfill(3)

            # Save the sheet
            self.save_sheet(problem_id, sheet)

            # Store the answer
            self.answers_dict[problem_id] = answer  # or modify if the filename differs

            # Store processed annotations
            self.annotations_dict[problem_id] = self.get_annotations(problem_id)

        # Save answers
        solutions_dir = os.path.join(
            self.output_data_folder_path, "direct", "vcog-bench", "raven", "solutions"
        )
        os.makedirs(solutions_dir, exist_ok=True)
        with open(os.path.join(solutions_dir, "raven_solutions.json"), "w") as f:
            json.dump(self.answers_dict, f, indent=4)

        with open(os.path.join(solutions_dir, "raven_annotations.json"), "w") as f:
            json.dump(self.annotations_dict, f, indent=4)

    def load_image(self, problem_id: str, image_index: int):
        image_path = os.path.join(
            self.raw_data_folder_path, problem_id, "choice", "image", f"{image_index}.jpeg"
        )
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_question_image(self, problem_id: str):
        image_dir = os.path.join(self.raw_data_folder_path, problem_id, "question", "image")
        try:
            for fname in os.listdir(image_dir):
                if fname.lower().endswith(".jpeg"):
                    return Image.open(os.path.join(image_dir, fname)).convert("RGB")
            print(f"No question image found in {image_dir}")
            return None
        except Exception as e:
            print(f"Error loading question image: {e}")
            return None

    def load_answer_image(self, problem_id: str):
        image_dir = os.path.join(self.raw_data_folder_path, problem_id, "answer", "image")
        try:
            for fname in os.listdir(image_dir):
                if fname.lower().endswith(".jpeg"):
                    # Extract numeric part, e.g. "3.jpeg" -> 3
                    num_str = os.path.splitext(fname)[0]
                    if num_str.isdigit():

                        num = int(num_str)
                        # Map 0–7 → A–H
                        letter = string.ascii_uppercase[num]
                    else:
                        letter = None

                    image_path = os.path.join(image_dir, fname)
                    return Image.open(image_path).convert("RGB"), letter

            print(f"No answer image found in {image_dir}")
            return None, None
        
        except Exception as e:
            print(f"Error loading answer image: {e}")
            return None, None


    def save_sheet(self, problem_id: str, sheet):
        save_dir = os.path.join(self.output_data_folder_path, "direct", "vcog-bench", "raven", "images")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{problem_id}.png")
        sheet.save(save_path)

    def get_annotations(self, problem_id: str):
        problem_id = str(int(problem_id)) 
        annot_path = os.path.join(
            self.raw_data_folder_path, problem_id, "choice", "text", "annotation.json"
        )

        if not os.path.exists(annot_path):
            print(f"Annotation file not found: {annot_path}")
            return None

        try:
            with open(annot_path, "r") as f:
                annotations = json.load(f)

            processed_annotations = []

            # If annotations is a dict (e.g., {"0.jpeg": "...", "1.jpeg": "..."})
            if isinstance(annotations, dict):
                # Sort by numeric key before adding A–H
                sorted_items = sorted(
                    annotations.items(),
                    key=lambda x: int(x[0].split(".")[0])  # "0.jpeg" -> 0
                )
                processed_annotations = [desc for _, desc in sorted_items]

            # If it's already a list, keep as-is
            elif isinstance(annotations, list):
                processed_annotations = annotations

            else:
                print(f"Unexpected annotation format in {annot_path}")
                return None

            # Add letter prefixes (A–H)
            letters = string.ascii_uppercase
            processed_annotations = {
                letters[i]: desc for i, desc in enumerate(processed_annotations)
            }
            return processed_annotations

        except Exception as e:
            print(f"Error reading annotations for {problem_id}: {e}")
            return None
