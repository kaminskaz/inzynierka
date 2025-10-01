import os
import string
from PIL import Image, ImageDraw
import json
import random
import re
from code.preprocessing.vcog.VCOGsheetmaker import VCOGSheetMaker

class MARSProcessor:
    def __init__(self):
        self.raw_data_folder_path = os.path.join("data_raw", "vcog-bench", "marsvqa")
        self.output_data_folder_path = "data"
        self.sheetmaker = VCOGSheetMaker()
        self.answers_dict = {}
        self.shuffle_orders = {}
        self.shuffled_annotations = {}

    def process(self):
        for problem_id in os.listdir(self.raw_data_folder_path):
            problem_path = os.path.join(self.raw_data_folder_path, problem_id)
            # print(f"Processing problem: {problem_id}, problem_path: {problem_path}")
            if not os.path.isdir(problem_path):
                continue

            images = [self.load_image(problem_id, i) for i in range(4)]  # 4 images per problem

            if None in images:
                print(f"Skipping problem {problem_id} due to missing images.")
                continue

            question_image = self.load_question_image(problem_id)

            # direct strategy
            sheet, answer, shuffle_order = self.sheetmaker.generate_question_sheet_from_images(images, question_image=question_image, shuffle_answers=True, true_answer_index=0)

            # optional: change problem_id (fill with 0 in front to make it 3 digits)
            problem_id_old = problem_id
            problem_id = problem_id.zfill(3)

            self.save_sheet(problem_id, sheet, strategy="direct")

            # store answer and shuffle order
            self.answers_dict[problem_id] = answer
            self.shuffle_orders[problem_id] = shuffle_order
            self.shuffled_annotations[problem_id] = self.get_shuffled_annotations(problem_id_old, shuffle_order)

        # save answers to json
        # create directory with json if it doesn't exist
        os.makedirs(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","marsvqa", "solutions"), exist_ok=True)
        with open(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","marsvqa", "solutions", "marsvqa_solutions.json"), "w") as f:
            json.dump(self.answers_dict, f, indent=4)

        # save shuffle orders to json
        with open(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","marsvqa", "solutions", "marsvqa_shuffle_orders.json"), "w") as f:
            json.dump(self.shuffle_orders, f, indent=4)

        # save shuffled annotations to json
        with open(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","marsvqa", "solutions", "marsvqa_shuffled_annotations.json"), "w") as f:
            json.dump(self.shuffled_annotations, f, indent=4)

    def load_image(self, problem_id: str, image_index: int):
        image_dir = os.path.join(self.raw_data_folder_path, problem_id, "choice", "image")
        target_pattern = f"T{image_index+1}"
        
        try:
            # look for any file that contains T{index+1} in its name
            for fname in os.listdir(image_dir):
                if target_pattern in fname:
                    image_path = os.path.join(image_dir, fname)
                    return Image.open(image_path).convert("RGB")
            
            print(f"No image found in {image_dir} matching pattern {target_pattern}")
            return None

        except Exception as e:
            print(f"Error loading image from {image_dir} with pattern {target_pattern}: {e}")
            return None

        
    def load_question_image(self, problem_id: str):
        image_dir = os.path.join(self.raw_data_folder_path, problem_id, "question", "image")
        
        try:
            # find the first .jpeg file in the directory
            for fname in os.listdir(image_dir):
                if fname.lower().endswith(".jpeg"):
                    image_path = os.path.join(image_dir, fname)
                    return Image.open(image_path).convert("RGB")
            
            print(f"No .jpeg image found in {image_dir}")
            return None

        except Exception as e:
            print(f"Error loading question image from {image_dir}: {e}")
            return None

    def save_sheet(self, problem_id: str, sheet, strategy: str = "direct"):
        if strategy == "direct":
            save_dir = os.path.join(self.output_data_folder_path, "direct", "vcog-bench","marsvqa", "images")
            os.makedirs(save_dir, exist_ok=True)  # make sure directory exists
            save_path = os.path.join(save_dir, f"{problem_id}.png")
            sheet.save(save_path)
        elif strategy == "contrastive":
            pass

    def get_shuffled_annotations(self, problem_id: str, shuffle_order: list[int] = None):
        annot_path = os.path.join(self.raw_data_folder_path, problem_id, "choice", "text", "annotation.json")

        if not os.path.exists(annot_path):
            print(f"Annotation file not found: {annot_path}")
            return None

        try:
            with open(annot_path, "r") as f:
                annotations = json.load(f)

            processed_annotations = []

            # If annotations is a dict (filename -> description)
            if isinstance(annotations, dict):
                for fname, desc in annotations.items():
                    match = re.search(r'_T(\d+)_', fname)
                    if match:
                        index = int(match.group(1))
                        processed_annotations.append((index-1, desc))
            # If annotations is already a list of strings
            elif isinstance(annotations, list):
                processed_annotations = list(enumerate(annotations, start=0))
            else:
                print(f"Unexpected annotation format in {annot_path}")
                return None

            # Sort by index to ensure order
            processed_annotations.sort(key=lambda x: x[0])
            processed_annotations = [desc for idx, desc in processed_annotations]

            # Shuffle if shuffle_order is provided
            if shuffle_order:
                processed_annotations = [processed_annotations[i] for i in shuffle_order]
            else:
                random.shuffle(processed_annotations)

            # Add letter prefixes: A:, B:, C:, ...
            letters = string.ascii_uppercase
            processed_annotations = [
                f"{letters[i % 26]}: {desc}" for i, desc in enumerate(processed_annotations)
            ]

            return processed_annotations

        except Exception as e:
            print(f"Error reading annotations: {e}")
            return None
