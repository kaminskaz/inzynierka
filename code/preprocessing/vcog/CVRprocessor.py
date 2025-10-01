import os
from PIL import Image, ImageDraw
import json
from code.preprocessing.vcog.VCOGsheetmaker import VCOGSheetMaker

class CVRProcessor:
    def __init__(self):
        self.raw_data_folder_path = os.path.join("data_raw", "vcog-bench", "cvr")
        self.output_data_folder_path = "data"
        self.sheetmaker = VCOGSheetMaker()
        self.answers_dict = {}
        self.shuffle_orders = {}

    def process(self):
        for problem_id in os.listdir(self.raw_data_folder_path):
            problem_path = os.path.join(self.raw_data_folder_path, problem_id)
            #print(f"Processing problem: {problem_id}, problem_path: {problem_path}")
            if not os.path.isdir(problem_path):
                continue

            images = [self.load_image(problem_id, i) for i in range(4)]  # 4 images per problem

            if None in images:
                print(f"Skipping problem {problem_id} due to missing images.")
                continue

            # direct strategy
            sheet, answer, shuffle_order = self.sheetmaker.generate_question_sheet_from_images(images, shuffle_answers=True, true_answer_index=3)

            #optional: change problem_id (fill with 0 in front to make it 3 digits)
            problem_id = problem_id.zfill(3)

            self.save_sheet(problem_id, sheet, strategy="direct")

            # store answer and shuffle order
            self.answers_dict[problem_id] = answer
            self.shuffle_orders[problem_id] = shuffle_order

        # save answers to json
        #create directory with json if it doesn't exist
        os.makedirs(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","cvr", "solutions"), exist_ok=True)
        with open(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","cvr", "solutions", "cvr_solutions.json"), "w") as f:
            json.dump(self.answers_dict, f, indent=4)

        # save shuffle orders to json
        with open(os.path.join(self.output_data_folder_path, "direct", "vcog-bench","cvr", "solutions", "cvr_shuffle_orders.json"), "w") as f:
            json.dump(self.shuffle_orders, f, indent=4)

    def load_image(self, problem_id: str, image_index: int):
        image_path = os.path.join(self.raw_data_folder_path, problem_id,"choice","image", f"sub_image_{image_index+1}.png")
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def save_sheet(self, problem_id: str, sheet, strategy: str = "direct"):
        if strategy == "direct":
            save_dir = os.path.join(self.output_data_folder_path, "direct", "vcog-bench","cvr", "images")
            os.makedirs(save_dir, exist_ok=True)  # make sure directory exists
            save_path = os.path.join(save_dir, f"{problem_id}.png")
            sheet.save(save_path)
        elif strategy == "contrastive":
            pass

        