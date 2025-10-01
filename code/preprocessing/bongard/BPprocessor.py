import os
from PIL import Image, ImageDraw
import json

class BPProcessor:
    def __init__(self):
        self.raw_data_folder_path = "data_raw/bp"
        self.output_data_folder_path = "data"

    def process(self):
        for problem_id in os.listdir(self.raw_data_folder_path):
            problem_path = os.path.join(self.raw_data_folder_path, problem_id)
            #print(f"Processing problem: {problem_id}, problem_path: {problem_path}")
            if not os.path.isdir(problem_path):
                continue
            
            images = [self.load_image(problem_id, i) for i in range(12)]  # 12 images per problem

            if None in images:
                print(f"Skipping problem {problem_id} due to missing images.")
                continue

            # direct strategy
            sheet = self.generate_question_sheet_two_panels_no_labels(images)
            self.save_sheet(problem_id, sheet, strategy="direct")

        # answer renumbering
        self.renumber_answers(
            answers_json_raw_path=os.path.join("data_raw","bp","bp_solutions.json"),
            answers_json_processed_path=os.path.join(self.output_data_folder_path, "direct", "bp", "solutions", "bp_solutions.json")
        )

    def load_image(self, problem_id: str, image_index: int):
        image_path = os.path.join(self.raw_data_folder_path, problem_id, f"{image_index}.png")
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
    @staticmethod
    def generate_question_sheet_two_panels_no_labels(
            images: list[Image.Image], spacing: int = 10, margin: int = 20, border_thickness: int = 2, space_between_panels: int = 40
        ) -> Image.Image:
            
            if len(images) != 12:
                raise ValueError("Exactly 12 images are required (0-11).")

            max_width = max(img.width for img in images)
            resized_images = []
            for img in images:
                if img.width != max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height))
                resized_images.append(img)

            panels = [resized_images[:6], resized_images[6:]]

            col_width = max(img.width for img in images[:6])

            row_heights = []
            for col in range(2):
                row_heights_col = [panels[0][col*3 + r].height for r in range(3)]
                row_heights.append(row_heights_col)

            total_width = col_width*4 + spacing*3 + space_between_panels  # 2 columns per panel + spacing + panel gap
            total_height = max(
                sum([panels[0][col*3 + r].height for r in range(3)] + [panels[1][col*3 + r].height for r in range(3)]) 
                for col in range(2)
            ) + 2*margin

            sheet = Image.new('RGB', (total_width + 2*margin, total_height + 2*margin), color=(255, 255, 255))
            draw = ImageDraw.Draw(sheet)

            x_offset = margin
            y_offset = margin

            for panel_idx, panel in enumerate(panels):
                for col in range(2):
                    y_pos = margin
                    for row in range(3):
                        img_idx = col*3 + row
                        img = panel[img_idx]

                        sheet.paste(img, (x_offset, y_pos))

                        draw.rectangle(
                            [x_offset, y_pos, x_offset + img.width - 1, y_pos + img.height - 1],
                            outline="black", width=border_thickness
                        )

                        y_pos += img.height + spacing

                    x_offset += col_width + spacing

                if panel_idx == 0:
                    x_offset += space_between_panels - spacing

            return sheet


    def save_sheet(self, problem_id: str, sheet, strategy: str = "direct"):
        if strategy == "direct":
            save_dir = os.path.join(self.output_data_folder_path, "direct", "bp", "images")
            os.makedirs(save_dir, exist_ok=True)  # make sure directory exists
            save_path = os.path.join(save_dir, f"{problem_id}.png")
            sheet.save(save_path)
        elif strategy == "compressed":
            pass

    def renumber_answers(self, answers_json_raw_path: str, answers_json_processed_path: str):
        with open(answers_json_raw_path, 'r', encoding="utf-8") as f:
            answers_data = json.load(f)

        renumbered_data = {
            f"{int(k):03d}": answers_data[k] for k in sorted(answers_data.keys(), key=lambda x: int(x))
        }

        with open(answers_json_processed_path, 'w', encoding="utf-8") as f:
            json.dump(renumbered_data, f, indent=4, ensure_ascii=True)

        