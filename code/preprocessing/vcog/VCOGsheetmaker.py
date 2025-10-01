from PIL import Image, ImageDraw, ImageFont
import random
random.seed(42)

class VCOGSheetMaker:
    @staticmethod
    def generate_question_sheet_from_images(
        images, label_font_size=40, spacing=30, margin=20, question_image=None, border_thickness=2, 
        shuffle_answers=False, true_answer_index=None)-> tuple[Image.Image, str, list[int]|None]:
        
        if shuffle_answers:
            if true_answer_index is None or not (0 <= true_answer_index < len(images)):
                raise ValueError("true_answer_index must be provided and valid when shuffle_answers is True.")
            indices = list(range(len(images)))
            random.shuffle(indices)
            shuffled_images = [images[i] for i in indices]
            answer_label = chr(ord('A') + indices.index(true_answer_index))
            images = shuffled_images
            shuffle_order = indices

        max_height = max(img.height for img in images)
        resized_images = []
        for img in images:
            if img.height != max_height:
                ratio = max_height / img.height
                new_width = int(img.width * ratio)
                img = img.resize((new_width, max_height))
            resized_images.append(img)

        total_width = sum(img.width for img in resized_images) + spacing * (len(resized_images)-1)
        total_height = max_height + label_font_size + 10

        question_img_height = 0
        if question_image:
            # keep original resolution, just add its height
            question_img_height = question_image.height + spacing
            total_height += question_img_height

        # create sheet with margin
        sheet = Image.new('RGB', (total_width + 2*margin, total_height + 2*margin), color=(255,255,255))

        try:
            font = ImageFont.truetype("arial.ttf", label_font_size) # type: ignore
        except:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(sheet)

        y_offset = margin
        if question_image:
            # center question image without resizing
            x_center = margin + (total_width - question_image.width) // 2
            sheet.paste(question_image, (x_center, y_offset))
            # border around question image
            draw.rectangle(
                [x_center, y_offset, x_center + question_image.width - 1, y_offset + question_image.height - 1],
                outline="black", width=border_thickness
            )
            y_offset += question_img_height

        x_offset = margin
        labels = ['A','B','C','D']

        for i, img in enumerate(resized_images):
            bbox = draw.textbbox((0,0), labels[i], font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text((x_offset + (img.width - w)//2, y_offset), labels[i], fill='black', font=font)
            sheet.paste(img, (x_offset, y_offset + label_font_size + 10))
            # draw border around answer image
            draw.rectangle(
                [x_offset, y_offset + label_font_size + 10, x_offset + img.width - 1, y_offset + label_font_size + 10 + img.height - 1],
                outline="black", width=border_thickness
            )
            x_offset += img.width + spacing

        return sheet, answer_label, shuffle_order
    
    @staticmethod
    def generate_question_sheet_raven(images: list[Image.Image], label_font_size=40, spacing=30, margin=20, 
                                      question_image=None, border_thickness=2):
        pass