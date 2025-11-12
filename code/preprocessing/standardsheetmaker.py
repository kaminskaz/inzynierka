from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

random.seed(42)


class StandardSheetMaker:
    @staticmethod
    def generate_question_sheet_from_images(
        images,
        label_font_size=40,
        spacing=30,
        margin=20,
        question_image=None,
        border_thickness=2,
        shuffle_answers=False,
        true_answer_index=None,
        blackout=None,  # <-- can be int or None
        no_label: bool = False,
    ) -> tuple[Image.Image, str | None, list[int] | None]:

        if shuffle_answers:
            if true_answer_index is None or not (0 <= true_answer_index < len(images)):
                raise ValueError(
                    "true_answer_index must be provided and valid when shuffle_answers is True."
                )
            indices = list(range(len(images)))
            random.shuffle(indices)
            shuffled_images = [images[i] for i in indices]
            answer_label = chr(ord("A") + indices.index(true_answer_index))
            images = shuffled_images
            shuffle_order = indices
            # If blackout was provided, update its new position after shuffle
            if isinstance(blackout, int):
                blackout = indices.index(blackout)
        else:
            answer_label = None
            shuffle_order = None

        # Determine layout
        num_images = len(images)
        if num_images not in [2, 3, 4, 8]:
            raise ValueError("Currently only supports 2, 3, 4, or 8 answer images.")

        two_rows = num_images == 8

        # Uniform height resizing
        max_height = max(img.height for img in images)
        resized_images = []
        for img in images:
            if img.height != max_height:
                ratio = max_height / img.height
                new_width = int(img.width * ratio)
                img = img.resize((new_width, max_height))
            resized_images.append(img)

        # Apply blackout if needed
        if isinstance(blackout, int) and 0 <= blackout < len(resized_images):
            black_img = Image.new("RGB", resized_images[blackout].size, color=(0, 0, 0))
            resized_images[blackout] = black_img

        # Compute layout sizes
        if two_rows:
            row1 = resized_images[:4]
            row2 = resized_images[4:]
            row_width = sum(img.width for img in row1) + spacing * (len(row1) - 1)
            total_width = row_width
            total_height = (max_height + label_font_size + 10) * 2 + spacing
        else:
            total_width = sum(img.width for img in resized_images) + spacing * (
                len(resized_images) - 1
            )
            total_height = max_height + label_font_size + 10

        question_img_height = 0
        if question_image:
            question_img_height = question_image.height + spacing
            total_height += question_img_height

        # Create sheet
        sheet = Image.new(
            "RGB",
            (total_width + 2 * margin, total_height + 2 * margin),
            color=(255, 255, 255),
        )

        try:
            font = ImageFont.truetype("arial.ttf", label_font_size)
        except:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(sheet)

        y_offset = margin
        if question_image:
            x_center = margin + (total_width - question_image.width) // 2
            sheet.paste(question_image, (x_center, y_offset))
            draw.rectangle(
                [
                    x_center,
                    y_offset,
                    x_center + question_image.width - 1,
                    y_offset + question_image.height - 1,
                ],
                outline="black",
                width=border_thickness,
            )
            y_offset += question_img_height

        labels = [chr(ord("A") + i) for i in range(num_images)]

        if no_label:
            labels = [""] * num_images

        def draw_row(imgs, labels_row, y_offset):
            x_offset = margin
            for i, img in enumerate(imgs):
                bbox = draw.textbbox((0, 0), labels_row[i], font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                draw.text(
                    (x_offset + (img.width - w) // 2, y_offset),
                    labels_row[i],
                    fill="black",
                    font=font,
                )
                sheet.paste(img, (x_offset, y_offset + label_font_size + 10))
                draw.rectangle(
                    [
                        x_offset,
                        y_offset + label_font_size + 10,
                        x_offset + img.width - 1,
                        y_offset + label_font_size + 10 + img.height - 1,
                    ],
                    outline="black",
                    width=border_thickness,
                )
                x_offset += img.width + spacing

        if two_rows:
            draw_row(resized_images[:4], labels[:4], y_offset)
            y_offset += max_height + label_font_size + 10 + spacing
            draw_row(resized_images[4:], labels[4:], y_offset)
        else:
            draw_row(resized_images, labels, y_offset)

        return sheet, answer_label, shuffle_order

    @staticmethod
    def generate_question_filled(
        question_image: Image.Image,
        choice_images: list[Image.Image],
        dataset_folder: str,
        problem_id_standardized: str,
        output_base_path: Path,
        margin: int = 0,
        crop_px: int = 0,
    ) -> list[Image.Image]:
        """
        Generate new images where each is the question image with a choice image
        pasted in the bottom-right corner (no resizing). Keeps the order the same as the input.

        Args:
            question_image: The main question image.
            choice_images: A list of choice images.
            margin: Padding from the right and bottom edges when pasting.

        Returns:
            A list of new PIL.Image objects in the same order as `choice_images`.
        """
        images_list = []
        dataset_name = Path(dataset_folder).name
        results = []
        for choice in choice_images:
            # Copy base question image
            base = question_image.copy()
            # Apply cropping to choice (cut from top and left only)
            choice = choice.crop((crop_px, crop_px, choice.width, choice.height))

            # Compute position for bottom-right paste
            x = base.width - choice.width - margin
            y = base.height - choice.height - margin

            # Paste choice image directly
            base.paste(choice, (x, y))

            results.append(base)

        output_dir = (
            output_base_path
            / dataset_name
            / "problems"
            / problem_id_standardized
            / "filled_choices"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(results):
            label = chr(ord("A") + i)
            save_path = output_dir / f"{label}.png"
            img.save(save_path)
            images_list.append(img)

        return images_list
