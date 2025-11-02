from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch


class QwenModel:
    def __init__(self):
        print("Loading model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
        )
        print("Model loaded")

        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels = 256 * 28 * 28, max_pixels = 1024 * 28 * 28)
        print("Processor loaded")

        self.model_name = "Qwen2.5-VL-72B-Instruct"
        self.max_tokens = 256
        
    
    def text_image_prompt(self, text, image):
        '''
        Function to generate response based on a text and image prompt.
        
        Args:
            text (str): The text prompt.
            image (PIL.Image): The image prompt.
        '''

        messages = [
            {
                "role": "user",
                "content": [
                    { "type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")

        # inference
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_tokens
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return output_text[0]


    def text_prompt(self, text):
        '''
        Function to generate response based on a prompt.
        
        Args:
            text (str): The text prompt.
        '''
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")

        # inference
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_tokens
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return output_text[0]


