import base64
import os
from typing import Dict, List

from code.technical.content import Content, ImageContent, TextContent, is_image_supported


class PromptFormatter:
    def user_message(self, contents: List[Content]) -> Dict:
        combined_content = []

        for content in contents:
            if isinstance(content, TextContent):
                combined_content.append(content.text)
            elif isinstance(content, ImageContent):
                _, ext = os.path.splitext(content.image_path)
                raw_ext = ext.replace(".", "")
                if is_image_supported(content.image_path):
                    with open(content.image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    combined_content.append(f"[Image: data:image/{raw_ext};base64,{img_b64}]")
                else:
                    combined_content.append("Image format not supported.")

        # Join everything into a single string
        final_content = "\n".join(combined_content)

        return [{"role": "user", "content": final_content}]

        # messages = []
        # for content in contents:
        #     if isinstance(content, ImageContent):
        #         messages.append(self._format_image_content(content))
        #     elif isinstance(content, TextContent):
        #         messages.append(self._format_text_content(content))
        # return {"role": "user", "content": messages}

    def assistant_message(self, model_response: str) -> Dict:
        return {"role": "assistant", "content": model_response}

    def _format_text_content(self, content: TextContent) -> Dict:
        return {"type": "text", "text": content.text}

    def _format_image_content(self, content: ImageContent) -> Dict:
        _, ext = os.path.splitext(content.image_path)
        raw_ext = ext.replace(".", "")
        if ImageContent.is_image_supported(content.image_path):
            with open(content.image_path, "rb") as image_file:
                image = base64.b64encode(image_file.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/{raw_ext};base64,{image}"},
            }
        
        else:
            return "Image format not supported."