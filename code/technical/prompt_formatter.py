import base64
import os
from typing import Dict, List

from code.technical.content import Content, ImageContent, TextContent


class PromptFormatter:
    def user_message(self, contents: List[Content]) -> Dict:
        messages = []
        for content in contents:
            if isinstance(content, ImageContent):
                messages.append(self._format_image_content(content))
            elif isinstance(content, TextContent):
                messages.append(self._format_text_content(content))
        return {"role": "user", "content": messages}

    def assistant_message(self, model_response: str) -> Dict:
        return {"role": "assistant", "content": model_response}

    def _format_text_content(self, content: TextContent) -> Dict:
        return {"type": "text", "text": content.text}

    def _format_image_content(self, content: ImageContent) -> Dict:
        _, ext = os.path.splitext(content.image_path)
        raw_ext = ext.replace(".", "")
        with open(content.image_path, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{raw_ext};base64,{image}"},
        }
