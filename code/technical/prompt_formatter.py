import base64
import os
from typing import Dict, List, Optional
import logging

from code.technical.content import Content, ImageContent, TextContent, is_image_supported

logger = logging.getLogger(__name__)

class PromptFormatter:

    def user_message(self, contents: List[Content]) -> List[Dict]:
        formatted_contents = []

        for content in contents:
            if isinstance(content, TextContent):
                formatted_contents.append(self._format_text_content(content))
            elif isinstance(content, ImageContent):
                formatted = self._format_image_content(content)
                if formatted is not None:
                    formatted_contents.append(formatted)
                else:
                    logger.warning(f"Unsupported image format skipped: {content.image_path}")

        return [{"role": "user", "content": formatted_contents}]

    def assistant_message(self, model_response: str) -> Dict:
        return {"role": "assistant", "content": model_response}

    def _format_text_content(self, content: TextContent) -> Dict:
        return {"type": "text", "text": content.text}

    def _format_image_content(self, content: ImageContent) -> Optional[Dict]:
        _, ext = os.path.splitext(content.image_path)
        raw_ext = ext.replace(".", "")

        if is_image_supported(content.image_path):
            with open(content.image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/{raw_ext};base64,{image_b64}"},
            }
        else:
            return None
