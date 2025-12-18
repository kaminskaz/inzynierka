import os
from typing import Set, Optional

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png"}


class Content:
    def __init__(self):
        pass


class TextContent(Content):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __str__(self) -> str:
        return f"{self.__class__.__name__} - text: {self.text}"


class ImageContent(Content):
    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def __str__(self) -> str:
        return f"{self.__class__.__name__} - image_path: {self.image_path}"


def is_image_supported(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    ext = ext[1:]
    return ext in SUPPORTED_IMAGE_EXTENSIONS
