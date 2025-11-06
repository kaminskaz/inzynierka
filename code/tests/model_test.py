import unittest
from code.models.vllm import VLLMFactory
from code.technical.content import ImageContent, TextContent
import asyncio
import os

class TestVLLM(unittest.TestCase):

    def setUp(self):
        self.vllm = VLLMFactory(model_name="Qwen/Qwen2.5-VL-72B-Instruct")

    def test_vllm_with_text_input(self):
        text_content = TextContent("What is the capital of Norway?")
        response = asyncio.run(self.vllm.ask([text_content]))
        self.assertIn("Oslo", response)

    def test_vllm_with_multimodal_input(self):
        relative_path = "data_raw/bp/006/4.png"
        full_path_to_check = os.path.abspath(relative_path)

        self.assertTrue(
            os.path.exists(full_path_to_check),
            f"Test Aborted: Image file not found! "
            f"Expected absolute path: {full_path_to_check}"
        )

        text_content = TextContent("What shape do you see?")
        image_content = ImageContent(relative_path)
        contents = [text_content, image_content]
        response = asyncio.run(self.vllm.ask(contents))
        self.assertIn("triangle", response)