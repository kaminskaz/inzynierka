# write simple test to check if vllm model works properly on a random picture and text input
import unittest
from code.models.vllm import VLLMFactory
from code.technical.content import ImageContent, TextContent
import asyncio

class TestVLLM(unittest.TestCase):

    def setUp(self):
        self.vllm = VLLMFactory(model_name="test_model")

    def test_vllm_with_text_input(self):
        text_content = TextContent("What is the capital of Norway?")
        response = asyncio.run(self.vllm.ask([text_content]))
        self.assertIn("Oslo", response)

    def test_vllm_with_image_input(self):
        text_content = TextContent("What shape do you see?")
        image_content = ImageContent("../../../data_raw/bp/006/4.png")
        contents = [text_content, image_content]
        response = asyncio.run(self.vllm.ask(contents))
        self.assertIn("triangle", response)