
import os
import sys
from pydantic import BaseModel
from src.models.vllm import VLLM
from src.technical.content import ImageContent, TextContent

class ResponseSchema(BaseModel):
    shape: str
    confidence: float

def main():
    print("Preparing VLLM", flush=True)
    vllm = VLLM(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

    print("Test 1: Text-only prompt", flush=True)
    text_content = TextContent("What is the capital of Norway?")
    response1 = vllm.ask([text_content])
    print("Response (text):", response1, flush=True)

    relative_path = "data/bp/problems/006/choices/4.png"
    full_path = os.path.abspath(relative_path)

    if not os.path.exists(full_path):
        print(f"Image file not found: {full_path}", flush=True)
        return

    print("Test 2: Multimodal prompt", flush=True)
    text_content = TextContent("What shape do you see?")
    image_content = ImageContent(relative_path)
    response2 = vllm.ask([text_content, image_content], ResponseSchema)
    print("Response (multimodal):", response2, flush=True)
    
    vllm.stop()

    print("Test 3: Wrong model name", flush=True)
    vllm = VLLM(model_name="Qwen/Qwen2.5-VL-1B-Instruct")

    vllm.stop()


if __name__ == "__main__":
    main()
