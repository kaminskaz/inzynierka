
import os
import sys
from pydantic import BaseModel
from src.models.vllm import VLLM
from src.technical.content import ImageContent, TextContent

class ResponseSchema(BaseModel):
    shape: str
    confidence: float

def main():

    print("Test Model 1: OpenGVLab/InternVL3-38B", flush=True)
    print("Preparing VLLM", flush=True)
    vllm = VLLM(model_name="OpenGVLab/InternVL3-38B")

    relative_path = os.path.join("data", "bp", "problems", "006", "choices", "4.png")
    full_path = os.path.abspath(relative_path)

    if not os.path.exists(full_path):
        print(f"Image file not found: {full_path}", flush=True)
        return

    text_content = TextContent("What shape do you see?")
    image_content = ImageContent(relative_path)
    response1 = vllm.ask([text_content, image_content], ResponseSchema)
    print("Response (multimodal):", response1, flush=True)
    
    vllm.stop()

    print("\nTest Model 2: OpenGVLab/InternVL3-14B", flush=True)
    print("Preparing VLLM", flush=True)
    vllm = VLLM(model_name="OpenGVLab/InternVL3-14B")
    response2 = vllm.ask([text_content, image_content], ResponseSchema)
    print("Response (multimodal):", response2, flush=True)

    vllm.stop()

    print("\nTest Model 3: Qwen/Qwen2.5-VL-32B-Instruct", flush=True)
    print("Preparing VLLM", flush=True)
    vllm = VLLM(model_name="Qwen/Qwen2.5-VL-32B-Instruct")
    response3 = vllm.ask([text_content, image_content], ResponseSchema)
    print("Response (multimodal):", response3, flush=True)  

    vllm.stop()
    
    print("\nAll tests completed.", flush=True)

if __name__ == "__main__":
    main()
