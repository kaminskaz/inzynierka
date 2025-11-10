import asyncio
import os
import sys
from pydantic import create_model

from code.models.vllm import VLLM
from code.technical.content import ImageContent, TextContent

async def main():
    print("Preparing VLLM", flush=True)
    vllm = VLLM(model_name="Qwen/Qwen2.5-VL-72B-Instruct")

    text_content = TextContent("What is the capital of Norway?")
    response1 = await vllm.ask([text_content])
    print("Response (text):", response1, flush=True)

    relative_path = "data_raw/bp/006/4.png"
    full_path = os.path.abspath(relative_path)

    if not os.path.exists(full_path):
        print(f"Image file not found: {full_path}", flush=True)
        return
    
    schema = create_model(
        "responseSchema",
        shape=(str, ...),
        confidence=(float, None)
    )

    text_content = TextContent("What shape do you see?")
    image_content = ImageContent(relative_path)
    response2 = await vllm.ask_structured([text_content, image_content], schema)
    print("Response (multimodal):", response2, flush=True)

    vllm.stop()

if __name__ == "__main__":
    asyncio.run(main())