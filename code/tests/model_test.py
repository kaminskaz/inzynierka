import asyncio
import os
import sys
from pydantic import create_model

from code.models.vllm import VLLM
from code.technical.content import ImageContent, TextContent


async def main():
    print("Preparing VLLM", flush=True)
    vllm = VLLM(model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                custom_args=("--tensor-parallel-size", "4"))
    
    print("Test 1: Text-only prompt", flush=True)
    text_content = TextContent("What is the capital of Norway?")
    response1 = await vllm.ask([text_content])
    print("Response (text):", response1, flush=True)

    relative_path = "data_raw/bp/006/4.png"
    full_path = os.path.abspath(relative_path)

    if not os.path.exists(full_path):
        print(f"Image file not found: {full_path}", flush=True)
        return
    
    print("Test 2: Multimodal prompt", flush=True)
    schema = create_model(
        "responseSchema",
        shape=(str, ...),
        confidence=(float, None)
    )

    text_content = TextContent("What shape do you see?")
    image_content = ImageContent(relative_path)
    response2 = await vllm.ask_structured([text_content, image_content], schema)
    print("Response (multimodal):", response2, flush=True)

    print("Test 3: Wrong model name", flush=True)
    try:
        vllm = VLLM(model_name="Qwen/Qwen2.5-VL-10B-Instruct",
                custom_args=("--tensor-parallel-size", "4"))
    except Exception as e:
        print("Caught exception as expected:", e, flush=True)

    vllm.stop()


if __name__ == "__main__":
    asyncio.run(main())
