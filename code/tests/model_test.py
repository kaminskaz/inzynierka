import asyncio
from code.models.vllm import VLLMFactory
from code.technical.content import ImageContent, TextContent
import os
import sys

async def main():
    print("Preparing VLLM", flush=True)
    vllm = VLLMFactory(model_name="Qwen/Qwen2.5-VL-72B-Instruct")

    text_content = TextContent("What is the capital of Norway?")
    response1 = await vllm.ask([text_content])
    print("Response (text):", response1, flush=True)

    relative_path = "data_raw/bp/006/4.png"
    full_path = os.path.abspath(relative_path)

    if not os.path.exists(full_path):
        print(f"Image file not found: {full_path}", flush=True)
        return

    text_content = TextContent("What shape do you see?")
    image_content = ImageContent(relative_path)
    response2 = await vllm.ask([text_content, image_content])
    print("Response (multimodal):", response2, flush=True)

    vllm._stop_vllm_server()

if __name__ == "__main__":
    asyncio.run(main())