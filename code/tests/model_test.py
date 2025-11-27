
import os
import sys
from pydantic import BaseModel

from code.models.vllm import VLLM, get_model_architecture
from code.technical.content import ImageContent, TextContent

class ResponseSchema(BaseModel):
    shape: str
    confidence: float

def main():
    print("Preparing VLLM", flush=True)
    vllm = VLLM(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        max_output_tokens=1024,
        max_tokens=2048,
        cpu_local_testing=True
    )

    print("Test 1: Text-only prompt", flush=True)
    text_content = TextContent("What is the capital of Norway?")
    response1 = vllm.ask([text_content])
    print("Response (text):", response1, flush=True)

    relative_path = "data_raw/bp/006/4.png"
    full_path = os.path.abspath(relative_path)

    if not os.path.exists(full_path):
        print(f"Image file not found: {full_path}", flush=True)
        return

    print("Test 2: Multimodal prompt", flush=True)

    text_content = TextContent("What shape do you see?")
    image_content = ImageContent(relative_path)
    response2 = vllm.ask([text_content, image_content], ResponseSchema)
    print("Response (multimodal):", response2, flush=True)

    print("Test 3: Wrong model name", flush=True)
    try:
        vllm = VLLM(
            model_name="Qwen/Qwen2.5-VL-1B-Instruct",
            custom_args=("--tensor-parallel-size", "4"),
        )
    except Exception as e:
        print("Caught exception as expected:", e, flush=True)

    models_to_test = [
        "bert-base-uncased",      
        "google/vit-base-patch16-224", 
        "Qwen/Qwen2.5-VL-3B-Instruct",  # multimodal
        "nonexistent-model-xyz"    # invalid model to test exception handling
    ]

    print(f"Test 4: Model architecture detection", flush=True)
    for model_name in models_to_test:
        info = get_model_architecture(model_name)
        print("Supported:", info["supported"])
        print("Multi-modal:", info["is_multi_modal"])
        print("Architectures:", info.get("architectures"))
        print("Model type:", info.get("model_type"))
        print("\n")

    vllm.stop()


if __name__ == "__main__":
    main()
