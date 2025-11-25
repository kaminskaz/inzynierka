import subprocess
import time
from typing import List, Optional, Dict, Type, Any
import openai
import portpicker
import requests
from pydantic import BaseModel
import logging
from transformers import AutoConfig, PretrainedConfig
import sys
import os

from code.technical.content import Content, ImageContent, TextContent
from code.technical.prompt_formatter import PromptFormatter

logger = logging.getLogger(__name__)


class VLLM:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        max_output_tokens: int = 4096,
        limit_mm_per_prompt: int = 2,
        custom_args: List[str] = [],
    ):

        assert max_tokens > max_output_tokens, (
            "max_tokens must be greater than max_output_tokens."
        )        

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        model_info = get_model_architecture(model_name)
        if not model_info["supported"]:
            logger.critical(
                f"Model '{model_name}' not supported. Reason: {model_info['error']}"
            )
            raise ValueError(f"Model '{model_name}' unsupported.")

        if not model_info["is_multi_modal"]:
            logger.warning(f"'{model_name}' appears to be text-only.")

        port = portpicker.pick_unused_port()
        self.api_key = "NOT-USED"
        self.base_url = f"http://localhost:{port}"

        self.process = launch_vllm_server(
            model_name,
            self.base_url,
            timeout=1800,
            api_key=self.api_key,
            other_args=(
                "--port",
                str(port),
                "--max-model-len",
                str(max_tokens),
                "--trust-remote-code",
                *(
                    ("--limit-mm-per-prompt", f'{{"image": {limit_mm_per_prompt}}}')
                    if limit_mm_per_prompt > 0
                    else ()
                ),
                "--guided-decoding-backend",
                "outlines",
                *custom_args,
            ),
        )

        self.client = openai.AsyncClient(
            base_url=f"{self.base_url}/v1", api_key=self.api_key
        )
        self.formatter = PromptFormatter()
        logger.info(
            f"vLLM client initialized for '{self.model_name}' at {self.base_url}"
        )

    def get_model_name(self) -> str:
        return self.model_name

    async def ask(
        self, contents: List[Content], schema: Optional[Type[BaseModel]] = None
    ) -> str:
        message = self.formatter.user_message(contents)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            extra_body={"guided_json": schema.model_json_schema()} if schema else None,
        )

        model_response = response.choices[0].message.content
        return model_response.strip() if model_response else ""

    async def ask_structured(
        self, contents: List[Content], schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        message = self.formatter.user_message(contents)

        response = await self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            response_format=schema,
            extra_body=dict(guided_decoding_backend="outlines"),
            timeout=600.0
        )

        model_response = response.choices[0].message
        if model_response.parsed:
            return model_response.parsed
        else:
            logger.error(f"Failed to parse model response: {model_response}")
            return None

    def stop(self):
        """Stop the running vLLM server."""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=120)
                logger.info(f"vLLM server for '{self.model_name}' stopped.")
        except Exception as e:
            logger.warning(f"Error while stopping vLLM server: {e}")


# ---------- helper functions ----------
def launch_vllm_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: str,
    other_args: tuple = (),
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    command = ["vllm", "serve", model, "--api-key", api_key, *other_args]
    logger.info(f"Starting vLLM server for model '{model}'...")
    logger.debug(f"Command: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    start_time = time.time()
    stdout_buffer = []

    while time.time() - start_time < timeout:
        if process.stdout:
            line = process.stdout.readline()
            if line:
                stdout_buffer.append(line.strip())
                if any(
                    k in line.lower()
                    for k in ["error", "exception", "oom", "cuda", "fatal"]
                ):
                    logger.error(line.strip())

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=1)
            if response.status_code == 200:
                logger.info(f"vLLM server for '{model}' is ready at {base_url}")
                return process
        except requests.RequestException:
            pass

        if process.poll() is not None:
            logger.error("vLLM process exited unexpectedly before startup finished.")
            break

        time.sleep(1)

    logger.error("vLLM server failed to start before timeout.")

    if process.poll() is None:
        process.terminate()
        time.sleep(1)

    try:
        remaining_output, _ = process.communicate(timeout=5)
        if remaining_output:
            stdout_buffer.append(remaining_output)
    except subprocess.TimeoutExpired:
        pass

    full_log = "\n".join(stdout_buffer).lower()

    if "out of memory" in full_log or "cuda" in full_log:
        reason = "GPU OUT OF MEMORY — try smaller model or reduce --max-model-len."
    elif "not found" in full_log or "no such file" in full_log:
        reason = "Model not found — check model name or path."
    elif "timeout" in full_log:
        reason = "Initialization timeout — model took too long to load."
    elif "architecture" in full_log and "not supported" in full_log:
        reason = "Model architecture not supported by this vLLM version."
    else:
        reason = "Unknown error — check vLLM logs below."

    logger.critical(f"Failed to start vLLM server for '{model}'. Reason: {reason}")
    logger.debug(f"vLLM log excerpt (last 500 chars): {full_log[-500:]}")
    raise TimeoutError(
        f"vLLM server failed to start within {timeout}s. Reason: {reason}"
    )


def get_model_architecture(model_name: str) -> Dict[str, Any]:
    MMM_KEYWORDS = (
        "vision",
        "vl",
        "llava",
        "fuyu",
        "qwen2vl",
        "paligemma",
        "internvl",
        "gemma",
    )

    try:
        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        arch_names = getattr(config, "architectures", None)
        if not arch_names and hasattr(config, "model_type"):
            arch_names = [config.model_type]

        if not arch_names:
            return {
                "supported": False,
                "is_multi_modal": False,
                "error": "Missing 'architectures' or 'model_type' in config.json.",
            }

        is_multi_modal = any(
            keyword in arch.lower() for arch in arch_names for keyword in MMM_KEYWORDS
        )

        return {
            "supported": True,
            "is_multi_modal": is_multi_modal,
            "architectures": arch_names,
            "model_type": getattr(config, "model_type", None),
            "error": None,
        }

    except Exception as e:
        return {
            "supported": False,
            "is_multi_modal": False,
            "error": f"Failed to fetch configuration for '{model_name}'. Hugging Face Error: {str(e)}",
        }
