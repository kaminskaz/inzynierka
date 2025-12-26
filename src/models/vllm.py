import json
import re
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

from src.technical.content import Content, ImageContent, TextContent
from src.technical.prompt_formatter import PromptFormatter
from src.technical.utils import get_model_config

logger = logging.getLogger(__name__)


class VLLM:
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B",
        param_set_number: Optional[int] = None
    ):
        config = get_model_config(
                model_name, param_set_number=param_set_number)

        assert config.max_tokens > config.max_output_tokens, (
            "max_tokens must be greater than max_output_tokens."
        )        

        self.model_name = model_name
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.max_output_tokens = config.max_output_tokens
        self.cpu_local_testing = config.cpu_local_testing
        if self.cpu_local_testing:
            logger.info("CPU-only mode enabled for this request. Setting schema to None for the purpose of local testing.")

        port = portpicker.pick_unused_port()
        self.api_key = "NOT-USED"
        self.base_url = f"http://localhost:{port}"

        self.process = launch_vllm_server(
            model_name,
            self.base_url,
            timeout=3600.0,
            api_key=self.api_key,
            other_args=(
                "--port",
                str(port),
                "--max-model-len",
                str(config.max_tokens),
                "--tensor-parallel-size", str(config.tensor_parallel_size),
                "--gpu-memory-utilization", str(config.gpu_memory_utilization),
                *(
                    ("--chat-template", config.chat_template_path)
                    if config.chat_template_path
                    else ()
                ),
                "--trust-remote-code",
                *(
                    ("--limit-mm-per-prompt", f'{{"image": {config.limit_mm_per_prompt}}}')
                    if config.limit_mm_per_prompt > 0
                    else ()
                ),
            ),
        )

        self.client = openai.OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            timeout=60
        )
        self.formatter = PromptFormatter()
        logger.info(
            f"vLLM client initialized for '{self.model_name}' at {self.base_url}"
        )

    def get_model_name(self) -> str:
        return self.model_name

    def ask(
        self, contents: List[Content], 
        schema: Optional[Type[BaseModel]] = None
    ) -> str:

        if self.cpu_local_testing and schema is not None:
            schema = None
            
        message = self.formatter.user_message(contents)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            extra_body={"guided_json": schema.model_json_schema()} if schema else None,
        )

        model_response = _parse_response(response.choices[0].message.content)
        return model_response if model_response else {}


    def stop(self):
        """Stop the running vLLM server."""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=120)
                logger.info(f"vLLM server for '{self.model_name}' stopped.")
        except Exception as e:
            logger.warning(f"Error while stopping vLLM server: {e}")
            sys.exit(3)


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

    with open(f"vllm_startup_error_{model.replace('/', '_')}.log", "w") as f:
        f.write(full_log)

    logger.critical(f"Failed to start vLLM server for '{model}'.")
    logger.debug(f"vLLM log excerpt (last 500 chars): {full_log[-500:]}")
    raise ValueError(f"vLLM server failed to start. See log file for details.")


def _parse_response(response):
    if isinstance(response, dict):
        return response

    if hasattr(response, "dict"):
        return response.dict()

    if isinstance(response, str):
        text = response.strip()

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": response}

    return {"raw": str(response)}
    