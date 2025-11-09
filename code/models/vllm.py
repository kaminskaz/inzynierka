import subprocess
import time
from typing import List, Optional, Dict, Type
import openai
import portpicker
import requests
from pydantic import BaseModel
import datetime
import os
import logging
from transformers import AutoConfig, PretrainedConfig
from typing import Dict, Any

from code.technical.content import Content, ImageContent, TextContent
from code.technical.prompt_formatter import PromptFormatter

logger = logging.getLogger(__name__)

class VLLM():
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 1.0,
        max_output_tokens: int = 1024
    ):

        self.model_name = model_name
        self.keep_image_history = True
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = openai.AsyncClient(base_url=f"{base_url}/v1", api_key=api_key)
        self.formatter = PromptFormatter()

    async def ask(
        self,
        contents: List[Content],
        schema: Optional[Type[BaseModel]] = None,
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
        if model_response:
            model_response = model_response.strip()

        return model_response

    async def ask_structured(
        self, 
        contents: List[Content], 
        schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        
        message = self.formatter.user_message(contents)

        response = await self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=message,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            response_format=schema,
            extra_body=dict(guided_decoding_backend="outlines"),
        )

        model_response = response.choices[0].message
        if model_response.parsed:
            return model_response.parsed
        else:
            logger.error(f"Failed to parse model response: {model_response}")
            return None
        

class VLLMFactory:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1536,
        limit_mm_per_prompt: int = 2,
        custom_args: List[str] = [],
        force_multi_modal: bool = False, 
    ):
        
        model_info = get_model_architecture(model_name)
        
        if not model_info['supported']:
            logger.critical(f"Model '{model_name}' is not supported or not found on Hugging Face Hub. Reason: {model_info['error']}")
            raise ValueError(f"Model '{model_name}' is unsupported or not found.")
        
        model_is_mmm = model_info['is_multi_modal']
        config_is_mmm_enabled = limit_mm_per_prompt > 0
        
        if force_multi_modal and not model_is_mmm:
            logger.critical(
                f"Configuration Error: Multi-modal operation was FORCED (force_multi_modal=True), "
                f"but model '{model_name}' is text-only (Architecture: {', '.join(model_info['architectures'])})."
                "Operation aborted."
            )
            raise ValueError(f"Model '{model_name}' must be multimodal to proceed.")

        if config_is_mmm_enabled and not model_is_mmm:
            logger.warning(
                f"Configuration Mismatch: Attempting to run text-only model ('{model_name}') "
                f"with multi-modal limits enabled (limit_mm_per_prompt={limit_mm_per_prompt}). "
                "Image inputs will likely be ignored or cause errors."
            )
            
        elif not config_is_mmm_enabled and model_is_mmm:
            logger.warning(
                f"Configuration Warning: Model '{model_name}' is Multi-Modal, but "
                "limit_mm_per_prompt is 0. Image processing capabilities are DISABLED."
            )

        self.is_multi_modal_configured = model_is_mmm and config_is_mmm_enabled
        
        port = portpicker.pick_unused_port()

        self.api_key = "NOT-USED"
        self.base_url = f"http://localhost:{port}"

        self.model_name = model_name
        self.has_reasoning_content = "--enable-reasoning" in custom_args
        self.max_tokens = max_tokens

        self.process = launch_vllm_server(
            model_name,
            self.base_url,
            timeout=600,
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

    def make_vllm_messengers(
        self,
        temperature: float = 1.0,
        max_output_tokens: int = 1536,
        n: int = 1
    ) -> List[VLLM]:
        assert max_output_tokens < self.max_tokens

        return [
            VLLM(
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model_name,
                has_reasoning_content=self.has_reasoning_content,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                is_multi_modal=self.is_multi_modal_configured, 
                log_suffix=f"-agent-{index}",
            )
            for index in range(n)
        ]
    
def stop_vllm_server(process):
    """Stops the running vLLM server if active."""
    try:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
            logger.info("vLLM server stopped successfully.")
    except Exception as e:
        logger.warning(f"Error while stopping vLLM server: {e}")


def launch_vllm_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: str,
    other_args: tuple = (),
    env: Optional[Dict[str, str]] = None,
):
    command = ["vllm", "serve", model, "--api-key", api_key, *other_args]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  
        text=True,
        env=env,
    )

    start_time = time.time()
    stdout_lines = []

    while time.time() - start_time < timeout:
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    stdout_lines.append(line)
                    logger.info(line.strip())  

        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {api_key}",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                logger.info(f"VLLM server for '{model}' is ready on {base_url}.")
                return process
        except requests.RequestException:
            pass

        if process.poll() is not None:
            break

        time.sleep(1)

    if process.poll() is None:
        process.terminate()
        time.sleep(1)

    try:
        remaining_output, _ = process.communicate(timeout=5)
        if remaining_output:
            stdout_lines.append(remaining_output)
            logger.info(remaining_output)
    except subprocess.TimeoutExpired:
        logger.warning("Could not read remaining stdout from vLLM process.")

    full_output = "".join(stdout_lines).lower()
    error_detail = "Could not determine exact cause from vLLM output. Check full logs."

    if "out of memory" in full_output or "oom" in full_output or "cuda memory" in full_output:
        error_detail = "GPU OUT OF MEMORY (OOM) Error. Try reducing max-model-len or using a smaller model."
    elif "architecture" in full_output and "not supported" in full_output:
        error_detail = "MODEL ARCHITECTURE NOT SUPPORTED by this vLLM version."
    elif "error" in full_output or "exception" in full_output or "traceback" in full_output:
        error_detail = f"VLLM internal error/dependency issue. Full log excerpt:\n{full_output[:500]}..."

    logger.critical(f"VLLM Server launch FAILED for '{model}'. Reason: {error_detail}")
    raise TimeoutError(f"Server failed to start within the timeout period. Reason: {error_detail}")


def get_model_architecture(model_name: str) -> Dict[str, Any]:
    """
    Checks the model configuration from Hugging Face Hub for basic support 
    and multi-modal capabilities without downloading weights.
    """
    MMM_KEYWORDS = ('vision', 'vl', 'llava', 'fuyu', 'qwen2vl', 'paligemma', 'internvl', 'gemma')

    try:
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Basic Architecture Check
        if not hasattr(config, 'architectures') or not config.architectures:
            return {
                'supported': False,
                'is_multi_modal': False,
                'error': "Missing 'architectures' field in config.json."
            }
        
        # Multi-Modal Capability Check
        is_multi_modal = any(
            keyword in arch.lower() 
            for arch in config.architectures 
            for keyword in MMM_KEYWORDS
        )
        
        return {
            'supported': True,
            'is_multi_modal': is_multi_modal,
            'architectures': config.architectures,
            'error': None
        }

    except Exception as e:
        return {
            'supported': False,
            'is_multi_modal': False,
            'error': f"Failed to fetch configuration for '{model_name}'. Hugging Face Error: {str(e)}"
        }