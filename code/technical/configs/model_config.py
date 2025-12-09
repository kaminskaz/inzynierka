import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for a dataset processor."""
    model_name: str
    model_class: str
    max_tokens_limit: int
    num_params_billions: int
    gpu_split: bool = False
    temperature: float = 0.5
    max_tokens: int = 8192
    max_output_tokens: int = 4096
    limit_mm_per_prompt: int = 2
    cpu_local_testing: bool = False
    tensor_parallel_size: Optional[int] = 1
    gpu_memory_utilization: Optional[float] = 0.9
    chat_template_path: Optional[str] = None