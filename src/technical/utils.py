import os
import json
import pandas as pd
from PIL import Image
import torch
from typing import Any, Dict, Optional, Union
import logging
import re
from src.technical.configs.model_config import ModelConfig
from src.technical.configs.evaluation_config import EvaluationConfig
from pydantic import BaseModel
import random
import numpy as np
import pathlib
from dotenv import load_dotenv
from typing import Optional
load_dotenv()

from src.technical.configs.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset_config(
        dataset_name: str=None,
        raw_json: bool = False
    ) -> Optional[DatasetConfig | dict]:
    """Gets the DatasetConfig for a specific dataset."""
    if dataset_name is None:
        raw_json = True
    
    resolved_config_path = get_config_path("dataset")

    try:
        with open(resolved_config_path, "r") as f:
           dataset_configs_raw = json.load(f)
    except Exception as e:
        logger.error(
            f"Failed to load dataset config file at {resolved_config_path}: {e}"
        )
        return None
    
    if raw_json:
            return dataset_configs_raw
    
    raw_config = dataset_configs_raw.get(dataset_name)
    if not raw_config:
        logger.error(
            f"No config found for dataset: '{dataset_name}' in {resolved_config_path}"
        )
        return None
    
    try:
        return DatasetConfig.from_dict(raw_config)
    except Exception as e:
        logger.error(f"Error creating DatasetConfig for {dataset_name}: {e}")
        return None

def get_results_directory(
        dataset_name: str, 
        strategy_name: str, 
        model_name: str, 
        version: Optional[str] = None,
        create_dir: bool = True,
        force_new_version: bool = False
    ) -> str:
        base_results_dir = "results"
        short_model_name = shorten_model_name(model_name)
        prefix = os.path.join(base_results_dir, dataset_name, strategy_name, short_model_name)

        existing_versions = []
        if os.path.isdir(prefix):
            for entry in os.scandir(prefix):
                if entry.is_dir() and entry.name.startswith("ver"):
                    try:
                        ver_num = int(entry.name.replace("ver", ""))
                        existing_versions.append(ver_num)
                    except ValueError:
                        continue
                    
        if force_new_version:
            new_version = max(existing_versions, default=0) + 1
            path = os.path.join(prefix, f"ver{new_version}")

        elif version == "latest":
            latest_ver = max(existing_versions) if existing_versions else 1
            path = os.path.join(prefix, f"ver{latest_ver}")
        
        elif version is not None and version != "":
            path = os.path.join(prefix, f"ver{version}")
        
        else:
            new_version = max(existing_versions, default=0) + 1
            path = os.path.join(prefix, f"ver{new_version}")

        if create_dir:
            try:
                os.makedirs(path)
                logger.info(f"Results directory created at: {path}")
            except FileExistsError:
                logger.debug(f"Results directory already exists at: {path}")


        return path

def shorten_model_name(model_name: str) -> str:
    parts = model_name.split('/')
    if len(parts) >= 3:
        short_model_name = parts[1]
    elif len(parts) == 2:
        short_model_name = parts[1]
    else:
        short_model_name = model_name
    short_model_name = short_model_name.replace('/', '_')
    return short_model_name

def get_full_model_name_from_short(short_name_to_find: str) -> str:
    """
    Reverses the shorten_model_name logic by searching through the config keys.
    """
    json_data = get_model_config(raw_json=True)
    for full_model_name in json_data.keys():
        if shorten_model_name(full_model_name) == short_name_to_find:
            return full_model_name
            
    raise ValueError(f"Could not find a full model name matching: {short_name_to_find}")

def get_ensemble_directory(
        dataset_name: str, 
        type_name: str, 
        version: Optional[str] = None,
        create_dir: bool = True, 
        ) -> str:
        # creates a new directory for the ensemble results inside results/ensembles/{dataset_name}/{type_name}/ensemble_ver{version}
        # where {version} is incremented if previous versions exist
        base_results_dir = os.path.join("results", "ensembles", dataset_name, type_name)
        if create_dir:
            os.makedirs(base_results_dir, exist_ok=True)
        prefix = f"ensemble_"

        # get if version is provided
        if version is not None:
            dir_name = f"{prefix}ver{version}"
            path = os.path.join(base_results_dir, dir_name)
            if create_dir:
                os.makedirs(os.path.join(base_results_dir, f"{prefix}ver{version}"), exist_ok=True)
                logger.info(f"Ensemble results directory created at: {path} with version specified.")
            return path
        
        version_pattern = re.compile(rf"^{re.escape(prefix)}ver(\d+)$")
        existing_versions = []
        for entry in os.scandir(base_results_dir):
            if entry.is_dir():
                match = version_pattern.match(entry.name)
                if match:
                    existing_versions.append(int(match.group(1)))
        new_version = max(existing_versions, default=0) + 1
        new_dir_name = f"{prefix}ver{new_version}"
        new_dir_path = os.path.join(base_results_dir, new_dir_name)
        if create_dir:
            os.makedirs(new_dir_path, exist_ok=True)
            logger.info(f"Ensemble results directory created at: {new_dir_path}")
            return new_dir_path
        
        return ""

def get_field(obj, name, default=None):
    if isinstance(obj, dict) and name in obj:
        return obj.get(name, default)
    if isinstance(obj, BaseModel) and hasattr(obj, name):
        return getattr(obj, name)

    visited = set()

    def _search(o):
        oid = id(o)
        if oid in visited:
            return default
        visited.add(oid)

        if isinstance(o, dict):
            for v in o.values():
                result = _search(v)
                if result is not default:
                    return result

        elif isinstance(o, BaseModel):
            for v in o.__dict__.values():
                result = _search(v)
                if result is not default:
                    return result

        elif isinstance(o, str):
            try:
                return _search(json.loads(o))
            except json.JSONDecodeError:
                pass

        elif isinstance(o, (list, tuple)):
            for item in o:
                result = _search(item)
                if result is not default:
                    return result

        return default

    return _search(obj)


def get_model_config(
    target_model_name: Optional[str] = None,
    param_set_number: Optional[str | int] = None,
    raw_json: bool = False
    ) -> Optional[ModelConfig | dict]:

    if target_model_name is None:
        raw_json = True

    resolved_config_path = get_config_path("model")

    try:
        with open(resolved_config_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model config at {resolved_config_path}: {e}")
        return None

    if raw_json:
        return json_data

    if target_model_name not in json_data:
        logger.error(f"Model '{target_model_name}' not found in {resolved_config_path}")
        return None
    
    model_attrs = json_data[target_model_name]

    if param_set_number is None:
        param_set_number = '1'
        logger.warning("\n param_set_number not provided. Defaulting to '1'.\n")

    param_set_number = str(param_set_number)
    param_sets = model_attrs.get("param_sets", {})
    if not param_set_number.isdigit():
        raise ValueError(f"param_set_number must be an integer, or a string with a digit value, got '{param_set_number}'.  Available: {list(param_sets.keys())}")
    
    if param_set_number not in param_sets:
        raise ValueError(f"Version '{param_set_number}' not found for model '{target_model_name}'. Available: {list(param_sets.keys())}")

    target_params = param_sets[param_set_number]
    custom_args = target_params.get("custom_args", {})

    config_dict = {
        "model_name": target_model_name,
        "model_class": model_attrs.get("model_class"),
        "max_tokens_limit": model_attrs.get("max_tokens_limit"),
        "num_params_billions": model_attrs.get("num_params_billions"),
        "gpu_split": model_attrs.get("gpu_split", False),
    }
    
    config_dict.update({
        "temperature": target_params.get("temperature"),
        "max_tokens": target_params.get("max_tokens"),
        "max_output_tokens": target_params.get("max_output_tokens"),
        "limit_mm_per_prompt": target_params.get("limit_mm_per_prompt"),
        "cpu_local_testing": target_params.get("cpu_local_testing"),
        "chat_template_path": target_params.get("chat_template_path"),
        "tensor_parallel_size": custom_args.get("tensor_parallel_size"),
        "gpu_memory_utilization": custom_args.get("gpu_memory_utilization"),
        "disable_sliding_window": custom_args.get("disable_sliding_window"),
    })

    chat_template_path = config_dict.get("chat_template_path")
    if chat_template_path and not os.path.exists(chat_template_path):
        logger.error(f"Chat template not found: {chat_template_path}")
    
    try:
        clean_config = {k: v for k, v in config_dict.items() if v is not None}
        return ModelConfig(**clean_config)
    except Exception as e:
        logger.error(f"Error creating ModelConfig for {target_model_name}: {e}")
        return None
    
def get_config_path(config_type: str) -> str:
    """
    Resolves the file path for a configuration type.
    Checks environment variables first, then falls back to internal defaults.
    """
    configs = {
        "dataset": {
            "env_var": "DATASET_CONFIG_JSON_PATH",
            "default": os.path.join("src", "technical", "configs", "dataset_config.json")  
        },
        "model": {
            "env_var": "MODELS_CONFIG_JSON_PATH",
            "default": os.path.join("src", "technical", "configs", "models_config.json")   
        }
    }

    if config_type not in configs:
        raise ValueError(f"Invalid config_type: {config_type}. Choose 'dataset' or 'model'.")

    cfg = configs[config_type]
    env_path = os.getenv(cfg["env_var"])

    if env_path and env_path.strip() and os.path.isfile(env_path):
        return env_path
    
    if env_path:
        logger.warning(
            f"{cfg['env_var']} is set but invalid: '{env_path}'. "
            f"Falling back to default: {cfg['default']}"
        )
    
    return cfg["default"]
    
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # File handler
        file_handler = logging.FileHandler("data_processing.log")
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def check_if_members_equal(member_a: dict, member_b: dict) -> bool:
        """
        Compares two member configurations. 
        Ignores 'member_idx' and recursively checks 'config' if present.
        """
        a = {k: v for k, v in member_a.items() if k != 'member_idx'}
        b = {k: v for k, v in member_b.items() if k != 'member_idx'}

        if a.keys() != b.keys():
            return False

        for key in a:
            val_a = a[key]
            val_b = b[key]

            if isinstance(val_a, dict) and isinstance(val_b, dict):
                if not check_if_members_equal(val_a, val_b):
                    return False
            else:
                if val_a != val_b:
                    return False
                    
        return True

def set_all_seeds(seed: int = 42):
    # 1. Base Python random module
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. OS environment (important for some hashing operations)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    
    # 5. PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # 6. Cudnn Determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seeds set to {seed} for random, numpy, and torch.")

def get_eval_config_from_path(
    path: str,
    ensemble: bool = False,
    judge_model_name: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.3",
    param_set_number: int = 1,
    prompt_number: int = 1
):
    p = pathlib.Path(path)
    parts = p.parts
    
    try:
        if ensemble:
            # Expected: results/ensembles/dataset_name/type_name/ensemble_ver{version}
            if len(parts) < 3:
                raise ValueError(f"Path too short for ensemble: {path}")
            
            dataset_name = parts[-3]
            type_name = parts[-2]
            version_part = parts[-1]
            
            if "ensemble_ver" not in version_part:
                raise ValueError(f"Missing 'ensemble_ver' prefix in: {version_part}")
                
            version = version_part.replace("ensemble_ver", "")

            eval_config = EvaluationConfig(
                dataset_name=dataset_name,
                version=version,
                type_name=type_name,
                ensemble=True,
                judge_model_name=judge_model_name,
                judge_model_object=None,
                prompt_number=prompt_number,
                judge_param_set_number=param_set_number
            )
        else:
            # Expected: results/dataset_name/strategy_name/short_model_name/ver{version}
            if len(parts) < 4:
                raise ValueError(f"Path too short for standard eval: {path}")
            
            dataset_name = parts[-4]
            strategy_name = parts[-3]
            model_name = get_full_model_name_from_short(parts[-2])
            version_part = parts[-1]
            
            if "ver" not in version_part:
                raise ValueError(f"Missing 'ver' prefix in: {version_part}")

            version = version_part.replace("ver", "")

            eval_config = EvaluationConfig(
                dataset_name=dataset_name,
                version=version,
                strategy_name=strategy_name,
                model_name=model_name,
                ensemble=False,
                judge_model_name=judge_model_name,
                judge_model_object=None,
                prompt_number=prompt_number,
                judge_param_set_number=param_set_number
            )
            
        return eval_config

    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Failed to parse EvaluationConfig from path: '{path}'. "
            f"Error: {e}. Check if the folder structure matches the expected format."
        )