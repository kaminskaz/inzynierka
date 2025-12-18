import os
import json
import pandas as pd
from PIL import Image
import torch
from typing import Any, Dict, Optional, Union
import logging
import re
from src.technical.configs.model_config import ModelConfig
from pydantic import BaseModel

from src.technical.configs.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset_config(dataset_name: str, config_path="src/technical/configs/dataset_config.json") -> Optional[DatasetConfig]:
    """Gets the DatasetConfig for a specific dataset."""
    try:
        with open(config_path, "r") as f:
           dataset_configs_raw = json.load(f)
    except Exception as e:
        logger.error(
            f"Failed to load dataset config file at {config_path}: {e}"
        )
        return None
    
    raw_config = dataset_configs_raw.get(dataset_name)
    if not raw_config:
        logger.error(
            f"No config found for dataset: '{dataset_name}' in {config_path}"
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
    create_dir: bool = True
) -> str:
    base_results_dir = "results"
    short_model_name = shorten_model_name(model_name)
    prefix = os.path.join(base_results_dir, dataset_name, strategy_name, short_model_name)

    if version == "latest":
        existing_versions = []
        if os.path.isdir(prefix):
            for entry in os.scandir(prefix):
                if entry.is_dir() and entry.name.startswith("ver"):
                    try:
                        ver_num = int(entry.name.replace("ver", ""))
                        existing_versions.append(ver_num)
                    except ValueError:
                        continue
        
        if not existing_versions:
            raise FileNotFoundError(f"No existing versions found in {prefix} to restart from.")
        
        latest_ver = max(existing_versions)
        return os.path.join(prefix, f"ver{latest_ver}")

    if version is not None and version != "":
        path = os.path.join(prefix, f"ver{version}")
        if create_dir:
            os.makedirs(path, exist_ok=True)
        return path

    existing_versions = []
    if os.path.isdir(prefix):
        for entry in os.scandir(prefix):
            if entry.is_dir() and entry.name.startswith("ver"):
                try:
                    ver_num = int(entry.name.replace("ver", ""))
                    existing_versions.append(ver_num)
                except ValueError:
                    pass

    new_version = max(existing_versions, default=0) + 1
    new_dir_path = os.path.join(prefix, f"ver{new_version}")

    if create_dir:
        os.makedirs(new_dir_path, exist_ok=True)
        logger.info(f"Results directory created at: {new_dir_path}")

    return new_dir_path

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
        
        version_pattern = re.compile(rf"^{re.escape(prefix)}_ver(\d+)$")
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
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, BaseModel):
        return getattr(obj, name, default)
    return default

def get_model_config(
    target_model_name: str, 
    param_set_number: Optional[str | int] = None
    ) -> ModelConfig:
    """
    Extracts a specific configuration for a given model name and param_set version.
    """
    model_config_path = "src/technical/configs/models_config.json"
    with open(model_config_path, "r") as f:
        json_data = json.load(f)

    if target_model_name not in json_data:
        raise ValueError(f"""Model '{target_model_name}' not found in configuration. 
                         Please provide the missing configuration in src/technical/configs/model_config.json""")
    
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
    })

    #if chat_template_path does not exist, set to None
    chat_template_path = config_dict.get("chat_template_path")
    if chat_template_path and not os.path.exists(chat_template_path):
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
    
    clean_config = {k: v for k, v in config_dict.items() if v is not None}

    return ModelConfig(**clean_config)

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
