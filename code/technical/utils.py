import os
import json
import pandas as pd
from PIL import Image
import torch
from typing import Any, Dict, Optional
import logging
import re
from pydantic import BaseModel

from code.preprocessing.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


def get_dataset_config(dataset_name: str, config_path="code/preprocessing/dataset_config.json") -> Optional[ProcessorConfig]:
    """Gets the ProcessorConfig for a specific dataset."""
    dataset_configs_raw = load_all_dataset_configs(config_path)
    raw_config = dataset_configs_raw.get(dataset_name)
    if not raw_config:
        logger.error(
            f"No config found for dataset: '{dataset_name}' in {config_path}"
        )
        return None

    try:
        return ProcessorConfig.from_dict(raw_config)
    except Exception as e:
        logger.error(f"Error creating ProcessorConfig for {dataset_name}: {e}")
        return None

def load_all_dataset_configs(config_path="code/preprocessing/dataset_config.json") -> Dict[str, Any]:
    """Loads the main dataset_config.json file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(
            f"Failed to load dataset config file at {config_path}: {e}"
        )
        raise

def make_dir_for_results(
        dataset_name: str, 
        strategy_name: str, 
        model_name: str, 
        version: Optional[str] = None,
        create_dir: bool = True
        ) -> str:
    """
    Creates a new versioned results directory for the given dataset and strategy.
    If previous versions exist, increments the version number.
    """
    base_results_dir = "results"

    if create_dir:
        os.makedirs(base_results_dir, exist_ok=True)

    short_model_name = shorten_model_name(model_name)
    prefix = f"{strategy_name}_{dataset_name}_{short_model_name}"

    if version is not None:
        dir_name = f"{prefix}_ver{version}"
        path = os.path.join(base_results_dir, dir_name)

        if create_dir:
            os.makedirs(path, exist_ok=True)

        return path

    version_pattern = re.compile(rf"^{re.escape(prefix)}_ver(\d+)$")
    existing_versions = []

    if os.path.isdir(base_results_dir):
        for entry in os.scandir(base_results_dir):
            if entry.is_dir():
                match = version_pattern.match(entry.name)
                if match:
                    existing_versions.append(int(match.group(1)))

    new_version = max(existing_versions, default=0) + 1
    new_dir_name = f"{prefix}_ver{new_version}"
    new_dir_path = os.path.join(base_results_dir, new_dir_name)

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

def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, BaseModel):
        return getattr(obj, name, default)
    return default