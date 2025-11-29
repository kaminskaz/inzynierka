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
    
def _get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, BaseModel):
        return getattr(obj, name, default)
    return default