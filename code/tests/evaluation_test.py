import pandas as pd
import json
import logging
import os
from pathlib import Path
import torch

from code.evaluation.evaluation_base import EvaluationBase
from run_single_experiment import run_multiple_evaluations

def main():
    print("Evaluation")
    run_multiple_evaluations(
        strategy_names=["direct", "descriptive", "contrastive", "classification"],
        dataset_names=["cvr", "bp", "raven", "marsvqa"],
        model_names=["OpenGVLab/InternVL3-8B"],
        versions=["1", "2"]
    )

if __name__ == "__main__":
    main()
