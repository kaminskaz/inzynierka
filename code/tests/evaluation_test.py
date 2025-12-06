import pandas as pd
import json
import logging
import os
from pathlib import Path
import torch

from code.evaluation.evaluation_base import EvaluationBase

def main():
    print("Evaluation")
    evaluator = EvaluationBase()
    evaluator.run_multiple_evaluations(
        strategy_name=["direct", "descriptive", "contrastive", "classification"],
        dataset_name=["cvr", "bp", "raven", "marsvqa"],
        model_name=["InternVL3-8B"],
        version=["1"]
    )

if __name__ == "__main__":
    main()
    