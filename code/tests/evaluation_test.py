import pandas as pd
import json
import logging
import os
from pathlib import Path
import torch

from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge

def main():
    print("Evaluation standard")
    evaluator = EvaluationBasic()
    evaluator.run_evaluation(
        dataset_name="cvr",
        model_name="InternVL3-8B",
        strategy_name="descriptive",
        version="1",
        results_dir="results/descriptive_cvr_InternVL3-8B_ver1",
        answers_path="results/descriptive_cvr_InternVL3-8B_ver1/results.csv",
        key_path="data/cvr/jsons/cvr_solutions.json",
        evaluation_output_path="evaluation_results",
        concat=True,
        output_all_results_concat_path="results/all_results_concat.csv"
    )

    print("Evaluation with LLM Judge")
    evaluator = EvaluationWithJudge()
    evaluator.run_evaluation(
        dataset_name="bp",
        model_name="InternVL3-8B",
        strategy_name="descriptive",
        version="1",
        results_dir="results/descriptive_bp_InternVL3-8B_ver1",
        answers_path="results/descriptive_bp_InternVL3-8B_ver1/results.csv",
        key_path="data/bp/jsons/bp_solutions.json",
        evaluation_output_path="evaluation_results",
        concat=True,
        output_all_results_concat_path="results/all_results_concat.csv"
    )

if __name__ == "__main__":
    main()
    