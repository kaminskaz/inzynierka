import pandas as pd
import json
import logging
import os
from pathlib import Path
import torch

from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge

def show_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved
    print(f"GPU allocated: {allocated:.2f} GB")
    print(f"GPU reserved:  {reserved:.2f} GB")
    print(f"GPU free:      {free:.2f} GB / {total:.2f} GB")

def main():
    print("Evaluation standard")
    evaluator = EvaluationBasic()
    evaluator.run_evaluation(
        dataset_name="cvr",
        model_name="Qwen2.5-VL-7B-Instruct",
        strategy_name="contrastive",
        version="1",
        results_dir="results/contrastive_cvr_Qwen2.5-VL-7B-Instruct_ver1",
        answers_path="results/contrastive_cvr_Qwen2.5-VL-7B-Instruct_ver1/results.csv",
        key_path="data_test/cvr/jsons/cvr_solutions.json",
        evaluation_output_path="evaluation_results",
        concat=True,
        output_all_results_concat_path="results/all_results_concat.csv"
    )

    show_gpu_memory()


    print("Evaluation with LLM Judge")
    evaluator = EvaluationWithJudge()
    evaluator.run_evaluation(
        dataset_name="bp",
        model_name="Qwen2.5-VL-7B-Instruct",
        strategy_name="direct",
        version="1",
        results_dir="results/direct_bp_Qwen2.5-VL-7B-Instruct_ver1",
        answers_path="results/direct_bp_Qwen2.5-VL-7B-Instruct_ver1/results.csv",
        key_path="data_test/bp/jsons/bp_solutions.json",
        evaluation_output_path="evaluation_results",
        concat=True,
        output_all_results_concat_path="results/all_results_concat.csv"
    )

if __name__ == "__main__":
    main()
    