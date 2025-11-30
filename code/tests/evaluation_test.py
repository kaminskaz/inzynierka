import pandas as pd
import json
import logging
import os
from pathlib import Path

from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge

def main():
    print("Evaluation standard")
    evaluator = EvaluationBasic()
    evaluator.run_evaluation(
        dataset_name="cvr",
        model_name="Qwen2.5-VL-7B-Instruct",
        strategy_name="contrastive",
        version="1",
        results_dir="results",
        answers_path="results/contrastive_cvr_Qwen2.5-VL-7B-Instruct_ver1/results.csv",
        key_path="data_test/cvr/jsons/cvr_solutions.json",
        evaluation_output_path="evaluation_results",
        concat=True,
        output_all_results_concat_path="results/all_results_concat.csv"
    )

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
    