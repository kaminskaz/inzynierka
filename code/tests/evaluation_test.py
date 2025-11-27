import pandas as pd
import json
import logging
import os
from pathlib import Path

from code.evaluation.evaluation_basic import EvaluationBasic

def main():
    evaluator = EvaluationBasic()
    # evaluator.evaluate(
    #     results_dir="code/tests",
    #     answers_path="code/tests/cvr_example_results.csv",
    #     key_path="data_test/cvr/jsons/cvr_solutions.json",
    #     evaluation_output_path="evaluation_results"
    # )

    evaluator.run_evaluation(
        dataset_name="cvr",
        model_name="example_model",
        strategy_name="example_strategy",
        version="1",
        results_dir="code/tests",
        answers_path="code/tests/cvr_example_results.csv",
        key_path="data_test/cvr/jsons/cvr_solutions.json",
        evaluation_output_path="evaluation_results",
        concat=True,
        output_all_results_concat_path="code/tests/all_results_concat.csv"
    )

if __name__ == "__main__":
    main()
    