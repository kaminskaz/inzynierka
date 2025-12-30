import pandas as pd
import json
import os

from src.evaluation.evaluation_judge import EvaluationWithJudge
from src.evaluation.evaluation_basic import EvaluationBasic


def main():
    pd.set_option('display.max_colwidth', None) 
    print("Test 1: Standard evaluation", flush=True)
    cvr_path = os.path.join('src', 'tests', 'cvr_example_results.csv')
    answers_df = pd.read_csv(cvr_path, dtype={"problem_id": str}, encoding="utf-8")
    key_cvr = {
        "000": "C",
        "002": "A",
        "061": "B",
        "008": "D",
        "007": "C",
        "132": "B",
    }

    evaluator = EvaluationBasic()
    output_df = answers_df.copy()
    evaluator.evaluate(answers_df, key_cvr, output_df)
    print(output_df.drop(columns=["rationale"]), flush=True)

    print("\nTest 2: Evaluation with judge", flush=True)
    bp_path = os.path.join('src', 'tests', 'bp_example_results.csv')
    answers_df_judge = pd.read_csv(bp_path, dtype={"problem_id": str}, encoding="utf-8")
    key_bp = {
        "001": ["Empty picture", "Not empty picture"],
        "002": ["Triangles", "Circles"],
        "003": ["Red shapes", "Blue shapes"],
    }

    evaluator_judge = EvaluationWithJudge(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
    output_df_judge = answers_df_judge.copy()
    evaluator_judge.evaluate(answers_df_judge, key_bp, output_df_judge)
    print(output_df_judge.drop(columns=["rationale"]), flush=True)

    print("\nTest 3: Ensemble evaluation", flush=True)
    ensemble_path = os.path.join('src', 'tests', 'ensemble_eval_test.csv')
    answers_df_ensemble = pd.read_csv(ensemble_path, dtype={"problem_id": str}, encoding="utf-8")
    key_ensemble = {
        "013": ["Empty picture", "Not empty picture"],
        "043": ["Triangles", "Circles"],
        "066": ["Red shapes", "Blue shapes"],
    }
    output_df_ensemble = answers_df_ensemble.copy()
    evaluator_judge.evaluate(answers_df_ensemble, key_ensemble, output_df_ensemble)
    print(output_df_ensemble, flush=True)

    evaluator_judge.judge_model_object.stop()

    print("\nAll tests completed.", flush=True)

if __name__ == "__main__":
    main()
