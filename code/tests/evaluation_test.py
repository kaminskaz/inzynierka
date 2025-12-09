import pandas as pd
import json

from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.evaluation.evaluation_basic import EvaluationBasic

pd.set_option('display.max_colwidth', None) 

def main():
    print("Test 1: Standard evaluation", flush=True)
    answers_df = pd.read_csv('code/tests/cvr_example_results.csv', dtype={"problem_id": str}, encoding="utf-8")
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
    print(output_df)

    print("\nTest 2: Evaluation with judge", flush=True)
    answers_df_judge = pd.read_csv('code/tests/bp_example_results.csv', dtype={"problem_id": str}, encoding="utf-8")
    key_bp = {
        "001": ["Empty picture", "Not empty picture"],
        "002": ["Triangles", "Circles"],
        "003": ["Red shapes", "Blue shapes"],
    }

    evaluator_judge = EvaluationWithJudge()
    output_df_judge = answers_df_judge.copy()
    evaluator_judge.evaluate(answers_df_judge, key_bp, output_df_judge)
    print(output_df_judge)

    print("\nEnsemble evaluation", flush=True)
    answers_df_ensemble = pd.read_csv('code/tests/ensemble_eval_test.csv', dtype={"problem_id": str}, encoding="utf-8")
    key_ensemble = {
        "013": ["Empty picture", "Not empty picture"],
        "043": ["Triangles", "Circles"],
        "066": ["Red shapes", "Blue shapes"],
    }
    output_df_ensemble = answers_df_ensemble.copy()
    evaluator_judge.evaluate(answers_df_ensemble, key_ensemble, output_df_ensemble)
    print(output_df_ensemble)

    evaluator_judge.judge.stop()

    print("\nAll tests completed.", flush=True)

if __name__ == "__main__":
    main()
