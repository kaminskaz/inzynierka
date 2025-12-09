import pandas as pd
import json

from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.evaluation.evaluation_basic import EvaluationBasic


def main():
    print("Test 1: Standard evaluation", flush=True)
    answers_df = pd.read_csv('code/tests/cvr_example_results.csv')
    key_cvr = {
        "135": "C",
        "307": "A",
        "061": "B",
        "095": "D",
        "300": "C",
        "132": "B",
    }


    evaluator = EvaluationBasic()
    output_df = answers_df.copy()
    evaluator.evaluate(answers_df, key_cvr, output_df)
    print(output_df)

    print("\nTest 2: Evaluation with judge", flush=True)
    answers_df_judge = pd.read_csv('code/tests/bp_example_results.csv')
    key_bp = {
        "001": "Empty picture vs Not empty picture",
        "002": "Triangles vs Circles",
        "003": "Red shapes vs Blue shapes",
    }
    evaluator_judge = EvaluationWithJudge(model_name='Qwen/Qwen2.5-VL-3B-Instruct')
    output_df_judge = answers_df_judge.copy()
    evaluator_judge.evaluate(answers_df_judge, key_bp, output_df_judge)
    print(output_df_judge)

    print("\nEnsemble evaluation", flush=True)
    answers_df_ensemble = pd.read_csv('code/tests/ensemble_eval_test.csv')
    key_ensemble = {
        "013": "Single shape vs Multiple shapes",
        "043": "Nested shapes vs Non-nested shapes",
        "066": "Symmetric shapes vs Asymmetric shapes",
    }
    output_df_ensemble = answers_df_ensemble.copy()
    evaluator_judge.evaluate(answers_df_ensemble, key_ensemble, output_df_ensemble)
    print(output_df_ensemble)

    evaluator_judge.judge.stop()

    print("\nAll tests completed.", flush=True)

if __name__ == "__main__":
    main()
