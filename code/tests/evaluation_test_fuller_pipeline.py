from itertools import product
from typing import List
from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.evaluation.evaluation_base import EvaluationBase


def main():
    evaluator = EvaluationBasic()
    print("Test: Single model evaluation", flush=True)
    evaluator.run_evaluation(
        dataset_name="bp",
        strategy_name="direct",
        model_name="InternVL3-8B",
        version="1"
    )
    print("Single model evaluation completed.", flush=True) 

    print("\nTest: Ensemble evaluation", flush=True)
    evaluator.run_evaluation(
        dataset_name="bp",
        type_name="majority",
        version="1",
        ensemble=True
    )
    print("Ensemble evaluation completed.", flush=True)

if __name__ == "__main__":
    main()
