from itertools import product
from typing import List
from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.evaluation.evaluation_base import EvaluationBase


def main():
    print("Test: Evaluation", flush=True)
    evaluator = EvaluationBasic()
    evaluator.run_evaluation(
        dataset_name="bp",
        strategy_name="direct",
        model_name="InternVL3-8B",
        version="1",
        ensemble=True
    )

if __name__ == "__main__":
    main()
