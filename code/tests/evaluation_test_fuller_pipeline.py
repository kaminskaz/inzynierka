from itertools import product
from typing import List

from code.evaluation.evaluation_factory import EvaluationFactory


def main():
    eval_factory = EvaluationFactory()
    evaluator1 = eval_factory.create_evaluator(dataset_name="bp", ensemble=False)
    print("Test 1: Single model evaluation with judge", flush=True)
    evaluator1.run_evaluation(
        dataset_name="bp",
        strategy_name="direct",
        model_name="InternVL3-8B",
        version="1"
    )
    evaluator1.judge.stop()
    print("Single model evaluation with judge completed.", flush=True)

    print("\nTest 2: Single model evaluation basic", flush=True)
    evaluator2 = eval_factory.create_evaluator(dataset_name="raven", ensemble=False)
    evaluator2.run_evaluation(
        dataset_name="raven",
        strategy_name="direct",
        model_name="InternVL3-8B",
        version="1"
    )
    print("Single model basic evaluation completed.", flush=True)

    print("\nTest 3: Ensemble evaluation basic", flush=True)
    evaluator3 = eval_factory.create_evaluator(dataset_name="cvr", ensemble=True, type_name="majority")
    evaluator3.run_evaluation(
        dataset_name="cvr",
        type_name="majority",
        version="1",
        ensemble=True
    )
    print("Ensemble evaluation completed.", flush=True)

    print("\nTest 4: Ensemble evaluation with judge", flush=True)
    evaluator4 = eval_factory.create_evaluator(dataset_name="bp", ensemble=True, type_name="reasoning")
    evaluator4.run_evaluation(
        dataset_name="bp",
        type_name="reasoning",
        version="1",
        ensemble=True
    )
    evaluator4.judge.stop()
    print("Ensemble evaluation with judge completed.", flush=True)

if __name__ == "__main__":
    main()
