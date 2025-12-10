from itertools import product
from typing import List

from code.evaluation.evaluation_factory import EvaluationFactory


def main():
    llm_judge = "Qwen/Qwen2.5-VL-3B-Instruct"
    eval_factory = EvaluationFactory()
    evaluator1 = eval_factory.create_evaluator(dataset_name="bp", ensemble=False, model_name=llm_judge)
    print("Test: Single model evaluation", flush=True)
    evaluator1.run_evaluation(
        dataset_name="bp",
        strategy_name="direct",
        model_name="InternVL3-8B",
        version="1"
    )
    evaluator1.judge.stop()
    print("Single model evaluation completed.", flush=True)

    print("\nTest: Ensemble evaluation", flush=True)
    evaluator2 = eval_factory.create_evaluator(dataset_name="bp", ensemble=True, type_name="majority", model_name=llm_judge)
    evaluator2.run_evaluation(
        dataset_name="bp",
        type_name="majority",
        version="1",
        ensemble=True
    )
    evaluator2.judge.stop()
    print("Ensemble evaluation completed.", flush=True)

if __name__ == "__main__":
    main()
