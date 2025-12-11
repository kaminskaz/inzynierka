from itertools import product
from typing import List

from code.evaluation.evaluation_factory import EvaluationFactory
from code.models.llm_judge import LLMJudge


def main():
    eval_factory = EvaluationFactory()
    llm_judge = LLMJudge('Qwen/Qwen2.5-VL-3B-Instruct')
    evaluator1 = eval_factory.create_evaluator(dataset_name="bp", ensemble=False, model_object=llm_judge)
    print("Test 1: Single model evaluation with judge", flush=True)
    evaluator1.run_evaluation(
        dataset_name="bp",
        strategy_name="direct",
        model_name="InternVL3-8B",
        version="1"
    )
    print("Single model evaluation with judge completed.", flush=True)

    print("\nTest 2: Single model evaluation basic", flush=True)
    evaluator2 = eval_factory.create_evaluator(dataset_name="cvr", ensemble=False)
    evaluator2.run_evaluation(
        dataset_name="cvr",
        strategy_name="classification",
        model_name="InternVL3-8B",
        version="1"
    )
    print("Single model basic evaluation completed.", flush=True)

    print("\nTest 3: Ensemble evaluation basic", flush=True)
    evaluator3 = eval_factory.create_evaluator(dataset_name="cvr", ensemble=True, type_name="majority", model_object=llm_judge)
    evaluator3.run_evaluation(
        dataset_name="cvr",
        type_name="majority",
        version="1",
        ensemble=True
    )
    print("Ensemble evaluation completed.", flush=True)

    print("\nTest 4: Ensemble evaluation with judge", flush=True)
    evaluator4 = eval_factory.create_evaluator(dataset_name="bp", ensemble=True, type_name="reasoning", model_object=llm_judge)
    evaluator4.run_evaluation(
        dataset_name="bp",
        type_name="majority",
        version="1",
        ensemble=True
    )
    llm_judge.stop()
    print("Ensemble evaluation with judge completed.", flush=True)

if __name__ == "__main__":
    main()
