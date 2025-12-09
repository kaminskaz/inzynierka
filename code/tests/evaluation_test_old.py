from itertools import product
from typing import List
from code.evaluation.evaluation_basic import EvaluationBasic
from code.evaluation.evaluation_judge import EvaluationWithJudge
from code.technical.utils import get_dataset_config

def run_multiple_evaluations(
        strategy_names: List[str],
        dataset_names: List[str],
        model_names: List[str],
        versions: List[str],
        judge_prompt: str = None,
        evaluation_output_path: str = "evaluation_results"
    ):
        evaluator_judge = EvaluationWithJudge()
        evaluator_simple = EvaluationBasic()

        for d_name in dataset_names:
            d_category = get_dataset_config(d_name).category 
            for s_name, m_name, ver in product(strategy_names, model_names, versions):
                print(f"Evaluating dataset: {d_name} of category {d_category} for strategies: {s_name}")
                if d_category == "standard" or d_category == "choice_only":
                    evaluator = evaluator_simple
                    evaluator.run_evaluation(
                        dataset_name=d_name,
                        model_name=m_name,
                        strategy_name=s_name,
                        version=ver,
                        evaluation_output_path=evaluation_output_path,
                    )
                else:
                    evaluator = evaluator_judge
                    evaluator.run_evaluation(
                        dataset_name=d_name,
                        model_name=m_name,
                        strategy_name=s_name,
                        version=ver,
                        prompt=judge_prompt,
                        evaluation_output_path=evaluation_output_path,
                    )
        evaluator_judge.judge.stop()
        
def main():
    print("Evaluation", flush=True)
    run_multiple_evaluations(
        strategy_names=["direct", "descriptive", "contrastive", "classification"],
        dataset_names=["bp"],
        model_names=["OpenGVLab/InternVL3-8B"],
        versions=["1"]
    )

if __name__ == "__main__":
    main()
