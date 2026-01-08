from src.base import FullPipeline
from src.technical.configs.evaluation_config import EvaluationConfig

def run_all_evaluations():
    pipeline = FullPipeline()

    dataset_names = ["cvr","marsvqa","raven","bp"]
    strategy_names = ["classification","direct","descriptive","contrastive"]
    versions = ["3"]
    model_names = ["Qwen/Qwen2.5-VL-7B-Instruct", "OpenGVLab/InternVL3-8B", "llava-hf/llava-v1.6-mistral-7b-hf"]
    #model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    #model_name = "OpenGVLab/InternVL3-8B"
    #model_name = "llava-hf/llava-v1.6-mistral-7b-hf"


    # type_names = ["majority","confidence","reasoning","reasoning_with_image"]
    for model_name in model_names:
        for dataset in dataset_names:
            for strategy in strategy_names:
                for version in versions:
                # for type_name in type_names:
                    print("Attempting dataset:", dataset, " version:", version, " strategy:", strategy, " model:", model_name, flush=True)
                    eval_config = EvaluationConfig(
                            dataset_name=dataset,
                            version=version,
                            strategy_name=strategy,
                            model_name=model_name,
                            ensemble=False
                        )

                    # eval_config = EvaluationConfig(
                    #         dataset_name=dataset,
                    #         version="1",
                    #         type_name=type_name,
                    #         ensemble=True
                    #     )
                    print(f"running config: {eval_config}", flush=True)
                    pipeline.run_evaluation(eval_config)

def run_ensemble_evaluations():
    type_names = ["majority","confidence","reasoning","reasoning_with_image"]
    versions = ["1","2","3","4","5","6","7","8","9"]
    datasets = ["marsvqa", "cvr","raven", "bp"]

    pipeline = FullPipeline()
    for dataset in datasets:
        for type in type_names:
            for ver in versions:
                eval_config = EvaluationConfig(
                                    dataset_name=dataset,
                                    version=ver,
                                    type_name=type,
                                    ensemble=True
                                )
                print(f"running config: {eval_config}", flush=True)
                pipeline.run_evaluation(eval_config)

def run_evals_update():
    mods = ["Qwen/Qwen2.5-VL-7B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
    strats = ["direct","descriptive","descriptive","direct"]
    vers = ["3","3","2","2"]
    datasets = ["bp", "marsvqa", "bp","bp"]

    pipeline = FullPipeline()
    for i in range(len(mods)):
        eval_config = EvaluationConfig(
                dataset_name=datasets[i],
                version=vers[i],
                strategy_name=strats[i],
                model_name=mods[i],
                ensemble=False
            )
        print(f"running config: {eval_config}", flush=True)
        pipeline.run_evaluation(eval_config)

if __name__ == "__main__":
    #run_all_evaluations()
    #run_evals_update()
    pipeline = FullPipeline()
    pipeline.run_missing_evaluations_in_directory(path="results/ensembles")