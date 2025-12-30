from src.base import FullPipeline
from src.technical.configs.evaluation_config import EvaluationConfig
from src.preprocessing.standard_processor import StandardProcessor
from src.preprocessing.data_module import DataModule

def main():
    pipeline = FullPipeline()

    # print("Preparing data...")
    # pipeline.prepare_data()

    # print("Running experiment...")
    # pipeline.run_experiment(
    #     dataset_name="cvr",
    #     strategy_name="direct",
    #     model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    #     param_set_number=1,
    #     prompt_number=1
    # )

    # print("Running ensemble...")
    # pipeline.run_ensemble(
    #     dataset_name="cvr",
    #     members_configuration=[["direct", "Qwen/Qwen2.5-VL-3B-Instruct", "1"], ["classification", "OpenGVLab/InternVL3-8B", "1"]],
    #     type_name="reasoning",
    #     vllm_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    #     llm_model_name="Qwen/Qwen2.5-VL-3B-Instruct"
    # )
    
    print("Running evaluation...")
    eval_config = EvaluationConfig(
        dataset_name="cvr",
        version="1",
        strategy_name="direct",
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        ensemble=False
    )
    pipeline.run_evaluation(eval_config)

    eval_config_ensemble = EvaluationConfig(
        dataset_name="cvr",
        version="1",
        type_name="reasoning",
        judge_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        ensemble=True,
    )
    pipeline.run_evaluation(eval_config_ensemble)

    
if __name__ == "__main__":
    main()