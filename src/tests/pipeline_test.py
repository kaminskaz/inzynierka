from src.base import FullPipeline

def main():
    pipeline = FullPipeline()

    print("Preparing data...")
    pipeline.prepare_data()

    print("Running experiment...")
    pipeline.run_experiment(
        dataset_name="cvr",
        strategy_name="direct",
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        param_set_number=1,
        prompt_number=1
    )

    print("Running ensemble...")
    pipeline.run_ensemble(
        dataset_name="cvr",
        members_configuration=[["direct", "Qwen/Qwen2.5-VL-3B-Instruct", "2"], ["classification", "OpenGVLab/InternVL3-8B", "1"]],
        type_name="reasoning",
        vllm_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        llm_model_name="Qwen/Qwen2.5-VL-3B-Instruct"
    )

if __name__ == "__main__":
    main()