python run_single_experiment.py \
    --dataset_name bp \
    --strategy direct \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --temperature 0.5 \
    --max_tokens 2048 \
    --max_output_tokens 1024 \
    --limit_mm_per_prompt 2 \
    --debug \
    --local_testing True