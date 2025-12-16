#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=inz # Tu nazywasz jakoś swój proces, byle co szczerze mało warte bo i tak po nicku ja znajduje mój task
#SBATCH --time=1:00:00 # dla short to masz max 2h dla long i experimental masz chyba 3-4 dni to jest czas po którym slurm ubja tw>
#SBATCH --ntasks=1 # tutaj wystarczy 1 zawsze mieć chyba że chcesz multi gpu itp ale zapewne 1 GPU wam wystarczy
#SBATCH --gpus=1 # Jak nie potrzebujesz GPU to wyrzucasz tą linijke
#SBATCH --cpus-per-gpu=4 # Ile cpu na jedno gpu ma być w tym konfigu to po prostu ile cpu chcesz mieć mówiłem żeby dawać zawsze mi>
#SBATCH --mem=128gb # Ile ram chcesz mieć mamy dużo więc nie musisz dawać mało ale bez przesady
#SBATCH --partition=short # Tutaj podajesz short,long,experimental jedną z tych partycji z której chcesz korzystać shot i long ma>

# ---- PARAMETERS ----
DATASET_NAME=${1:-cvr}
STRATEGY=${2:-direct}
MODEL_NAME=${3:-"OpenGVLab/InternVL3-8B"}
RESTART_PROBLEM_ID=${4:-""}
PARAM_SET_NUMBER=${5:-"1"}

echo "Dataset: $DATASET_NAME"
echo "Strategy: $STRATEGY"
echo "Model: $MODEL_NAME"
echo "Restart Problem ID: $RESTART_PROBLEM_ID"
echo "Parameter Set Number: $PARAM_SET_NUMBER"

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=DEBUG

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

cd /mnt/evafs/groups/jrafalko-lab/inzynierka

python -m code.tests.strategy_test \
    --dataset_name "$DATASET_NAME" \
    --strategy "$STRATEGY" \
    --model_name "$MODEL_NAME" \
    --restart_problem_id "$RESTART_PROBLEM_ID" \
    --param_set_number "$PARAM_SET_NUMBER"
    