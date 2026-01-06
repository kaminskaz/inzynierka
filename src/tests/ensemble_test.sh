#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=im_reasoning_test # Tu nazywasz jakoś swój proces, byle co szczerze mało warte bo i tak po nicku ja znaj>
#SBATCH --time=5:00:00 # dla short to masz max 2h dla long i experimental masz chyba 3-4 dni to jest czas po którym slu>
#SBATCH --ntasks=1 # tutaj wystarczy 1 zawsze mieć chyba że chcesz multi gpu itp ale zapewne 1 GPU wam wystarczy
#SBATCH --gpus=1 # Jak nie potrzebujesz GPU to wyrzucasz tą linijke
#SBATCH --cpus-per-gpu=6 # Ile cpu na jedno gpu ma być w tym konfigu to po prostu ile cpu chcesz mieć mówiłem żeby dawa>
#SBATCH --mem=128gb # Ile ram chcesz mieć mamy dużo więc nie musisz dawać mało ale bez przesady
#SBATCH --partition=hopper # Tutaj podajesz short,long,experimental jedną z tych partycji z której chcesz korzystać sho>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=01180698@pw.edu.pl
# Debugging flags

# ---- PARAMETERS ----
DATASET_NAME=${1:-bp}

echo "Dataset: $DATASET_NAME"

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

export JOB_HF_HOME="/mnt/evafs/groups/jrafalko-lab/huggingface_${SLURM_JOB_ID}"
mkdir -p ${JOB_HF_HOME}

export JOB_TMPDIR="/mnt/evafs/groups/jrafalko-lab/tmp_${SLURM_JOB_ID}"
mkdir -p ${JOB_TMPDIR}

source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

python -m src.tests.ensemble_test \
    --dataset_name "$DATASET_NAME"

rm -rf ${JOB_HF_HOME}
rm -rf ${JOB_TMPDIR}