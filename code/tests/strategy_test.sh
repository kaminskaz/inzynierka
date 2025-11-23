#!/bin/bash
#SBATCH -A jrafalko-lab
#SBATCH --job-name=strat_test # Tu nazywasz jakoś swój proces, byle co szczerze mało warte bo i tak po nicku ja znajduje mój task
#SBATCH --time=01:00:00 # dla short to masz max 2h dla long i experimental masz chyba 3-4 dni to jest czas po którym slurm ubja twój proces (zasada jest że>
#SBATCH --ntasks=1 # tutaj wystarczy 1 zawsze mieć chyba że chcesz multi gpu itp ale zapewne 1 GPU wam wystarczy
#SBATCH --gpus=1 # Jak nie potrzebujesz GPU to wyrzucasz tą linijke
#SBATCH --cpus-per-gpu=8 # Ile cpu na jedno gpu ma być w tym konfigu to po prostu ile cpu chcesz mieć mówiłem żeby dawać zawsze minimum 6-8 bo inaczej kole>
#SBATCH --mem=128gb # Ile ram chcesz mieć mamy dużo więc nie musisz dawać mało ale bez przesady
#SBATCH --partition=short # Tutaj podajesz short,long,experimental jedną z tych partycji z której chcesz korzystać shot i long ma A100 short max 1d long dł>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=01180698@pw.edu.pl
# Debugging flags
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Set up job-specific temporary directories
export JOB_HF_HOME="/mnt/evafs/groups/jrafalko-lab/huggingface/tmp/${SLURM_JOB_ID}"
mkdir -p ${JOB_HF_HOME}

export JOB_TMPDIR="/mnt/evafs/groups/jrafalko-lab/tmp/${SLURM_JOB_ID}"
mkdir -p ${JOB_TMPDIR}

# Activate virtual environment
source /mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin/activate
export PATH=/mnt/evafs/groups/jrafalko-lab/inzynierka/.venv/bin:$PATH

# Navigate to the project directory
cd /mnt/evafs/groups/jrafalko-lab/inzynierka/

# Run the experiment
# Added line breaks (\) for readability
python run_single_experiment.py \
    --dataset_name bp \
    --strategy direct \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --temperature 0.5 \
    --max_tokens 12000 \
    --max_output_tokens 8123 \
    --limit_mm_per_prompt 2 \
    --debug \
    --custom_args --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-num-seqs 128 --max-model-len 16246

# Clean up temporary directories
rm -rf ${JOB_HF_HOME}
rm -rf ${JOB_TMPDIR}