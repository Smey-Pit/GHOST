#!/bin/bash
#SBATCH --job-name=ghost_encode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/encode_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate

# Encode all conditions sequentially
# Most conditions are cheap; vs_only and ghost are expensive
# (logprob queries against Qwen2.5-7B)

python src/encode.py \
    --config config.yaml \
    --conditions clean bidi_only homochar

python src/encode.py \
    --config config.yaml \
    --conditions vs_only

python src/encode.py \
    --config config.yaml \
    --conditions ghost

python src/encode.py \
    --config config.yaml \
    --conditions ghost_nfkc bae textfooler

echo "All encoding complete"
