#!/bin/bash
#SBATCH --job-name=ghost_eval_local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/eval_local_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate

# Evaluate all local models sequentially
# Each model is loaded, evaluated, unloaded before next loads

python src/evaluate.py \
    --config config.yaml \
    --mode local \
    --models llama32_3b llama31_8b qwen25_3b \
             qwen25_7b deepseek_r1_14b \
    --conditions all

echo "Local evaluation complete"
