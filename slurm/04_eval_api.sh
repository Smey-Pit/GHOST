#!/bin/bash
#SBATCH --job-name=ghost_eval_api
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/eval_api_%j.log

module load python/3.11
source venv/bin/activate

# API keys must be set as environment variables
# Set these before submitting:
# export OPENAI_API_KEY=...
# export ANTHROPIC_API_KEY=...
# export GOOGLE_API_KEY=...

python src/evaluate.py \
    --config config.yaml \
    --mode api \
    --models gpt55 gpt56_sol claude_sonnet \
             claude_opus gemini_31_pro \
    --conditions all

echo "API evaluation complete"
