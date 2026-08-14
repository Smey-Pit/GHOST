#!/bin/bash
#SBATCH --job-name=ghost_inctx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/inctx_%j.log

module load python/3.11
source venv/bin/activate
python src/inctx_defense.py --config config.yaml
echo "In-context defense evaluation complete"
