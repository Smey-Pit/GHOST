#!/bin/bash
#SBATCH --job-name=ghost_dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/dataset_%j.log

module load python/3.11
source venv/bin/activate
python src/dataset.py --config config.yaml
echo "Dataset generation complete"
