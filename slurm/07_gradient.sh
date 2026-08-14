#!/bin/bash
#SBATCH --job-name=ghost_gradient
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/gradient_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate
python src/gradient.py --config config.yaml
echo "Gradient evaluation complete"
