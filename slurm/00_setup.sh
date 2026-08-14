#!/bin/bash
#SBATCH --job-name=ghost_setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/setup_%j.log

module load python/3.11
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pre-download all HuggingFace models to avoid timeout during GPU jobs
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

models_to_download = [
    cfg["proxy_model"]
] + [m["hf_id"] for m in cfg["local_models"].values()]

for model_id in models_to_download:
    print(f"Downloading {model_id}...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    print(f"Done: {model_id}")
EOF

echo "Setup complete"
