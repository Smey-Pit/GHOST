#!/bin/bash
# submit_all.sh
# Run from ghost/ directory after completing Phase 0 setup.
# Set API keys as environment variables before running.

set -e  # exit on any error

echo "Submitting GHOST experiment pipeline..."

# Check API keys are set
: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"
: "${ANTHROPIC_API_KEY:?Need ANTHROPIC_API_KEY}"
: "${GOOGLE_API_KEY:?Need GOOGLE_API_KEY}"

# Phase 1: Dataset
JOB1=$(sbatch --parsable slurm/01_dataset.sh)
echo "Phase 1 (dataset): job $JOB1"

# Phase 2: Encoding (depends on Phase 1)
JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    slurm/02_encode.sh)
echo "Phase 2 (encode): job $JOB2"

# Phase 3a: Local evaluation (depends on Phase 2)
JOB3A=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    slurm/03_eval_local.sh)
echo "Phase 3a (eval local): job $JOB3A"

# Phase 3b: API evaluation (depends on Phase 2)
JOB3B=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    --export=ALL,OPENAI_API_KEY,ANTHROPIC_API_KEY,GOOGLE_API_KEY \
    slurm/04_eval_api.sh)
echo "Phase 3b (eval API): job $JOB3B"

# Phase 4: In-context defense (depends on Phase 3b)
JOB4=$(sbatch --parsable \
    --dependency=afterok:$JOB3B \
    --export=ALL,OPENAI_API_KEY,ANTHROPIC_API_KEY \
    slurm/05_inctx_defense.sh)
echo "Phase 4 (in-context defense): job $JOB4"

# Phase 5: Normalization attack (depends on Phase 2)
# Run local + API in parallel
JOB5=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    slurm/06_norm_attack.sh)
echo "Phase 5 (normalization attack): job $JOB5"

# Phase 6: Prior strength gradient (depends on Phase 2)
JOB6=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    slurm/07_gradient.sh)
echo "Phase 6 (gradient): job $JOB6"

# Phase 7: Metrics (depends on all evaluation phases)
sbatch \
    --dependency=afterok:$JOB3A:$JOB3B:$JOB4:$JOB5:$JOB6 \
    slurm/08_metrics.sh
echo "Phase 7 (metrics): submitted with all dependencies"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
echo "Logs in: logs/"
