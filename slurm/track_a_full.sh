#!/bin/bash
#SBATCH --job-name=track_a_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=03:45:00
#SBATCH --partition=gpu-a100-short
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/track_a_full_%j.log

# Track A Phase 2 (TRACK_A_SPEC.md): 250 documents/domain, 1000 total.
# ~1 min/instance observed in Phase 1 pilot -> ~16-17h expected, far
# longer than any single job on this partition can run.
#
# gpu-a100 (7-day limit) had a >600-job backlog and this account's low
# current fair-share put the scheduler's estimated start over a week
# out -- a dead end for a job that needs to run soon. gpu-a100-short
# (4h limit) is turning jobs over within seconds/minutes in practice,
# so this chains multiple short jobs back-to-back instead.
#
# IMPORTANT: if SLURM kills the job at its hard time limit, the WHOLE
# batch script -- including anything written after the python command
# below -- is killed too, so a plain "run python, then check-and-
# resubmit" pattern silently breaks the chain the first time a job
# actually runs out of time (which is the normal case here, not an
# edge case). --signal=B:USR1@180 asks SLURM to send SIGUSR1 to this
# script 180s before the hard kill; the trap below catches it and
# resubmits pre-emptively, then lets the python subprocess get killed
# a few seconds later by the real time limit. Every instance is
# checkpointed as it completes (src/track_a_generate.py), so losing
# the in-flight instance to that kill is expected and harmless.
#
# `python ... &` + `wait` (not running python in the foreground
# directly) is required for the trap to fire promptly -- bash only
# delivers a trapped signal between commands when it's not blocked on
# a foreground child; backgrounding + `wait` lets the signal interrupt
# `wait` immediately instead of queuing behind python's exit.
#
# RESUBMIT_COUNT_FILE caps the chain length as a backstop against a
# permanently-broken run (e.g. a bug that crashes before any progress)
# resubmitting forever -- 12 hops * ~3.75h = 45h, well above the
# ~16-17h estimate, but bounded rather than infinite.

module load python/3.11
source venv/bin/activate

OUTPUT_DIR="data/track_a/full"
DONE_SENTINEL="$OUTPUT_DIR/.generation_complete.full"
RESUBMIT_COUNT_FILE="$OUTPUT_DIR/.resubmit_count"
MAX_RESUBMITS=12

mkdir -p "$OUTPUT_DIR"

resubmit_if_not_done() {
    if [ -f "$DONE_SENTINEL" ]; then
        return 0
    fi
    COUNT=0
    if [ -f "$RESUBMIT_COUNT_FILE" ]; then
        COUNT=$(cat "$RESUBMIT_COUNT_FILE")
    fi
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -gt "$MAX_RESUBMITS" ]; then
        echo "Hit MAX_RESUBMITS=$MAX_RESUBMITS without completing -- not resubmitting. " \
             "Check logs/track_a_full_*.log and generation_log.jsonl for a stuck/failing " \
             "instance before resubmitting manually (sbatch slurm/track_a_full.sh)."
        return 1
    fi
    echo "$COUNT" > "$RESUBMIT_COUNT_FILE"
    echo "Not complete yet (resubmit $COUNT/$MAX_RESUBMITS) -- resubmitting."
    sbatch slurm/track_a_full.sh
}

trap 'echo "Caught pre-timeout warning signal -- resubmitting before hard kill."; resubmit_if_not_done; exit 0' USR1

python src/track_a_generate.py --phase full --config config.yaml &
PYTHON_PID=$!
wait $PYTHON_PID

# Reached only if python exited on its own (finished, or crashed) before the
# time-limit signal ever fired.
resubmit_if_not_done
if [ -f "$DONE_SENTINEL" ]; then
    echo "Track A Phase 2 generation complete."
fi
