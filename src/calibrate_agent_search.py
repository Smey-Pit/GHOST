"""
src/calibrate_agent_search.py

Calibrates GHOST-Agent's search-time STOPPING heuristic against a real
frontier model, without contaminating the transferability claim for
that model.

THE PROBLEM THIS ADDRESSES: run_ghost_agent's search loop stops the
moment config.yaml's agent_ensemble.consensus_threshold (3 of 4 cheap
local models) reports "defended" -- observed in practice to converge in
1 iteration on real fields/sentences this session, well before anything
resembling a maximally complex encoding. The obvious fix -- consult a
real frontier model (e.g. gpt56_sol, which the user manually confirmed
extracts a real converged encoding via the ChatGPT UI) during search and
require IT to also fail -- would optimise the search directly against
that model, which then can't be reported as a held-out transferability
result for it (same risk already flagged in this repo for qwen25_7b as
encoding-time proxy, and for deepseek_r1_14b as agent backbone vs.
ensemble member).

THE FIX: run this calibration ONLY against data/raw/calibration.json --
the 20-document split already carved out disjoint from the main eval
set (originally for src/encode.py's threshold_tau calibration, see
CLAUDE.md). Compare, per field:
  - baseline: run_ghost_agent with the default local-ensemble-only
    stopping rule (frontier_check_fn=None)
  - gated: run_ghost_agent with frontier_check_fn wired to a real
    frontier model via adversary.make_frontier_check_fn -- the local
    ensemble's "defended" verdict is only accepted if that frontier
    model ALSO fails to extract (src/ensemble.py's frontier_check_fn
    gate)

The OUTPUT of this script is evidence for a stopping-rule DECISION
(e.g. "raise consensus_threshold to 4/4", "require a Hamming margin
past first pass", "always try >=2 iterations regardless") -- not a
reportable defense number for the frontier model checked here. Once a
heuristic is chosen from this comparison, apply it with
frontier_check_fn=None again on the real eval/Track A data -- the
frontier model stays a legitimate held-out check there.

Real GPU (local ensemble + local backbone if selected) + real API cost
(the agent backbone's own calls, plus one extra frontier call per
attempt that already passed the local tier) -- run on a small
--n_fields sample first.
"""

import argparse
import csv
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from ghost_agent import run_ghost_agent  # noqa: E402
from ensemble import load_ensemble_config  # noqa: E402
from adversary import make_frontier_check_fn  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iter_calibration_fields(calibration_path: str, n_fields: int):
    """
    Yield (doc_id, domain, field_name, field_value) tuples from
    data/raw/calibration.json, up to n_fields total (not n_fields per
    document) -- this script is meant to be run on a handful of fields
    before scaling up, given the real API/GPU cost per field (baseline
    run + gated run, each a full agent search).
    """
    with open(calibration_path) as f:
        docs = json.load(f)

    count = 0
    for doc in docs:
        if count >= n_fields:
            return
        for field_name, field_value in doc["fields"].items():
            if count >= n_fields:
                return
            yield doc["id"], doc["domain"], field_name, field_value
            count += 1


def run_calibration(
    calibration_path: str = None,
    config_path: str = None,
    output_dir: str = None,
    n_fields: int = 5,
    max_iterations: int = 5,
    frontier_model_key: str = "gpt56_sol",
    agent_backbone_name: str = None,
    run_ghost_agent_fn=run_ghost_agent,
) -> list:
    """
    Args:
        calibration_path: path to data/raw/calibration.json. Defaults to
            the repo-root-anchored location (same convention as
            convergence.py) -- deliberately NEVER data/raw/documents.json
            or any Track A file; this must stay the disjoint split.
        n_fields: total fields to run (baseline + gated each), not per
            document -- controls real cost directly.
        frontier_model_key: key into config.yaml's api_models, e.g.
            "gpt56_sol" (matches the user's own manual UI confirmation)
            or "gemini_31_pro"/"claude_sonnet".
        agent_backbone_name: same meaning as convergence.py's own param
            -- None uses run_ghost_agent's default Anthropic backbone.

    Returns:
        list of per-field comparison dicts (also written to
        results/tables/agent_search_calibration.csv).
    """
    calibration_path = calibration_path or os.path.join(
        _REPO_ROOT, "data", "raw", "calibration.json"
    )
    config_path = config_path or os.path.join(_REPO_ROOT, "config.yaml")
    output_dir = output_dir or os.path.join(_REPO_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble_config = load_ensemble_config(config)
    frontier_check_fn = make_frontier_check_fn(frontier_model_key, config)

    results = []

    for doc_id, domain, field_name, field_value in iter_calibration_fields(
        calibration_path, n_fields,
    ):
        print(f"\n[{doc_id}/{domain}] {field_name} = '{field_value}'")

        print("  -- baseline (local-ensemble-only stopping) --")
        baseline = run_ghost_agent_fn(
            target=field_value,
            field_name=field_name,
            ensemble_members=ensemble_config["members"],
            consensus_threshold=ensemble_config["consensus_threshold"],
            clean_floor_check=ensemble_config["clean_floor_check"],
            max_iterations=max_iterations,
            verbose=False,
            agent_backbone_name=agent_backbone_name,
            backbone_config=config if agent_backbone_name else None,
            use_memory=False,  # calibration must not pollute strategy_memory.json
        )

        print(f"  -- gated (also requires {frontier_model_key} to fail) --")
        gated = run_ghost_agent_fn(
            target=field_value,
            field_name=field_name,
            ensemble_members=ensemble_config["members"],
            consensus_threshold=ensemble_config["consensus_threshold"],
            clean_floor_check=ensemble_config["clean_floor_check"],
            max_iterations=max_iterations,
            verbose=False,
            agent_backbone_name=agent_backbone_name,
            backbone_config=config if agent_backbone_name else None,
            use_memory=False,
            frontier_check_fn=frontier_check_fn,
        )

        row = {
            "doc_id": doc_id,
            "domain": domain,
            "field_name": field_name,
            "field_length": len(field_value),
            "frontier_model": frontier_model_key,
            "baseline_success": baseline["success"],
            "baseline_iterations": baseline["n_iterations"],
            "baseline_hamming": baseline["final_hamming"],
            "gated_success": gated["success"],
            "gated_iterations": gated["n_iterations"],
            "gated_hamming": gated["final_hamming"],
            "iterations_delta": gated["n_iterations"] - baseline["n_iterations"],
            "hamming_delta": gated["final_hamming"] - baseline["final_hamming"],
        }
        results.append(row)
        print(
            f"  baseline: success={baseline['success']} "
            f"iters={baseline['n_iterations']} hamming={baseline['final_hamming']}"
        )
        print(
            f"  gated:    success={gated['success']} "
            f"iters={gated['n_iterations']} hamming={gated['final_hamming']}"
        )

    out_path = os.path.join(output_dir, "agent_search_calibration.csv")
    if results:
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*55}")
    print("SEARCH CALIBRATION SUMMARY")
    print(f"{'='*55}")
    print(f"Fields evaluated: {len(results)}")
    if results:
        n_gated_needed_more = sum(1 for r in results if r["iterations_delta"] > 0)
        n_baseline_fooled_frontier = sum(
            1 for r in results if r["baseline_success"] and not r["gated_success"]
        )
        print(
            f"Fields where the frontier gate forced more iterations: "
            f"{n_gated_needed_more}/{len(results)}"
        )
        print(
            f"Fields where baseline 'succeeded' but the frontier model "
            f"would still have extracted it (never resolved within "
            f"max_iterations under the gate): {n_baseline_fooled_frontier}/"
            f"{len(results)}"
        )
        print(f"Results written to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate GHOST-Agent's search stopping rule against a real "
            "frontier model, on the disjoint data/raw/calibration.json "
            "split only. Never run this against real eval/Track A data."
        )
    )
    parser.add_argument("--n_fields", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--frontier_model", default="gpt56_sol")
    parser.add_argument("--agent_backbone_name", default=None)
    parser.add_argument("--calibration_path", default=None)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_calibration(
        calibration_path=args.calibration_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        n_fields=args.n_fields,
        max_iterations=args.max_iterations,
        frontier_model_key=args.frontier_model,
        agent_backbone_name=args.agent_backbone_name,
    )
