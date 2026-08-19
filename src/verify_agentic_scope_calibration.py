"""
src/verify_agentic_scope_calibration.py (scratch, calibration-split ONLY
-- never run this against real eval/Track A data, see
calibrate_agent_search.py's docstring for why)

Purpose: actually WATCH require_unanimous / widen_scope fire on a real
run, instead of just confirming the plumbing via a mocked smoke test.
The first real attempt (src/verify_agentic_scope_track_a.py) didn't
exercise either mechanism -- a 6-character field's permutation space was
small enough that combine_permute's default seed=0 already hit the true
Hamming maximum AND full local-ensemble unanimity on iteration 1, so
there was nothing to retry or escalate past.

This run uses the SAME lever calibrate_agent_search.py already proved
forces extra iterations: a real frontier_check_fn (gpt56_sol) gating the
local ensemble's "defended" verdict, calibration-split ONLY (never real
eval data -- this optimizes search directly against gpt56_sol, which is
exactly why it's firewalled to the disjoint data/raw/calibration.json
split, per CLAUDE.md's "Agentic-ness critique" section).

Field: cal_fin_001 / reference_number = 'REF-7W8R4NEA' -- already shown
(calibrate_agent_search.py's real run) to need 3 iterations under this
exact gate at the OLD binary stopping rule, with Hamming barely moving
(10 -> 11) across those extra attempts. That "barely moving" result is
exactly what require_unanimous's richer signal + widen_scope's escape
hatch are meant to fix -- this run adds both on top of the same gate to
see whether the agent does something different with them available.

wider_context = the field's own sentence ("Reference: REF-7W8R4NEA."),
located via sentence_utils.find_field_sentence on the calibration doc's
flat text (no char_span needed -- same shape as data/raw/documents.json).
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from encode import load_proxy_model  # noqa: E402
from ghost_agent import run_ghost_agent  # noqa: E402
from ensemble import load_ensemble_config  # noqa: E402
from sentence_utils import find_field_sentence  # noqa: E402
from adversary import make_frontier_check_fn  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_ID = "cal_fin_001"
FIELD_NAME = "reference_number"
FRONTIER_MODEL_KEY = "gpt56_sol"


def main():
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble_config = load_ensemble_config(config)
    frontier_check_fn = make_frontier_check_fn(FRONTIER_MODEL_KEY, config)

    with open(os.path.join(_REPO_ROOT, "data", "raw", "calibration.json")) as f:
        calibration_docs = json.load(f)
    doc = next(d for d in calibration_docs if d["id"] == DOC_ID)
    field_value = doc["fields"][FIELD_NAME]

    located = find_field_sentence(doc["text"], field_value)
    sentence_text, s_start, s_end = located
    print(f"Doc: {DOC_ID}  Field: {FIELD_NAME} = {field_value!r}")
    print(f"Sentence: {sentence_text!r}\n")
    print(
        f"Gate: local ensemble 'defended' verdicts are only accepted if "
        f"{FRONTIER_MODEL_KEY} ALSO fails to extract (calibration-split "
        f"only, per adversary.make_frontier_check_fn's docstring).\n"
    )

    print("Loading Qwen2.5-7B-Instruct proxy (GPU)...")
    proxy = load_proxy_model(config)

    try:
        result = run_ghost_agent(
            target=field_value,               # FIELD scope at the start
            field_name=FIELD_NAME,
            ensemble_members=ensemble_config["members"],
            consensus_threshold=ensemble_config["consensus_threshold"],
            clean_floor_check=ensemble_config["clean_floor_check"],
            max_iterations=5,
            tool_budget_per_iter=20,
            agent_response_max_tokens=8000,
            verbose=True,
            agent_backbone_name="claude_haiku",
            backbone_config=config,
            use_memory=False,   # calibration must not pollute strategy_memory.json
            proxy=proxy,
            wider_context=sentence_text,
            require_unanimous=True,
            frontier_check_fn=frontier_check_fn,
        )
    finally:
        import torch
        del proxy
        torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"Agent result: success={result['success']} "
          f"iterations={result['n_iterations']} "
          f"scope_widened={result['scope_widened']} "
          f"hamming={result['final_hamming']}/{len(result['final_target'])}")

    for a in result["history"]:
        print(
            f"  iter {a.iteration}: bidi={a.bidi_config} vs={a.vs_payload} "
            f"hamming={a.hamming_dist} n_failed={a.n_failed}/{a.n_valid} "
            f"local_defended={a.extraction_defeated} "
            f"frontier_gate={a.ensemble_result.get('frontier_check')}"
        )

    out = {
        "doc_id": DOC_ID,
        "field_name": FIELD_NAME,
        "field_value": field_value,
        "sentence": sentence_text,
        "agent_backbone": "claude_haiku",
        "require_unanimous": True,
        "frontier_gate_model": FRONTIER_MODEL_KEY,
        "success": result["success"],
        "n_iterations": result["n_iterations"],
        "scope_widened": result["scope_widened"],
        "final_target": result["final_target"],
        "final_hamming": result["final_hamming"],
        "best_encoding": result["best_encoding"],
        "history": [
            {
                "iteration": a.iteration,
                "bidi_config": a.bidi_config,
                "vs_payload": a.vs_payload,
                "hamming_dist": a.hamming_dist,
                "ensemble_defended": a.extraction_defeated,
                "n_failed": a.n_failed,
                "n_valid": a.n_valid,
                "frontier_check": a.ensemble_result.get("frontier_check"),
                "agent_reasoning": a.agent_reasoning,
            }
            for a in result["history"]
        ],
    }
    out_path = os.path.join(
        _REPO_ROOT, "results", "raw",
        "verify_agentic_scope_calibration_reference_number.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
