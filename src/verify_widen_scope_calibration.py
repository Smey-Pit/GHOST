"""
src/verify_widen_scope_calibration.py (scratch, calibration-split ONLY --
same firewall rule as verify_agentic_scope_calibration.py/
calibrate_agent_search.py: frontier_check_fn must never touch real
eval/Track A data).

Purpose: actually trigger widen_scope() for real. Neither real run so
far (Track A jurisdiction_code, calibration reference_number) needed
it -- both found a win within field scope via seed/payload retries
alone. This run deliberately targets a field ALREADY documented as hard
even for the weak local search-tier ensemble (CLAUDE.md's self-improving
audit: "dosage failed in BOTH batches (cold and warm, identically)" in
convergence.py's own batch pilot, no frontier gate needed to see it
struggle) -- a genuinely difficult short field, not a strawman, on the
theory that seed/payload retries alone are less likely to close the gap
here, giving widen_scope() an actual opening.

Field: cal_med_001 / dosage = '416ml' (5 chars). Sentence: "The dosage
is set at 416ml." (28 chars) -- located via sentence_utils, no char_span
needed (calibration.json's flat text+fields shape).

max_iterations raised to 8 (vs. the usual 5) to give the agent real room
to exhaust seed/payload retries before widen_scope becomes the only
remaining lever, per the system prompt's own instructed order
(ghost_agent_prompt.py: retry a seed/payload FIRST, widen only after).
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

DOC_ID = "cal_med_001"
FIELD_NAME = "dosage"
FRONTIER_MODEL_KEY = "gpt56_sol"
MAX_ITERATIONS = 8


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
        f"only). max_iterations={MAX_ITERATIONS} (raised from the usual "
        f"5 to give room for seed/payload retries before widen_scope).\n"
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
            max_iterations=MAX_ITERATIONS,
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
        "max_iterations": MAX_ITERATIONS,
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
        "verify_widen_scope_calibration_dosage.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
