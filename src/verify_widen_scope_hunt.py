"""
src/verify_widen_scope_hunt.py (scratch, real eval data -- Track A --
so frontier_check_fn is NOT used, per the calibration-only firewall in
CLAUDE.md's "Agentic-ness critique" section)

Purpose: both real agentic-scope runs so far (verify_agentic_scope_track_a.py,
verify_agentic_scope_calibration.py) found success via seed/payload retries
alone -- widen_scope() has NEVER actually fired on a real run. Both prior
targets were short (jurisdiction_code, 6 chars; reference_number, 12 chars),
where combine_permute's search either already hits the true Hamming maximum
immediately, or a stronger VS payload alone closes the gap.

This sweeps several LONGER Track A fields (bigger permutation search space,
so seed=0 at a fixed trial budget is less likely to land on-target) across
different documents/domains, starting the agent at FIELD scope with
wider_context=<field's sentence> and require_unanimous=True, hunting for
the first real case where:
  (a) the agent needs >1 iteration to reach full local-ensemble unanimity, or
  (b) it never reaches unanimity within max_iterations and/or calls
      widen_scope() itself.

Proxy (Qwen2.5-7B) loaded ONCE and reused across all fields in the sweep,
mirroring convergence.py's --use_proxy load-once pattern -- not reloaded
per field.
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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diverse in domain + longer than the two fields already tested this
# session (jurisdiction_code=6 chars, reference_number=12 chars).
CANDIDATES = [
    ("legalrec_0002", "statute_reference"),   # avg 36.9 chars, longest field
    ("acadid_0001", "orcid_fragment"),        # avg 19.0 chars, fixed dash format
    ("legalrec_0003", "case_filing_number"),  # avg 14.1 chars
    ("acadid_0002", "doi_suffix"),            # avg 16.3 chars
    ("regfiling_0000", "ABN"),                 # avg 14.4 chars, digit-grouped
]


def main():
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble_config = load_ensemble_config(config)

    documents = {}
    with open(os.path.join(_REPO_ROOT, "data", "track_a", "full", "documents.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            documents[d["doc_id"]] = d

    fields_by_key = {}
    with open(os.path.join(_REPO_ROOT, "data", "track_a", "full", "fields.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            fields_by_key[(d["doc_id"], d["field_name"])] = d

    print("Loading Qwen2.5-7B-Instruct proxy (GPU, once for the whole sweep)...")
    proxy = load_proxy_model(config)

    results = []
    try:
        for doc_id, field_name in CANDIDATES:
            field = fields_by_key.get((doc_id, field_name))
            if field is None:
                print(f"SKIP: {doc_id}/{field_name} not found in fields.jsonl")
                continue

            original_text = documents[doc_id]["carrier_text"]
            field_value = field["ground_truth"]
            char_span = tuple(field["char_span"])

            try:
                sentence_text, s_start, s_end = find_field_sentence(
                    original_text, field_value, char_span=char_span
                )
            except Exception as e:
                print(f"SKIP: {doc_id}/{field_name} sentence lookup failed: {e}")
                continue

            print(f"\n{'='*70}")
            print(f"Doc: {doc_id}  Field: {field_name} = {field_value!r} "
                  f"({len(field_value)} chars)")
            print(f"Sentence ({len(sentence_text)} chars): {sentence_text!r}")

            result = run_ghost_agent(
                target=field_value,          # FIELD scope at the start
                field_name=field_name,
                ensemble_members=ensemble_config["members"],
                consensus_threshold=ensemble_config["consensus_threshold"],
                clean_floor_check=ensemble_config["clean_floor_check"],
                max_iterations=5,
                tool_budget_per_iter=20,
                agent_response_max_tokens=8000,
                verbose=True,
                agent_backbone_name="claude_haiku",
                backbone_config=config,
                use_memory=False,
                proxy=proxy,
                wider_context=sentence_text,  # agent MAY request this
                require_unanimous=True,
                # No frontier_check_fn: this is real Track A eval data.
            )

            print(
                f"\n>>> {doc_id}/{field_name}: success={result['success']} "
                f"iterations={result['n_iterations']} "
                f"scope_widened={result['scope_widened']} "
                f"hamming={result['final_hamming']}/{len(result['final_target'])}"
            )

            results.append({
                "doc_id": doc_id,
                "field_name": field_name,
                "field_value": field_value,
                "field_len": len(field_value),
                "sentence": sentence_text,
                "sentence_len": len(sentence_text),
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
                    }
                    for a in result["history"]
                ],
            })
    finally:
        import torch
        del proxy
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\nSWEEP SUMMARY\n{'='*70}")
    for r in results:
        print(
            f"{r['doc_id']}/{r['field_name']:22s} "
            f"iters={r['n_iterations']} widened={r['scope_widened']} "
            f"hamming={r['final_hamming']}/{len(r['final_target'])} "
            f"success={r['success']}"
        )

    out_path = os.path.join(
        _REPO_ROOT, "results", "raw", "verify_widen_scope_hunt.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
