"""
src/verify_prior_strength_learning.py (scratch, real Track A eval data --
no frontier_check_fn used anywhere here, per the calibration-only
firewall; this is a real per-document search run, same status as
every other real GHOST-Agent run this session)

First real (non-mocked) exercise of src/prior_strength.py's whole chain
(prior_strength.py -> ensemble.run_ensemble_prior_strength ->
ghost_agent.py's prior_strength_fn wiring -> strategy_memory.py's
prior_strength_ratio field) -- everything up to now was verified against
fakes only.

SENTENCE SCOPE, deliberately -- field-level scope was explicitly
rejected as insufficient for this question (a bare field value rarely
carries much of the "is this recognizable/familiar" signal the whole
point of prior-strength scoring is about; a sentence like "INNOVATECH
SOLUTIONS PTY LTD (ABN: ...) has been registered..." has real semantic
content a field value alone doesn't).

use_memory=True (real strategy_memory.json / experience_log.jsonl
writes, intentionally -- the whole question is whether real persistent
memory + this new profiling signal show ANY learning signal across
documents, not just that the plumbing works). No frontier_check_fn:
this is real Track A data, not the disjoint calibration split.

10 examples across all 4 Track A domains (business filing, legal,
academic, technical), chosen to vary in how much recognizable semantic
content the surrounding sentence carries -- company names, legal
boilerplate, personal-name-adjacent identifiers, vs. purely opaque
codes -- specifically to get real variance in prior_strength_ratio to
check against.
"""

import json
import os
import sys
from functools import partial

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from encode import load_proxy_model  # noqa: E402
from ghost_agent import run_ghost_agent  # noqa: E402
from ensemble import load_ensemble_config, run_ensemble_prior_strength  # noqa: E402
from sentence_utils import find_field_sentence  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = [
    ("regfiling_0000", "jurisdiction_code"),   # short opaque code
    ("regfiling_0000", "ABN"),                 # sentence carries a real company name
    ("regfiling_0001", "company_type"),        # very short, generic ("PTE LTD")
    ("regfiling_0002", "jurisdiction_code"),   # same field type, different doc
                                                # (swapped from regfiling_0003
                                                # 2026-09-09: that field's
                                                # char_span_status is
                                                # "unresolved" -- a genuine
                                                # paraphrase, ground truth
                                                # "QLD-10" never appears
                                                # verbatim in the text -- so
                                                # find_field_sentence now
                                                # correctly returns None for
                                                # it post-fix, instead of the
                                                # wrong sentence it silently
                                                # returned before. Not
                                                # locatable, so not a valid
                                                # candidate for this script.
    ("legalrec_0000", "case_filing_number"),   # legal boilerplate sentence
    ("legalrec_0002", "statute_reference"),    # longest field, formulaic legal phrasing
    ("acadid_0000", "orcid_fragment"),         # sentence names a real person
    ("acadid_0001", "doi_suffix"),             # academic identifier, less personal
    ("techdoc_0002", "ip_address"),            # purely technical/opaque
    ("techdoc_0001", "serial_number"),         # purely technical/opaque
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

    print("Loading Qwen2.5-7B-Instruct proxy (GPU, once for the whole run)...")
    proxy = load_proxy_model(config)

    results = []
    try:
        for i, (doc_id, field_name) in enumerate(CANDIDATES):
            field = fields_by_key[(doc_id, field_name)]
            original_text = documents[doc_id]["carrier_text"]
            field_value = field["ground_truth"]
            char_span = tuple(field["char_span"])

            sentence_text, s_start, s_end = find_field_sentence(
                original_text, field_value, char_span=char_span
            )

            print(f"\n{'='*70}")
            print(f"[{i+1}/10] {doc_id}/{field_name} = {field_value!r}")
            print(f"Sentence ({len(sentence_text)} chars): {sentence_text!r}")

            captured = {}

            def capturing_prior_strength_fn(target, context, _captured=captured):
                res = run_ensemble_prior_strength(
                    target, context, members=ensemble_config["members"],
                )
                _captured["result"] = res
                return res

            result = run_ghost_agent(
                target=sentence_text,           # SENTENCE scope
                field_name=field_name,
                field_ground_truth=field_value,  # grade against the field, not the sentence
                ensemble_members=ensemble_config["members"],
                consensus_threshold=ensemble_config["consensus_threshold"],
                clean_floor_check=ensemble_config["clean_floor_check"],
                max_iterations=5,
                tool_budget_per_iter=20,
                agent_response_max_tokens=8000,
                verbose=True,
                agent_backbone_name="claude_haiku",
                backbone_config=config,
                use_memory=True,   # real persistent memory, intentionally
                doc_index=i,
                proxy=proxy,
                prior_strength_fn=capturing_prior_strength_fn,
            )

            prior_strength_result = captured.get("result")
            mean_ratio = prior_strength_result["mean_ratio"] if prior_strength_result else None
            final_target = result["final_target"]
            hamming_pct = (
                result["final_hamming"] / len(final_target) if final_target else None
            )

            hamming_pct_str = f"{hamming_pct:.2%}" if hamming_pct is not None else "N/A"
            print(
                f"\n>>> success={result['success']} iterations={result['n_iterations']} "
                f"hamming={result['final_hamming']}/{len(final_target)} ({hamming_pct_str}) "
                f"prior_strength_ratio={mean_ratio} "
                f"n_principles_available={result['n_principles_available']}"
            )

            results.append({
                "doc_id": doc_id,
                "field_name": field_name,
                "field_value": field_value,
                "sentence": sentence_text,
                "sentence_len": len(sentence_text),
                "prior_strength_ratio": mean_ratio,
                "prior_strength_detail": prior_strength_result,
                "success": result["success"],
                "n_iterations": result["n_iterations"],
                "final_hamming": result["final_hamming"],
                "hamming_pct": hamming_pct,
                "n_principles_available": result["n_principles_available"],
                "best_encoding": result["best_encoding"],
            })
    finally:
        import torch
        del proxy
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(
        f"{'doc/field':40s} {'prior_ratio':>11s} {'iters':>6s} "
        f"{'hamming%':>9s} {'n_princ':>8s} {'success':>8s}"
    )
    for r in results:
        ratio_str = f"{r['prior_strength_ratio']:.4f}" if r["prior_strength_ratio"] is not None else "N/A"
        hpct_str = f"{r['hamming_pct']:.1%}" if r["hamming_pct"] is not None else "N/A"
        print(
            f"{r['doc_id']+'/'+r['field_name']:40s} {ratio_str:>11s} "
            f"{r['n_iterations']:>6d} {hpct_str:>9s} "
            f"{r['n_principles_available']:>8d} {str(r['success']):>8s}"
        )

    out_path = os.path.join(
        _REPO_ROOT, "results", "raw", "verify_prior_strength_learning.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
