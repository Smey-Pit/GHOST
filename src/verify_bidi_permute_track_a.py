"""
src/verify_bidi_permute_track_a.py (scratch, same status as
verify_gemini_run.py etc. -- not wired into the SLURM pipeline)

Static-GHOST comparison point requested directly: run bidi_permute's
random separable-permutation search (2000 trials, keeping the best
round-trip-valid Hamming distance -- see CLAUDE.md's correction on what
this search actually guarantees, i.e. best-of-sampled, not a proven
global maximum) on ONE real Track A field, splice the encoded field back
into its real document, verify the FULL document still renders
byte-identical to the original (not just the isolated field -- bidi
context can extend past the local span, this is why sentence_utils.py's
smoke test does the same full-document check), then query gpt56_sol
(real API call) on the real extraction task.

This is a single N=1 data point for the static-permutation mechanism,
directly comparable to the GHOST-Agent N=2 calibration numbers already
gathered this session (results/tables/agent_search_calibration.csv) --
not a claim about which approach is better at scale.
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from bidi_permute import encode_field, visible  # noqa: E402
from adversary import query_adversary, check_extraction  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_ID = "regfiling_0000"
FIELD_NAME = "jurisdiction_code"
TRIALS = 2000


def main():
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    frontier_spec = config["api_models"]["gpt56_sol"]

    documents = {}
    with open(os.path.join(_REPO_ROOT, "data", "track_a", "full", "documents.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            documents[d["doc_id"]] = d

    field = None
    with open(os.path.join(_REPO_ROOT, "data", "track_a", "full", "fields.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] == DOC_ID and d["field_name"] == FIELD_NAME:
                field = d
                break
    if field is None:
        raise RuntimeError(f"field {FIELD_NAME} not found for {DOC_ID}")

    original_text = documents[DOC_ID]["carrier_text"]
    field_value = field["ground_truth"]
    start, end = field["char_span"]
    assert original_text[start:end] == field_value, (
        "char_span in fields.jsonl no longer matches documents.jsonl -- "
        "re-check after the 2026-08-16 post-hoc patch"
    )

    print(f"Doc: {DOC_ID}  Field: {FIELD_NAME} = {field_value!r}  "
          f"(span {start}:{end})")

    enc, stored, dist = encode_field(field_value, trials=TRIALS, seed=0)
    if enc is None:
        raise RuntimeError(
            f"find_target_permutation exhausted {TRIALS} trials with no "
            f"round-trip-valid candidate for {field_value!r}"
        )

    print(f"Best Hamming distance found: {dist}/{len(field_value)}")
    print(f"Stored (tokenized) sequence: {stored!r}")
    assert visible(enc) == field_value, "isolated-field round-trip check failed"

    encoded_document_text = original_text[:start] + enc + original_text[end:]
    if visible(encoded_document_text) != original_text:
        raise RuntimeError(
            "FULL-DOCUMENT round-trip failed after splicing the encoded "
            "field back in -- bidi context outside the field span changed "
            "the rendering. This must never ship; treat as a hard stop, "
            "not a warning."
        )
    print("Full-document round-trip: OK (renders byte-identical to original)")

    print(f"\nQuerying {frontier_spec['model_id']} ({frontier_spec['provider']})...")
    response = query_adversary(
        encoded_text=encoded_document_text,
        field_name=FIELD_NAME,
        model_id=frontier_spec["model_id"],
        provider=frontier_spec["provider"],
        max_tokens=frontier_spec.get("max_tokens", 100),
    )
    result = check_extraction(response, field_value)

    print(f"Raw response: {response!r}")
    print(f"Extracted correctly: {result['extracted']}")
    print(f"Refusal: {result['refusal']}")
    print(f"Defended: {not result['extracted']}")

    out = {
        "doc_id": DOC_ID,
        "field_name": FIELD_NAME,
        "field_value": field_value,
        "trials": TRIALS,
        "hamming_distance": dist,
        "field_length": len(field_value),
        "stored_sequence": stored,
        "full_document_round_trip_ok": True,
        "frontier_model": "gpt56_sol",
        "frontier_response": result["response"],
        "extracted": result["extracted"],
        "refusal": result["refusal"],
        "defended": not result["extracted"],
    }
    out_path = os.path.join(
        _REPO_ROOT, "results", "raw", "verify_bidi_permute_track_a_jurisdiction_code.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
