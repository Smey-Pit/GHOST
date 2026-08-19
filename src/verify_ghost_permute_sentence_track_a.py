"""
src/verify_ghost_permute_sentence_track_a.py (scratch, same status as
verify_bidi_permute_track_a.py etc.)

Direct follow-up to verify_bidi_permute_track_a.py: that script permuted
ONLY the field value ('VIC-23') and found gpt56_sol still extracted it
at max Hamming distance. This script tests the combination the user
asked for next: sentence-scope permutation (the whole sentence
containing the field, not just the field's own characters -- removes
the contextual scaffolding an adversary could otherwise lean on) PLUS
real logprob-guided VS injection on top (the production ghost/
ghost_permute mechanism, via the real Qwen2.5-7B-Instruct proxy on GPU --
not the fixed-payload approximation GHOST-Agent's tools use).

This exercises the NEW encode_ghost_permute_sentence (src/encode.py),
added this session specifically for this test -- the "ghost_permute_
sentence" condition CLAUDE.md already flagged as explicitly deferred.
Not yet wired into config.yaml's encoding_conditions or run at corpus
scale; this is a single real end-to-end check.

Real GPU load (Qwen2.5-7B-Instruct proxy) + one real gpt56_sol API call.
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from encode import load_proxy_model, encode_ghost_permute_sentence  # noqa: E402
from ghost_tools import tool_render, tool_get_stored, tool_hamming  # noqa: E402
from sentence_utils import find_field_sentence  # noqa: E402
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
    seed = config["seed"]
    salt = config["disruption_payload"]
    threshold_tau = config["threshold_tau"]

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
    char_span = tuple(field["char_span"])

    located = find_field_sentence(original_text, field_value, char_span=char_span)
    if located is None:
        raise RuntimeError(f"could not locate sentence for {FIELD_NAME!r} in {DOC_ID}")
    sentence_text, sent_start, sent_end = located
    print(f"Doc: {DOC_ID}  Field: {FIELD_NAME} = {field_value!r}")
    print(f"Sentence ({len(sentence_text)} chars): {sentence_text!r}\n")

    item = {
        "id": DOC_ID,
        "text": original_text,
        "fields": {FIELD_NAME: field_value},
    }

    print("Loading Qwen2.5-7B-Instruct proxy (GPU)...")
    proxy = load_proxy_model(config)

    stats = []
    try:
        encoded_item = encode_ghost_permute_sentence(
            item, proxy, seed, salt, threshold_tau, stats,
            trials=TRIALS, char_spans={FIELD_NAME: char_span},
        )
    finally:
        import torch
        del proxy
        torch.cuda.empty_cache()

    encoded_text = encoded_item["text"]

    # Recover the encoded sentence itself (for reporting) by diffing
    # against where the original sentence used to be -- the surrounding
    # text is untouched, so the encoded sentence is whatever now sits
    # where the original occupied a prefix/suffix-identical span.
    prefix = original_text[:sent_start]
    suffix = original_text[sent_end:]
    assert encoded_text.startswith(prefix), "text before the sentence must be untouched"
    assert encoded_text.endswith(suffix), "text after the sentence must be untouched"
    encoded_sentence = encoded_text[len(prefix):len(encoded_text) - len(suffix)]

    rendered_sentence = tool_render(encoded_sentence)
    stored_sentence = tool_get_stored(encoded_sentence)
    dist = tool_hamming(stored_sentence, sentence_text) if len(stored_sentence) == len(sentence_text) else -1

    print(f"VS injection iterations per character: min={min(stats)} "
          f"max={max(stats)} mean={sum(stats)/len(stats):.1f} (n={len(stats)})")
    print(f"Stored (tokenized) sentence: {stored_sentence!r}")
    print(f"Hamming distance (stored vs. original sentence): {dist}/{len(sentence_text)}")
    assert rendered_sentence == sentence_text, "isolated-sentence round-trip check failed"

    if tool_render(encoded_text) != original_text:
        raise RuntimeError(
            "FULL-DOCUMENT round-trip failed after splicing the encoded "
            "sentence back in -- must never ship; treat as a hard stop."
        )
    print("Full-document round-trip: OK (renders byte-identical to original)")

    print(f"\nQuerying {frontier_spec['model_id']} ({frontier_spec['provider']})...")
    response = query_adversary(
        encoded_text=encoded_text,
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
        "sentence": sentence_text,
        "sentence_length": len(sentence_text),
        "trials": TRIALS,
        "threshold_tau": threshold_tau,
        "hamming_distance": dist,
        "vs_iterations_min": min(stats),
        "vs_iterations_max": max(stats),
        "vs_iterations_mean": sum(stats) / len(stats),
        "stored_sentence": stored_sentence,
        "full_document_round_trip_ok": True,
        "frontier_model": "gpt56_sol",
        "frontier_response": result["response"],
        "extracted": result["extracted"],
        "refusal": result["refusal"],
        "defended": not result["extracted"],
    }
    out_path = os.path.join(
        _REPO_ROOT, "results", "raw",
        "verify_ghost_permute_sentence_track_a_jurisdiction_code.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
