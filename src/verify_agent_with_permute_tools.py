"""
src/verify_agent_with_permute_tools.py (scratch, same status as the
other verify_*.py scripts)

First real (non-mocked) test of GHOST-Agent given the new
bidi_permute/encode_vs_logprob/combine_permute tools (see CLAUDE.md's
"GHOST-Agent given the static pipeline's real tools" section) against a
live Qwen2.5-7B-Instruct proxy on GPU.

Uses the SAME field/sentence as the static bidi-permute and
ghost_permute_sentence checks (src/verify_bidi_permute_track_a.py,
src/verify_ghost_permute_sentence_track_a.py) for a direct comparison:
regfiling_0000 / jurisdiction_code = 'VIC-23', sentence "The company
operates within the jurisdiction of VIC-23, effective as of 15 October
2023." (88 chars).

This is a REAL eval-data field (Track A), so frontier_check_fn is NOT
used during search (that gate is calibration-split-only, see
CLAUDE.md/config.yaml's explicit constraint) -- the agent converges
against the search-tier local ensemble alone, exactly like every other
real per-document GHOST-Agent run this session. gpt56_sol is queried
AFTERWARD as an independent verify-tier check on the converged
encoding, same convention as every other real single-field run
(frontier_verify_every_n_iterations: 0 -- only the final encoding).

Real cost: Qwen2.5-7B proxy resident on GPU (needed for the new tools)
+ the search-tier ensemble's own per-iteration load/unload (4 members)
+ real Claude Haiku API calls (agent backbone) + one real gpt56_sol API
call at the end.
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from encode import load_proxy_model  # noqa: E402
from ghost_agent import run_ghost_agent  # noqa: E402
from ghost_tools import tool_render  # noqa: E402
from ensemble import load_ensemble_config  # noqa: E402
from sentence_utils import find_field_sentence  # noqa: E402
from adversary import query_adversary, check_extraction  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_ID = "regfiling_0000"
FIELD_NAME = "jurisdiction_code"


def main():
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble_config = load_ensemble_config(config)
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
    char_span = tuple(field["char_span"])

    located = find_field_sentence(original_text, field_value, char_span=char_span)
    sentence_text, sent_start, sent_end = located
    print(f"Doc: {DOC_ID}  Field: {FIELD_NAME} = {field_value!r}")
    print(f"Sentence ({len(sentence_text)} chars): {sentence_text!r}\n")

    print("Loading Qwen2.5-7B-Instruct proxy (GPU)...")
    proxy = load_proxy_model(config)

    try:
        result = run_ghost_agent(
            target=sentence_text,
            field_name=FIELD_NAME,
            field_ground_truth=field_value,
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
        )
    finally:
        import torch
        del proxy
        torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"Agent result: success={result['success']} "
          f"iterations={result['n_iterations']} "
          f"hamming={result['final_hamming']}/{len(sentence_text)}")

    encoded_sentence = result["best_encoding"]
    if encoded_sentence is None:
        print("Agent never produced a usable encoding -- nothing to verify further.")
        return

    if tool_render(encoded_sentence) != sentence_text:
        print("WARNING: agent's best encoding does not render back to the "
              "original sentence -- skipping full-document splice/verify.")
        encoded_document_text = None
    else:
        encoded_document_text = (
            original_text[:sent_start] + encoded_sentence + original_text[sent_end:]
        )
        if tool_render(encoded_document_text) != original_text:
            print("WARNING: full-document round-trip failed after splicing.")
            encoded_document_text = None
        else:
            print("Full-document round-trip: OK")

    frontier_result = None
    if encoded_document_text is not None:
        print(f"\nQuerying {frontier_spec['model_id']} ({frontier_spec['provider']}) "
              f"on the converged encoding (verify tier, post-hoc)...")
        response = query_adversary(
            encoded_text=encoded_document_text,
            field_name=FIELD_NAME,
            model_id=frontier_spec["model_id"],
            provider=frontier_spec["provider"],
            max_tokens=frontier_spec.get("max_tokens", 100),
        )
        check = check_extraction(response, field_value)
        frontier_result = {
            "raw_response": response,
            "extracted": check["extracted"],
            "refusal": check["refusal"],
            "defended": not check["extracted"],
        }
        print(f"Raw response: {response!r}")
        print(f"Extracted correctly: {check['extracted']}")
        print(f"Defended: {not check['extracted']}")

    out = {
        "doc_id": DOC_ID,
        "field_name": FIELD_NAME,
        "field_value": field_value,
        "sentence": sentence_text,
        "agent_backbone": "claude_haiku",
        "search_tier_success": result["success"],
        "n_iterations": result["n_iterations"],
        "final_hamming": result["final_hamming"],
        "sentence_length": len(sentence_text),
        "best_encoding": encoded_sentence,
        "full_document_round_trip_ok": encoded_document_text is not None,
        "frontier_model": "gpt56_sol",
        "frontier_result": frontier_result,
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
    }
    out_path = os.path.join(
        _REPO_ROOT, "results", "raw",
        "verify_agent_with_permute_tools_jurisdiction_code.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
