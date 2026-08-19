"""
src/verify_agentic_scope_track_a.py (scratch, same status as the other
verify_*.py scripts)

First real (non-mocked) test of the two "make the agent actually
self-direct" fixes (CLAUDE.md's request: "if the agent can't
autonomously improve itself, it isn't ready"):
  1. require_unanimous=True -- stop only at FULL ensemble consensus, not
     the previous binary consensus_threshold bar that let the agent stop
     on iteration 1 essentially every real run tested so far.
  2. wider_context / widen_scope() -- the agent starts at FIELD scope
     (just the bare value, not the sentence) and can request the
     containing sentence itself mid-run if the field alone isn't
     defensible, instead of scope being fixed externally for the whole
     run.

Also loads the live Qwen2.5-7B proxy so combine_permute/bidi_permute
(with seed retry) are available -- this is also the first real test of
whether the agent actually retries with a different seed when Hamming
is disappointing, per CLAUDE.md's "Seed fix" section ("Not yet done: ...
re-test").

SAME field as every other real Track A check this session, for direct
comparison against prior results:
  regfiling_0000 / jurisdiction_code = 'VIC-23'
  sentence (88 chars): "The company operates within the jurisdiction of
  VIC-23, effective as of 15 October 2023."

Prior results on this exact field, for reference:
  - static bidi-only, field scope, max Hamming (6/6): gpt56_sol EXTRACTED
  - static ghost_permute_sentence (bidi search + real logprob VS,
    sentence scope): Hamming 86/88, gpt56_sol DEFENDED (empty response)
  - agent + combine_permute, sentence scope handed in externally,
    require_unanimous NOT used (old binary stopping rule), seed fix
    applied: Hamming 81/88, search tier 4/4, gpt56_sol EXTRACTED anyway

This run starts the agent at FIELD scope (not sentence) and see whether
it autonomously decides to retry seeds / widen to the sentence under the
stricter require_unanimous bar -- a genuinely different experiment from
all three above, none of which gave the agent a choice about scope.

This is a REAL eval-data field (Track A), so frontier_check_fn is NOT
used during search (calibration-split-only constraint, see CLAUDE.md/
config.yaml) -- frontier models are only queried post-hoc on the
converged encoding.
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
VERIFY_MODELS = ["claude_sonnet", "gpt56_sol", "gemini_31_pro"]


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
            use_memory=False,
            proxy=proxy,
            wider_context=sentence_text,      # agent MAY request this
            require_unanimous=True,           # ...and has a real reason to
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

    encoding = result["best_encoding"]
    if encoding is None:
        print("Agent never produced a usable encoding -- nothing to verify further.")
        return

    final_target = result["final_target"]
    if tool_render(encoding) != final_target:
        print("WARNING: agent's best encoding does not render back to its "
              "own final target -- skipping full-document splice/verify.")
        encoded_document_text = None
    elif result["scope_widened"]:
        encoded_document_text = (
            original_text[:sent_start] + encoding + original_text[sent_end:]
        )
    else:
        f_start, f_end = char_span
        encoded_document_text = (
            original_text[:f_start] + encoding + original_text[f_end:]
        )

    if encoded_document_text is not None and tool_render(encoded_document_text) != original_text:
        print("WARNING: full-document round-trip failed after splicing.")
        encoded_document_text = None
    elif encoded_document_text is not None:
        print("Full-document round-trip: OK")

    verify_results = {}
    if encoded_document_text is not None:
        for name in VERIFY_MODELS:
            model_cfg = config["api_models"][name]
            print(f"\nQuerying {model_cfg['model_id']} ({model_cfg['provider']}) "
                  f"on the converged encoding (verify tier, post-hoc)...")
            response = query_adversary(
                encoded_text=encoded_document_text,
                field_name=FIELD_NAME,
                model_id=model_cfg["model_id"],
                provider=model_cfg["provider"],
                max_tokens=model_cfg.get("max_tokens", 100),
            )
            check = check_extraction(response, field_value)
            verify_results[name] = {
                "raw_response": response,
                "extracted": check["extracted"],
                "refusal": check["refusal"],
                "defended": not check["extracted"],
            }
            print(f"[{name}] response={response!r}")
            print(f"[{name}] defended={not check['extracted']}")

    out = {
        "doc_id": DOC_ID,
        "field_name": FIELD_NAME,
        "field_value": field_value,
        "sentence": sentence_text,
        "agent_backbone": "claude_haiku",
        "require_unanimous": True,
        "search_tier_success": result["success"],
        "n_iterations": result["n_iterations"],
        "scope_widened": result["scope_widened"],
        "final_target": final_target,
        "final_hamming": result["final_hamming"],
        "best_encoding": encoding,
        "full_document_round_trip_ok": encoded_document_text is not None,
        "verify_results": verify_results,
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
        "verify_agentic_scope_track_a_jurisdiction_code.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
