"""
src/verify_prior_strength_frontier.py (scratch, real API calls)

Post-hoc frontier verification for the 10 real sentence-scope examples
in results/raw/verify_prior_strength_learning.json. That run's own
"success" field reflects ONLY the local 4-model search-tier ensemble --
this script is what actually answers "does this transfer to real
frontier models," same distinction this whole session has repeatedly
had to make explicit (the search/verify gap).

Queries the standard frontier_verify_models trio from config.yaml
(claude_sonnet, gpt56_sol, gemini_31_pro) -- deliberately NOT
claude_haiku, which was this run's own agent backbone, avoiding the
same backbone-vs-verify-tier circularity already flagged elsewhere in
this project.

For each of the 10 items:
  1. Skip (with a recorded reason) if best_encoding is None, or if it
     doesn't render back to the exact sentence it's supposed to be
     (catches the real bug found live during the source run --
     regfiling_0001/company_type produced a length-mismatched partial
     encoding that should never have been accepted as "success" in the
     first place; querying frontier models on that would be
     meaningless, not just uninformative).
  2. Splice the encoding back into its real Track A document at the
     sentence's real span, and verify the FULL document still round-trips
     to the original before querying anything -- the hard rendering
     constraint, checked for real, not assumed.
  3. Query all 3 frontier models against the spliced document with the
     real field-extraction prompt.

No GPU needed (pure API calls + the bidi render oracle) -- safe to run
even while the H100 is occupied by something else.
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from ghost_tools import tool_render  # noqa: E402
from sentence_utils import find_field_sentence  # noqa: E402
from adversary import query_adversary, check_extraction  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_PATH = os.path.join(
    _REPO_ROOT, "results", "raw", "verify_prior_strength_learning.json"
)
OUT_PATH = os.path.join(
    _REPO_ROOT, "results", "raw", "verify_prior_strength_frontier.json"
)


def main():
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    verify_model_keys = config["agent_ensemble"]["frontier_verify_models"]

    with open(SOURCE_PATH) as f:
        items = json.load(f)

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

    out = []
    for i, item in enumerate(items):
        doc_id = item["doc_id"]
        field_name = item["field_name"]
        field_value = item["field_value"]
        sentence = item["sentence"]
        encoding = item["best_encoding"]

        print(f"\n{'='*70}\n[{i+1}/{len(items)}] {doc_id}/{field_name} = {field_value!r}")

        if encoding is None:
            print("SKIPPED: no best_encoding produced by the source run")
            out.append({**item, "frontier_skipped_reason": "no best_encoding", "verify_results": {}})
            continue

        rendered = tool_render(encoding)
        if rendered != sentence:
            reason = (
                f"encoding does not round-trip to its target sentence "
                f"(rendered {len(rendered)} chars vs. sentence {len(sentence)} chars) "
                f"-- almost certainly the same length-mismatch bug found live during "
                f"the source run; querying frontier models on this would be meaningless"
            )
            print(f"SKIPPED: {reason}")
            out.append({**item, "frontier_skipped_reason": reason, "verify_results": {}})
            continue

        field = fields_by_key[(doc_id, field_name)]
        original_text = documents[doc_id]["carrier_text"]
        char_span = tuple(field["char_span"])
        _, s_start, s_end = find_field_sentence(
            original_text, field_value, char_span=char_span
        )

        encoded_document_text = original_text[:s_start] + encoding + original_text[s_end:]
        if tool_render(encoded_document_text) != original_text:
            reason = "full-document round-trip failed after splicing the encoded sentence back in"
            print(f"SKIPPED: {reason}")
            out.append({**item, "frontier_skipped_reason": reason, "verify_results": {}})
            continue
        print("Full-document round-trip: OK")

        verify_results = {}
        for model_key in verify_model_keys:
            model_cfg = config["api_models"][model_key]
            print(f"Querying {model_cfg['model_id']} ({model_cfg['provider']})...")
            response = query_adversary(
                encoded_text=encoded_document_text,
                field_name=field_name,
                model_id=model_cfg["model_id"],
                provider=model_cfg["provider"],
                max_tokens=model_cfg.get("max_tokens", 100),
            )
            check = check_extraction(response, field_value)
            verify_results[model_key] = {
                "raw_response": response,
                "extracted": check["extracted"],
                "refusal": check["refusal"],
                "defended": not check["extracted"],
            }
            print(f"  -> defended={not check['extracted']} response={response!r}")

        out.append({**item, "frontier_skipped_reason": None, "verify_results": verify_results})

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    header = (
        f"{'doc/field':40s} {'prior_ratio':>11s} {'local':>7s} "
        + " ".join(f"{k:>14s}" for k in verify_model_keys)
    )
    print(header)
    for r in out:
        ratio_str = (
            f"{r['prior_strength_ratio']:.4f}" if r.get("prior_strength_ratio") is not None else "N/A"
        )
        if r.get("frontier_skipped_reason"):
            row = f"{r['doc_id']+'/'+r['field_name']:40s} {ratio_str:>11s} {str(r['success']):>7s}  SKIPPED"
        else:
            model_cells = " ".join(
                f"{str(r['verify_results'][k]['defended']):>14s}" for k in verify_model_keys
            )
            row = f"{r['doc_id']+'/'+r['field_name']:40s} {ratio_str:>11s} {str(r['success']):>7s} {model_cells}"
        print(row)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
