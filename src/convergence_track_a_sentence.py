"""
src/convergence_track_a_sentence.py

First BATCH driver for GHOST-Agent's sentence-scope extension
(ghost_agent.py's field_ground_truth param) against real Track A data
(data/track_a/full/{documents,fields}.jsonl) -- everything before this
was a single hand-run script on one field
(verify_sentence_agent_haiku.py, regfiling_0003/jurisdiction_code only).

DELIBERATELY SEPARATE from src/convergence.py, not a flag added to it:
that script assumes data/raw/documents.json's field-scope shape
(doc["fields"] = {name: value}, target=field_value). Track A's shape is
different (documents.jsonl + fields.jsonl, char_span-based), and the
target here is a SENTENCE, not a bare field, with field_ground_truth
used for the actual extraction success check. Mixing both shapes into
one script was judged more confusing than two small scripts sharing the
same run_ghost_agent/backbone/proxy plumbing.

SHARED-SENTENCE DEDUP: when two+ fields fall in the same sentence (a
real case in Track A, same as the static bidi_permute_sentence
condition already handles), the agent is run ONCE per unique sentence,
not once per field -- matches how this would actually ship, and avoids
wasting agent/API calls re-obfuscating the same text twice. The agent's
own stopping condition (field_ground_truth) is driven by exactly ONE
field per sentence -- the FIRST field encountered for that span, by
fields.jsonl order. Any other field co-occupying the same sentence is
recorded as metadata (co_occurring_fields) but its own extractability
against the converged encoding is NOT independently re-verified in this
script -- a known, documented simplification for a first pilot, not an
oversight.

Fields whose ground truth cannot be located in the document at all
(find_field_sentence returns None -- Track A's own triage found ~38
such genuinely-paraphrased fields, see CLAUDE.md's Track A section) are
skipped and counted, not silently dropped.

use_memory defaults to False, opposite of convergence.py's default:
strategy_memory.json's existing principles were all distilled from
FIELD-scope attempts under a schema that doesn't distinguish scope
(same caution already documented in verify_sentence_agent_haiku.py) --
this batch script should not read OR write that file unless explicitly
asked to via --use_memory.

Produces:
  results/tables/agent_convergence_track_a_sentence.csv
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Callable

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from ghost_agent import run_ghost_agent  # noqa: E402
from ghost_tools import tool_combine, tool_render, tool_get_stored, tool_hamming  # noqa: E402
from ghost_tools_structural import tool_analyse_structure  # noqa: E402
from ensemble import load_ensemble_config  # noqa: E402
from agent_backbone import resolve_backbone  # noqa: E402
from sentence_utils import find_field_sentence  # noqa: E402
import strategy_memory  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_static_ghost(text: str) -> dict:
    """Same quick full_rtl+ghost baseline convergence.py uses, applied
    to a sentence instead of a bare field -- generic over string length."""
    encoded = tool_combine(text, "full_rtl", "ghost")
    stored = tool_get_stored(encoded)
    dist = tool_hamming(stored, text)
    rendered = tool_render(encoded)
    valid = (rendered == text)
    return {"encoding": encoded, "stored": stored, "hamming": dist, "valid": valid}


def load_track_a(track_a_dir: str):
    """
    Returns {doc_id: {..doc fields.., "field_records": [field_dict, ...]}}
    with field_records in the same order they appear in fields.jsonl
    (matters for "first field in a shared sentence is primary").
    """
    docs = {}
    with open(os.path.join(track_a_dir, "documents.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            d["field_records"] = []
            docs[d["doc_id"]] = d

    with open(os.path.join(track_a_dir, "fields.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec["doc_id"] in docs:
                docs[rec["doc_id"]]["field_records"].append(rec)

    return docs


def group_fields_by_sentence(doc: dict) -> list:
    """
    For one Track A doc, locate each field's sentence and group fields
    sharing the same (start, end) span. Returns a list of dicts:
      {"sentence_text", "s_start", "s_end", "fields": [field_record, ...]}
    in first-encountered order (fields.jsonl order), plus a separate
    count of fields that could not be located at all.
    """
    groups_by_span = {}
    order = []
    n_unlocatable = 0

    for field in doc["field_records"]:
        located = find_field_sentence(
            doc["carrier_text"], field["ground_truth"],
            char_span=tuple(field["char_span"]),
        )
        if located is None:
            n_unlocatable += 1
            continue
        sentence_text, s_start, s_end = located
        key = (s_start, s_end)
        if key not in groups_by_span:
            groups_by_span[key] = {
                "sentence_text": sentence_text,
                "s_start": s_start, "s_end": s_end,
                "fields": [],
            }
            order.append(key)
        groups_by_span[key]["fields"].append(field)

    return [groups_by_span[k] for k in order], n_unlocatable


def run_convergence_analysis_track_a_sentence(
    track_a_dir: str,
    config_path: str = None,
    n_samples: int = 10,
    max_iterations: int = 5,
    seed: int = 168,
    output_dir: str = None,
    start_index: int = 0,
    use_memory: bool = False,
    agent_backbone_name: str = None,
    use_proxy: bool = False,
    tool_budget_per_iter: int = 20,
    agent_response_max_tokens: int = 8000,
    run_ghost_agent_fn: Callable = run_ghost_agent,
) -> list:
    config_path = config_path or os.path.join(_REPO_ROOT, "config.yaml")
    output_dir = output_dir or os.path.join(_REPO_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble_config = load_ensemble_config(config)

    docs = load_track_a(track_a_dir)
    doc_ids = sorted(docs.keys())

    rng = random.Random(seed)
    shuffled = doc_ids[:]
    rng.shuffle(shuffled)
    sampled_ids = shuffled[start_index:start_index + n_samples]

    results = []
    n_unlocatable_total = 0
    n_sentences_total = 0

    backbone = None
    if agent_backbone_name is not None:
        backbone = resolve_backbone(agent_backbone_name, config)
    backbone_label = agent_backbone_name or config["agent_backbones"]["default"]

    proxy = None
    if use_proxy:
        from encode import load_proxy_model
        print("Loading Qwen2.5-7B-Instruct proxy (GPU)...")
        proxy = load_proxy_model(config)

    try:
        for i, doc_id in enumerate(sampled_ids):
            global_doc_index = start_index + i
            doc = docs[doc_id]
            domain = doc["domain"]

            sentence_groups, n_unlocatable = group_fields_by_sentence(doc)
            n_unlocatable_total += n_unlocatable
            n_sentences_total += len(sentence_groups)

            for group in sentence_groups:
                sentence_text = group["sentence_text"]
                primary_field = group["fields"][0]
                co_fields = [f["field_name"] for f in group["fields"][1:]]
                field_name = primary_field["field_name"]
                field_value = primary_field["ground_truth"]

                print(
                    f"\n[doc_index={global_doc_index}] {doc_id} / "
                    f"{field_name} (sentence, {len(sentence_text)} chars, "
                    f"co_fields={co_fields}): {sentence_text!r}"
                )

                static = run_static_ghost(sentence_text)
                content_type = tool_analyse_structure(sentence_text)["content_type"]

                agent_result = run_ghost_agent_fn(
                    target=sentence_text,
                    field_name=field_name,
                    field_ground_truth=field_value,
                    ensemble_members=ensemble_config["members"],
                    consensus_threshold=ensemble_config["consensus_threshold"],
                    clean_floor_check=ensemble_config["clean_floor_check"],
                    max_iterations=max_iterations,
                    verbose=False,
                    backbone=backbone,
                    doc_index=global_doc_index,
                    use_memory=use_memory,
                    proxy=proxy,
                    tool_budget_per_iter=tool_budget_per_iter,
                    agent_response_max_tokens=agent_response_max_tokens,
                )

                row = {
                    "doc_index": global_doc_index,
                    "doc_id": doc_id,
                    "domain": domain,
                    "field_name": field_name,
                    "field_value": field_value,
                    "co_occurring_fields": ";".join(co_fields),
                    "sentence_length": len(sentence_text),
                    "content_type": content_type,
                    "agent_backbone": backbone_label,
                    "static_hamming": static["hamming"],
                    "static_valid": static["valid"],
                    "agent_success": agent_result["success"],
                    "agent_iterations": agent_result["n_iterations"],
                    "agent_hamming": agent_result["final_hamming"],
                    "agent_improvement":
                        agent_result["final_hamming"] - static["hamming"],
                    "n_principles_available":
                        agent_result.get("n_principles_available", 0),
                }
                results.append(row)
                print(
                    f"  Static hamming: {static['hamming']}  "
                    f"Agent: success={agent_result['success']} "
                    f"iters={agent_result['n_iterations']} "
                    f"hamming={agent_result['final_hamming']} "
                    f"principles_used={row['n_principles_available']}"
                )

            if use_memory:
                current_memory = strategy_memory.load_memory()
                current_memory["n_documents_processed"] = (
                    current_memory.get("n_documents_processed", 0) + 1
                )
                strategy_memory.save_memory(current_memory)
    finally:
        if backbone is not None:
            backbone.close()
        if proxy is not None:
            import torch
            del proxy
            torch.cuda.empty_cache()

    out_path = os.path.join(
        output_dir, "agent_convergence_track_a_sentence.csv",
    )
    if results:
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*55}")
    print("CONVERGENCE ANALYSIS SUMMARY (Track A, sentence scope, search-tier ensemble only)")
    print(f"{'='*55}")
    print(f"Documents sampled: {len(sampled_ids)}")
    print(f"Unique sentences evaluated: {n_sentences_total}")
    print(f"Fields skipped (unlocatable / genuine paraphrase): {n_unlocatable_total}")
    if results:
        success_rate = sum(1 for r in results if r["agent_success"]) / len(results)
        mean_iters = sum(r["agent_iterations"] for r in results) / len(results)
        mean_improvement = sum(r["agent_improvement"] for r in results) / len(results)
        print(f"Agent success rate: {success_rate:.1%}")
        print(f"Mean iterations to success: {mean_iters:.1f}")
        print(f"Mean hamming improvement over static: {mean_improvement:.1f}")

    print(f"\nResults written to {out_path}")
    print(
        "NOTE: agent_success above is against the SEARCH-TIER local "
        "ensemble only, and only the PRIMARY field per sentence was "
        "used to drive the stopping condition -- co_occurring_fields "
        "were NOT independently re-verified against the converged "
        "encoding in this script. Run frontier_verify_models on these "
        "converged encodings before citing any transferability number."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "GHOST-Agent sentence-scope convergence analysis against "
            "real Track A data. Paths default to the repo root "
            "regardless of CWD."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=168)
    parser.add_argument(
        "--track_a_dir",
        default=os.path.join(_REPO_ROOT, "data", "track_a", "full"),
    )
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--use_memory", action="store_true",
        help=(
            "enable strategy_memory.py for this run -- OFF by default "
            "since existing principles were distilled from field-scope "
            "attempts under a schema that doesn't distinguish scope"
        ),
    )
    parser.add_argument(
        "--agent_backbone", default=None,
        help="Key into config.yaml's agent_backbones.options (e.g. claude_haiku).",
    )
    parser.add_argument(
        "--use_proxy", action="store_true",
        help="Load Qwen2.5-7B-Instruct once for combine_permute/encode_vs_logprob. Needs a GPU.",
    )
    parser.add_argument("--tool_budget_per_iter", type=int, default=20)
    parser.add_argument("--agent_response_max_tokens", type=int, default=8000)
    args = parser.parse_args()

    run_convergence_analysis_track_a_sentence(
        track_a_dir=args.track_a_dir,
        config_path=args.config_path,
        n_samples=args.n_samples,
        max_iterations=args.max_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
        start_index=args.start_index,
        use_memory=args.use_memory,
        agent_backbone_name=args.agent_backbone,
        use_proxy=args.use_proxy,
        tool_budget_per_iter=args.tool_budget_per_iter,
        agent_response_max_tokens=args.agent_response_max_tokens,
    )
