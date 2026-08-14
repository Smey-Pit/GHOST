"""
src/convergence.py

Convergence analysis for GHOST-Agent (Task 6 of GHOST_AGENT_TASKS.md).

Measures:
  1. Success rate: what fraction of fields cross the ensemble panel's
     consensus threshold?
  2. Convergence speed: how many iterations needed?
  3. Content-type breakdown: by domain (financial/medical/legal/technical)
  4. Comparison: agent vs. a quick static-GHOST baseline

DEVIATION FROM THE ORIGINAL TASK SPEC: adversary_model_id/
adversary_provider (a single frontier model) is replaced by the
search-tier ensemble panel resolved from config.yaml's agent_ensemble
section (src/ensemble.py) -- same substitution as Tasks 2-5.

THIS IS A SEARCH-TIER NUMBER ONLY. agent_success/agent_iterations here
measure whether the LOCAL ENSEMBLE proxy was defeated, exactly like the
old Qwen logprob proxy's role in the fixed pipeline -- never the
paper's reported transferability result. Before reporting these
numbers, run config.yaml's agent_ensemble.frontier_verify_models on
the converged encodings this script produces and report the gap
between the two tiers (see project discussion + config.yaml comments).

Run on a small sample (e.g. 3-10 documents) before committing to a
larger run -- both because of ensemble/API cost and because Task 5's
extract_proposed_encoding bugfix (delimiter-based parsing) has not yet
been validated against the REAL agent model at scale, only mocked.

Produces:
  results/tables/agent_convergence.csv
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
from ghost_tools import (  # noqa: E402
    tool_combine, tool_render, tool_get_stored, tool_hamming,
)
from ghost_tools_structural import tool_analyse_structure  # noqa: E402
from ensemble import load_ensemble_config  # noqa: E402
import strategy_memory  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_static_ghost(field_value: str) -> dict:
    """
    Apply a quick static-GHOST baseline (full_rtl + one fixed payload)
    for a fast agent-vs-static comparison point.

    NOT the same as the real pipeline's `ghost` condition
    (src/encode.py), which uses per-character keyed payloads via
    derive_payload_bytes -- that distinction matters for the paper's
    actual Table 3/4 numbers, not for this quick comparison.
    """
    encoded = tool_combine(field_value, "full_rtl", "ghost")
    stored = tool_get_stored(encoded)
    dist = tool_hamming(stored, field_value)
    rendered = tool_render(encoded)
    valid = (rendered == field_value)
    return {
        "encoding": encoded,
        "stored": stored,
        "hamming": dist,
        "valid": valid,
    }


def run_convergence_analysis(
    documents_path: str,
    config_path: str = None,
    n_samples: int = 50,
    max_iterations: int = 5,
    seed: int = 168,
    output_dir: str = None,
    start_index: int = 0,
    use_memory: bool = True,
    run_ghost_agent_fn: Callable = run_ghost_agent,
) -> list:
    """
    Run convergence analysis on a sample of documents.

    For each document, for each field:
      1. Run static GHOST — quick baseline hamming distance
      2. Run GHOST-Agent against the search-tier ensemble — measure
         success (ensemble consensus defeated), iterations, hamming

    Args:
        documents_path: path to data/raw/documents.json
        config_path: path to config.yaml (agent_ensemble section).
            Defaults to the repo root's config.yaml, resolved from this
            file's own location -- NOT CWD, so this works whether
            invoked as `python src/convergence.py` (repo root) or `cd
            src && python convergence.py`.
        n_samples: number of documents to take from the sample window
        max_iterations: max agent iterations per field
        seed: random seed for the document shuffle (see start_index)
        output_dir: where to write CSV results. Same repo-root-anchored
            default as config_path.
        start_index: offset into the SAME seeded shuffle of `documents`
            -- not a fresh independent sample. This is what makes two
            separate process invocations (e.g. `--start_index 0
            --n_samples 3` then `--start_index 3 --n_samples 3`) walk a
            deterministic, non-overlapping sequence of documents, which
            is what "Document 4" meaning something specific and
            reproducible across sessions requires
            (GHOST_self_improving.md's Level 3 definition).
        use_memory: passed straight through to run_ghost_agent --
            False disables strategy_memory.py entirely, reproducing
            this function's pre-memory behaviour.
        run_ghost_agent_fn: defaults to ghost_agent.run_ghost_agent --
            overridable for tests so this can run without a GPU or a
            real Anthropic API key
    """
    config_path = config_path or os.path.join(_REPO_ROOT, "config.yaml")
    output_dir = output_dir or os.path.join(_REPO_ROOT, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble_config = load_ensemble_config(config)

    with open(documents_path) as f:
        documents = json.load(f)

    rng = random.Random(seed)
    shuffled = documents[:]
    rng.shuffle(shuffled)
    sampled = shuffled[start_index:start_index + n_samples]

    results = []

    for i, doc in enumerate(sampled):
        global_doc_index = start_index + i
        domain = doc["domain"]
        doc_id = doc["id"]

        for field_name, field_value in doc["fields"].items():
            print(f"\n[doc_index={global_doc_index}] {doc_id} / "
                  f"{field_name}: '{field_value}'")

            static = run_static_ghost(field_value)
            content_type = tool_analyse_structure(field_value)["content_type"]

            agent_result = run_ghost_agent_fn(
                target=field_value,
                field_name=field_name,
                ensemble_members=ensemble_config["members"],
                consensus_threshold=ensemble_config["consensus_threshold"],
                clean_floor_check=ensemble_config["clean_floor_check"],
                max_iterations=max_iterations,
                verbose=False,
                doc_index=global_doc_index,
                use_memory=use_memory,
            )

            row = {
                "doc_index": global_doc_index,
                "doc_id": doc_id,
                "domain": domain,
                "field_name": field_name,
                "field_value": field_value,
                "field_length": len(field_value),
                "content_type": content_type,
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
            # One document = one increment, regardless of how many
            # fields it has or whether any of them succeeded --
            # counting "documents processed" at the field level (as
            # ghost_agent.py used to) overcounts by ~4x (this dataset's
            # avg fields/doc). This is the only place in the codebase
            # that actually knows where one document ends and the next
            # begins, so it's the only correct place for this counter.
            current_memory = strategy_memory.load_memory()
            current_memory["n_documents_processed"] = (
                current_memory.get("n_documents_processed", 0) + 1
            )
            strategy_memory.save_memory(current_memory)

    out_path = os.path.join(output_dir, "agent_convergence.csv")
    if results:
        # Append, don't overwrite -- a second process invocation with a
        # different start_index (the whole point of --start_index: a
        # genuinely separate later "session" over new documents) needs
        # its rows added to the same convergence curve, not replacing
        # the first invocation's rows.
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    print(f"\n{'='*55}")
    print("CONVERGENCE ANALYSIS SUMMARY (search-tier ensemble only)")
    print(f"{'='*55}")
    print(f"Documents sampled: {len(sampled)}")
    print(f"Fields evaluated: {len(results)}")
    if results:
        success_rate = sum(
            1 for r in results if r["agent_success"]
        ) / len(results)
        mean_iters = sum(
            r["agent_iterations"] for r in results
        ) / len(results)
        mean_improvement = sum(
            r["agent_improvement"] for r in results
        ) / len(results)
        print(f"Agent success rate: {success_rate:.1%}")
        print(f"Mean iterations to success: {mean_iters:.1f}")
        print(
            f"Mean hamming improvement over static: {mean_improvement:.1f}"
        )

        for domain in ["financial", "medical", "legal", "technical"]:
            domain_results = [
                r for r in results if r["domain"] == domain
            ]
            if domain_results:
                domain_success = sum(
                    1 for r in domain_results if r["agent_success"]
                ) / len(domain_results)
                print(f"  {domain}: {domain_success:.1%} success")

    print(f"\nResults written to {out_path}")
    print(
        "NOTE: agent_success above is against the SEARCH-TIER local "
        "ensemble only -- it is not the paper's reported number. Run "
        "config.yaml's agent_ensemble.frontier_verify_models on these "
        "converged encodings and report the search/verify gap before "
        "citing this success rate as a transferability result."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "GHOST-Agent convergence analysis. Paths default to the "
            "repo root regardless of CWD."
        ),
    )
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=168)
    parser.add_argument(
        "--documents_path",
        default=os.path.join(_REPO_ROOT, "data", "raw", "documents.json"),
    )
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--no_memory", action="store_true",
        help="disable strategy_memory.py (Level 3 persistence) entirely",
    )
    args = parser.parse_args()

    run_convergence_analysis(
        documents_path=args.documents_path,
        config_path=args.config_path,
        n_samples=args.n_samples,
        max_iterations=args.max_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
        start_index=args.start_index,
        use_memory=not args.no_memory,
    )
