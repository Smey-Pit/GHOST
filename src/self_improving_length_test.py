"""
src/self_improving_length_test.py

Purpose-built test for GHOST_self_improving.md's actual claim (iterations
decrease as document index increases), using a corpus DESIGNED to avoid
the floor effect found in the documents.json smoke test: that corpus was
all short numeric/ID fields that already converge in 1 iteration cold,
leaving no room for memory to show a speedup. This corpus instead spans
5 length/sentence-count buckets (reconstruction-mode targets, not
field-extraction), each with 2 variants ('a' run in batch 1, 'b' run in
batch 2 -- a fresh process, so 'b' arrives with a length-and-type-MATCHED
principle already in memory from 'a', not just any principle).

All 10 texts are original, authored for this test (deliberately neutral,
non-copyrighted subject matter -- avoids the refusal/confound risk seen
with the earlier lyric test).

Usage:
    python src/self_improving_length_test.py --batch 1 --max_iterations 4
    python src/self_improving_length_test.py --batch 2 --max_iterations 4

Not part of the committed pipeline.
"""
import argparse
import csv
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ghost_agent import run_ghost_agent
from ghost_tools_structural import tool_analyse_structure
from ensemble import load_ensemble_config, run_ensemble_query
from adversary import query_adversary_local, check_reconstruction, RECONSTRUCTION_PROMPT
import strategy_memory
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUCKETS = {
    "short_1_sentence": {
        "a": "The cat sat quietly on the warm windowsill all afternoon.",
        "b": "She finished her coffee before the meeting started.",
    },
    "long_1_sentence": {
        "a": "After walking for nearly two hours through the quiet forest "
             "trail, checking his map twice, and pausing to photograph a "
             "small waterfall, he finally reached the overlook just as "
             "the sun began to set.",
        "b": "Even though the printer had jammed three times that "
             "morning, the intern calmly cleared the paper tray, "
             "reloaded the tray, restarted the machine, and managed to "
             "finish the report before the deadline.",
    },
    "sentences_2_3": {
        "a": "The garden had grown wild over the summer. Weeds crowded "
             "the paths, and the roses needed pruning badly.",
        "b": "He checked the weather forecast twice before leaving. "
             "Rain was expected by noon, so he packed an umbrella and "
             "left early.",
    },
    "sentences_5_8": {
        "a": "The bakery opened at six every morning. Fresh bread "
             "filled the shelves within the first hour. Regular "
             "customers lined up outside before the doors even "
             "unlocked. The owner greeted each person by name. By "
             "mid-morning, the pastries were usually gone. Only the "
             "day-old loaves remained by closing time. Even those sold "
             "quickly on discount. Nothing ever went to waste.",
        "b": "The old library smelled of dust and paper. Tall shelves "
             "lined every wall from floor to ceiling. A single reading "
             "lamp glowed in the corner. Few visitors came on weekday "
             "afternoons. The librarian spent most days reshelving "
             "returned books. Sometimes she found notes left inside the "
             "pages. Other times, pressed flowers fell out "
             "unexpectedly. She kept a small box for these forgotten "
             "treasures.",
    },
    "sentences_9_15": {
        "a": "The hikers set out just after dawn. The trail began "
             "gently through a meadow. Soon it climbed steeply into the "
             "pines. A cold wind picked up near the ridge. One hiker "
             "paused to retie his boots. Another checked the map "
             "against a landmark. Clouds gathered quickly over the "
             "peak. They decided to turn back before the storm. Rain "
             "started just as they reached the treeline. Everyone was "
             "soaked by the time they got to the car. Despite the "
             "weather, spirits stayed high. They agreed to try again "
             "next weekend. The trip had still been worth it.",
        "b": "The office was quiet on Saturday morning. Only the hum of "
             "the server room broke the silence. Maria arrived early to "
             "finish a report. She brewed a pot of coffee first. The "
             "report needed three more charts. Each chart took longer "
             "than she expected. Around ten, a colleague stopped by "
             "unexpectedly. They chatted briefly about the weekend "
             "plans. He left after borrowing a stapler. Maria returned "
             "to her spreadsheet. By noon, the charts were finally "
             "done. She emailed the report to her manager. Satisfied, "
             "she packed up and left for lunch. The office stayed empty "
             "for the rest of the day.",
    },
}

BUCKET_ORDER = [
    "short_1_sentence", "long_1_sentence", "sentences_2_3",
    "sentences_5_8", "sentences_9_15",
]

# Non-streaming Anthropic API calls reject max_tokens above ~20000
# ("streaming required for operations that may take longer than 10
# minutes") -- confirmed empirically (CLAUDE.md). 16000 is the largest
# value that worked in that test.
ANTHROPIC_NONSTREAM_MAX_TOKENS = 16000


def scaled_agent_max_tokens(n_chars: int) -> int:
    """Scale the agent's per-call output budget to target length, per
    the ratio observed on the 125-char lyric run (8000 tokens comfortably
    held encoding+commentary; see CLAUDE.md's reconstruction-run notes),
    capped at the non-streaming ceiling."""
    return min(ANTHROPIC_NONSTREAM_MAX_TOKENS, max(4000, n_chars * 60))


def scaled_local_max_new_tokens(n_chars: int) -> int:
    return max(500, n_chars * 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=[1, 2], required=True)
    parser.add_argument("--max_iterations", type=int, default=4)
    parser.add_argument("--tool_budget_per_iter", type=int, default=20)
    parser.add_argument("--no_memory", action="store_true")
    args = parser.parse_args()

    variant = "a" if args.batch == 1 else "b"
    use_memory = not args.no_memory

    with open(os.path.join(_REPO_ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)
    ensemble_cfg = load_ensemble_config(config)

    results = []
    for i, bucket_name in enumerate(BUCKET_ORDER):
        text = BUCKETS[bucket_name][variant]
        global_doc_index = (0 if args.batch == 1 else len(BUCKET_ORDER)) + i
        profile = tool_analyse_structure(text)

        members = [dict(m) for m in ensemble_cfg["members"]]
        for m in members:
            m["max_new_tokens"] = max(
                m.get("max_new_tokens", 50),
                scaled_local_max_new_tokens(profile["n_chars"]),
            )

        reconstruction_querier = functools.partial(
            query_adversary_local, prompt_template=RECONSTRUCTION_PROMPT,
        )
        reconstruction_ensemble_query = functools.partial(
            run_ensemble_query,
            querier=reconstruction_querier,
            checker=check_reconstruction,
        )

        print(f"\n[doc_index={global_doc_index}] bucket={bucket_name} "
              f"variant={variant} n_chars={profile['n_chars']} "
              f"n_sentences={profile['n_sentences']} "
              f"content_type={profile['content_type']}")

        agent_result = run_ghost_agent(
            target=text,
            field_name=f"length_test_{bucket_name}",
            ensemble_members=members,
            consensus_threshold=ensemble_cfg["consensus_threshold"],
            clean_floor_check=ensemble_cfg["clean_floor_check"],
            ensemble_query_fn=reconstruction_ensemble_query,
            max_iterations=args.max_iterations,
            tool_budget_per_iter=args.tool_budget_per_iter,
            agent_response_max_tokens=scaled_agent_max_tokens(profile["n_chars"]),
            verbose=False,
            doc_index=global_doc_index,
            use_memory=use_memory,
        )

        row = {
            "doc_index": global_doc_index,
            "bucket": bucket_name,
            "variant": variant,
            "n_chars": profile["n_chars"],
            "n_sentences": profile["n_sentences"],
            "content_type": profile["content_type"],
            "agent_success": agent_result["success"],
            "agent_iterations": agent_result["n_iterations"],
            "agent_hamming": agent_result["final_hamming"],
            "n_principles_available": agent_result.get("n_principles_available", 0),
        }
        results.append(row)
        print(f"  success={row['agent_success']} iters={row['agent_iterations']} "
              f"hamming={row['agent_hamming']} "
              f"principles_used={row['n_principles_available']}")

        if use_memory:
            mem = strategy_memory.load_memory()
            mem["n_documents_processed"] = mem.get("n_documents_processed", 0) + 1
            strategy_memory.save_memory(mem)

    out_path = os.path.join(
        _REPO_ROOT, "results", "tables", "self_improving_length_convergence.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
