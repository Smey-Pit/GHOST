"""
Second reconstruction-mode run, this time on a much longer target (a
~1185-char book-summary paragraph, vs. the 127-char lyric sentence in
verify_reconstruction_run.py) -- testing whether GHOST-Agent's
single-shot "write the whole encoding in one final text block" approach
scales to paragraph-length text, or hits a real infrastructure ceiling
(Anthropic's non-streaming client enforces max_tokens <= ~16000-20000
before requiring stream=True, confirmed empirically -- see CLAUDE.md).

Local ensemble members' max_new_tokens (config.yaml, tuned for short
field-extraction responses: 50, or 2048 for the one reasoning member)
are overridden here to something that could plausibly hold a full
paragraph reconstruction attempt -- otherwise a "defended" verdict would
be confounded by the same response-budget-too-small bug already found
and fixed twice in this project (short-field defaults silently starving
longer-output tasks).

Not part of the committed pipeline.
"""
import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml
from ghost_agent import run_ghost_agent
from ensemble import load_ensemble_config, run_ensemble_query
from adversary import (
    query_adversary, query_adversary_local, check_reconstruction,
    RECONSTRUCTION_PROMPT,
)

FIELD_NAME = "book_summary"
TARGET = (
    "Even the smartest among us can feel inept as we fail to figure our "
    "which light switch or oven burner to turn on, or whether to push, "
    "pull, or slide a door. The fault, argues this book, lies not in "
    "ourselves, but in product design that ignores the needs of users "
    "and the principles of cognitive psychology. The problems range "
    "from ambiguous and hidden controls to arbitrary relationships "
    "between controls and functions, coupled with a lack of feedback "
    "or other assistance and unreasonable demands on memorization. The "
    "book presents examples aplenty, among them, the VCR, computer, "
    "and office telephone, all models of how not to design for people. "
    "But good, usable design is possible. The rules are simple: make "
    "things visible, exploit natural relationships that couple "
    "function and control, and make intelligent use of constraints. "
    "The goal: guide the user effortlessly to the right action on the "
    "right control at the right time. But the designer must care. The "
    "author is a world-famous psychologist and pioneer in the "
    "application of cognitive science. His aim is to raise the "
    "consciousness of both consumers and designers to the delights of "
    "products that are easy to use and understand"
)

ANTHROPIC_NONSTREAM_MAX_TOKENS = 16000
LOCAL_RECONSTRUCTION_MAX_NEW_TOKENS = 3000

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

ensemble_cfg = load_ensemble_config(config)
for member in ensemble_cfg["members"]:
    member["max_new_tokens"] = max(
        member.get("max_new_tokens", 50), LOCAL_RECONSTRUCTION_MAX_NEW_TOKENS,
    )

reconstruction_querier = functools.partial(
    query_adversary_local, prompt_template=RECONSTRUCTION_PROMPT,
)
reconstruction_ensemble_query = functools.partial(
    run_ensemble_query,
    querier=reconstruction_querier,
    checker=check_reconstruction,
)

print(f"Running GHOST-Agent search tier on {FIELD_NAME!r} ({len(TARGET)} chars) ...")
result = run_ghost_agent(
    target=TARGET,
    field_name=FIELD_NAME,
    ensemble_members=ensemble_cfg["members"],
    consensus_threshold=ensemble_cfg["consensus_threshold"],
    clean_floor_check=ensemble_cfg["clean_floor_check"],
    ensemble_query_fn=reconstruction_ensemble_query,
    tool_budget_per_iter=25,
    agent_response_max_tokens=ANTHROPIC_NONSTREAM_MAX_TOKENS,
)

print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))

encoding = result["best_encoding"]
if encoding is None:
    print("No encoding produced -- aborting verify tier.")
    sys.exit(1)

print(f"\nConverged encoding length: {len(encoding)} chars "
      f"(defended={result['success']})")

verify_results = {}
for name in config["agent_ensemble"]["frontier_verify_models"]:
    model_cfg = config["api_models"][name]
    max_tokens = max(model_cfg["max_tokens"], ANTHROPIC_NONSTREAM_MAX_TOKENS)
    response = query_adversary(
        encoded_text=encoding,
        field_name=FIELD_NAME,
        model_id=model_cfg["model_id"],
        provider=model_cfg["provider"],
        max_tokens=max_tokens,
        prompt_template=RECONSTRUCTION_PROMPT,
    )
    verdict = check_reconstruction(response, TARGET)
    verify_results[name] = {"response": response, **verdict}
    print(f"\n[{name}] max_tokens={max_tokens}")
    print(f"[{name}] response={response[:300]!r}")
    print(f"[{name}] similarity={verdict['similarity']:.3f} "
          f"reconstructed={verdict['extracted']} refusal={verdict['refusal']}")

out_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "raw",
    "verify_reconstruction_book_summary.json",
)
with open(out_path, "w") as f:
    json.dump(
        {"field": FIELD_NAME, "target": TARGET, "encoding": encoding,
         "search_result": {k: v for k, v in result.items() if k != "history"},
         "verify_results": verify_results},
        f, indent=2,
    )
print(f"\nSaved to {out_path}")
