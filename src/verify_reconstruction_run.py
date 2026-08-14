"""
Scratch driver: run GHOST-Agent on a natural-language sentence (not a
numeric field) and score with reconstruction cosine similarity instead
of exact-match extraction -- the Phase 6 gradient convention (CLAUDE.md),
applied through GHOST-Agent's per-instance search instead of the fixed
static pipeline. gradient.py itself is still a stub; this reuses
adversary.check_reconstruction / RECONSTRUCTION_PROMPT (added for this
run) via run_ensemble_query's existing querier/checker injection points
rather than duplicating the ensemble load/unload lifecycle.

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

FIELD_NAME = "lyric"
TARGET = (
    "Loving him is like driving a new Maserati down a dead-end street, "
    "Faster than the wind, passionate as sin, ending so suddenly"
)

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

ensemble_cfg = load_ensemble_config(config)

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
    tool_budget_per_iter=20,
    agent_response_max_tokens=8000,
)

print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))

encoding = result["best_encoding"]
if encoding is None:
    print("No encoding produced -- aborting verify tier.")
    sys.exit(1)

print(f"\nConverged encoding (search tier defended={result['success']}):")
print(repr(encoding))

verify_results = {}
for name in config["agent_ensemble"]["frontier_verify_models"]:
    model_cfg = config["api_models"][name]
    response = query_adversary(
        encoded_text=encoding,
        field_name=FIELD_NAME,
        model_id=model_cfg["model_id"],
        provider=model_cfg["provider"],
        max_tokens=model_cfg["max_tokens"],
        prompt_template=RECONSTRUCTION_PROMPT,
    )
    verdict = check_reconstruction(response, TARGET)
    verify_results[name] = {"response": response, **verdict}
    print(f"\n[{name}] response={response!r}")
    print(f"[{name}] similarity={verdict['similarity']:.3f} reconstructed={verdict['extracted']}")

out_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "raw",
    "verify_reconstruction_lyric.json",
)
with open(out_path, "w") as f:
    json.dump(
        {"field": FIELD_NAME, "target": TARGET, "encoding": encoding,
         "search_result": {k: v for k, v in result.items() if k != "history"},
         "verify_results": verify_results},
        f, indent=2,
    )
print(f"\nSaved to {out_path}")
