"""
Scratch driver: run GHOST-Agent search tier on one field, then verify the
converged encoding against all three frontier models in
config.yaml's agent_ensemble.frontier_verify_models -- including
gemini_31_pro, which has not yet been tested against a converged
GHOST-Agent encoding (only smoke-tested for raw API connectivity so far).

Not part of the committed pipeline -- convergence.py still needs the real
frontier-verify runner written in (see CLAUDE.md). This reproduces what
the prior uncommitted scratch script did for claude_sonnet/gpt56_sol, and
extends it to gemini_31_pro.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml
from ghost_agent import run_ghost_agent
from ensemble import load_ensemble_config, run_ensemble_query
from adversary import query_adversary, check_extraction

FIELD_NAME = "account_number"
TARGET = "5926847003"

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

ensemble_cfg = load_ensemble_config(config)

print(f"Running GHOST-Agent search tier on {FIELD_NAME}={TARGET} ...")
result = run_ghost_agent(
    target=TARGET,
    field_name=FIELD_NAME,
    ensemble_members=ensemble_cfg["members"],
    consensus_threshold=ensemble_cfg["consensus_threshold"],
    clean_floor_check=ensemble_cfg["clean_floor_check"],
    ensemble_query_fn=run_ensemble_query,
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
    )
    verdict = check_extraction(response, TARGET)
    verify_results[name] = {"response": response, **verdict}
    print(f"\n[{name}] response={response!r}")
    print(f"[{name}] verdict={verdict}")

out_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "raw",
    f"verify_gemini_{FIELD_NAME}.json",
)
with open(out_path, "w") as f:
    json.dump(
        {"field": FIELD_NAME, "target": TARGET, "encoding": encoding,
         "search_result": {k: v for k, v in result.items() if k != "history"},
         "verify_results": verify_results},
        f, indent=2,
    )
print(f"\nSaved to {out_path}")
