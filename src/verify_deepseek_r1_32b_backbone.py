"""
Scratch driver: run GHOST-Agent's search tier with the agent backbone set
to deepseek_r1_32b (config.yaml's agent_backbones.options -- see
src/agent_backbone.py) instead of the claude-sonnet-4-6 default, on one
field pulled from the Track A pilot set
(data/track_a/pilot/pilot_fields.jsonl). Then verify the converged
encoding against all three frontier models in config.yaml's
agent_ensemble.frontier_verify_models -- same pattern as
verify_gemini_run.py, extended to a non-default agent backbone.

Not part of the committed pipeline. First real (non-mocked) exercise of
LocalReActBackbone against a real GPU + real target model.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml
from ghost_agent import run_ghost_agent
from ensemble import load_ensemble_config, run_ensemble_query
from agent_backbone import resolve_backbone
from adversary import query_adversary, check_extraction

FIELD_ID = "regfiling_0000_f01"
FIELD_NAME = "jurisdiction_code"
TARGET = "QLD-17"
DOC_ID = "regfiling_0000"
BACKBONE_NAME = "deepseek_r1_32b"

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

ensemble_cfg = load_ensemble_config(config)

print(f"Loading agent backbone '{BACKBONE_NAME}' ...", flush=True)
backbone = resolve_backbone(BACKBONE_NAME, config)
print("Backbone loaded.", flush=True)

# Debug instrumentation: ghost_agent.py's own prints truncate agent_text
# to 300 chars, which hides whether _parse_tool_calls' fenced-json fix
# actually changed tool-execution behavior on the REAL per-iteration
# prompt (as opposed to the synthetic probe in diagnose_local_backbone.py).
# Wrap run_turn to capture the full untruncated text every call, without
# touching ghost_agent.py itself.
_full_texts = []
_orig_run_turn = backbone.run_turn
def _capturing_run_turn(*args, **kwargs):
    text = _orig_run_turn(*args, **kwargs)
    _full_texts.append(text)
    return text
backbone.run_turn = _capturing_run_turn

print(f"\nRunning GHOST-Agent search tier on {FIELD_NAME}={TARGET!r} "
      f"(doc={DOC_ID}, field_id={FIELD_ID}) with backbone={BACKBONE_NAME} ...",
      flush=True)

try:
    result = run_ghost_agent(
        target=TARGET,
        field_name=FIELD_NAME,
        ensemble_members=ensemble_cfg["members"],
        consensus_threshold=ensemble_cfg["consensus_threshold"],
        clean_floor_check=ensemble_cfg["clean_floor_check"],
        ensemble_query_fn=run_ensemble_query,
        backbone=backbone,
        doc_index=0,
    )
finally:
    backbone.close()

print(f"\n{'#'*70}\nFULL UNTRUNCATED TEXT PER ITERATION ({len(_full_texts)} calls)\n{'#'*70}")
for i, text in enumerate(_full_texts, 1):
    print(f"\n----- call {i} ({len(text)} chars) -----")
    print(text)
print(f"{'#'*70}\n")

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
    REPO_ROOT, "results", "raw",
    f"verify_deepseek_r1_32b_backbone_{FIELD_NAME}.json",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(
        {
            "field_id": FIELD_ID, "doc_id": DOC_ID, "field": FIELD_NAME,
            "target": TARGET, "agent_backbone": BACKBONE_NAME,
            "encoding": encoding,
            "search_result": {k: v for k, v in result.items() if k != "history"},
            "verify_results": verify_results,
        },
        f, indent=2,
    )
print(f"\nSaved to {out_path}")
