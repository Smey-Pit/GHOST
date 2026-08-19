"""
Scratch driver: first real test of GHOST-Agent's sentence-scope extension
(src/ghost_agent.py's new field_ground_truth param) -- point the agent (using
Claude Haiku 4.5 as backbone, config.yaml's agent_backbones.options.
claude_haiku) at the SENTENCE containing a field, not just the bare field
value, then verify the resulting encoding against a frontier adversary panel.

Pilot example: regfiling_0000 (main convergence pipeline's dataset) is
deliberately NOT reused here -- regfiling_0003's jurisdiction_code sentence
contains only that one field, cleaner to interpret for a first end-to-end
test than a sentence with two co-occurring fields.

CAVEAT, not silently proceeded past: claude_haiku is used as BOTH the agent
backbone (proposing/searching for the encoding) and one of the four
frontier verify-tier adversaries testing it. If Haiku "defeats" its own
encoding in the verify tier, that column's result should be read with the
same skepticism as a model grading its own homework -- the other three
verify-tier models are the more informative signal.

use_memory=False deliberately: strategy_memory.json's principles were all
distilled from FIELD-scope attempts under a schema that doesn't distinguish
scope. Keep this exploratory run isolated rather than reading/writing that
shared file under a fundamentally different target scope.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml
from ghost_agent import run_ghost_agent
from ensemble import load_ensemble_config, run_ensemble_query
from agent_backbone import resolve_backbone
from sentence_utils import find_field_sentence
from adversary import query_adversary, check_extraction

DOC_ID = "regfiling_0003"
FIELD_NAME = "jurisdiction_code"
BACKBONE_NAME = "claude_haiku"
VERIFY_MODELS = ["claude_haiku", "claude_sonnet", "gpt56_sol", "gemini_31_pro"]

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

ensemble_cfg = load_ensemble_config(config)

# ── Locate the pilot example and its sentence ───────────────────────────

pilot_docs = {}
with open(os.path.join(REPO_ROOT, "data", "track_a", "pilot", "pilot_documents.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        pilot_docs[d["doc_id"]] = d

pilot_field = None
with open(os.path.join(REPO_ROOT, "data", "track_a", "pilot", "pilot_fields.jsonl")) as f:
    for line in f:
        rec = json.loads(line)
        if rec["doc_id"] == DOC_ID and rec["field_name"] == FIELD_NAME:
            pilot_field = rec
            break
assert pilot_field is not None, f"{FIELD_NAME} not found for {DOC_ID}"

TARGET_VALUE = pilot_field["ground_truth"]
carrier_text = pilot_docs[DOC_ID]["carrier_text"]

located = find_field_sentence(carrier_text, TARGET_VALUE, char_span=tuple(pilot_field["char_span"]))
assert located is not None, "could not locate field's sentence"
sentence_text, s_start, s_end = located

print(f"Doc: {DOC_ID}  Field: {FIELD_NAME} = {TARGET_VALUE!r}")
print(f"Sentence ({len(sentence_text)} chars): {sentence_text!r}\n")

# ── Search tier: GHOST-Agent with Haiku as backbone, sentence scope ─────

print(f"Loading agent backbone '{BACKBONE_NAME}' ...", flush=True)
backbone = resolve_backbone(BACKBONE_NAME, config)

print(f"Running GHOST-Agent (sentence scope) with backbone={BACKBONE_NAME} ...", flush=True)
try:
    result = run_ghost_agent(
        target=sentence_text,
        field_name=FIELD_NAME,
        field_ground_truth=TARGET_VALUE,
        ensemble_members=ensemble_cfg["members"],
        consensus_threshold=ensemble_cfg["consensus_threshold"],
        clean_floor_check=ensemble_cfg["clean_floor_check"],
        ensemble_query_fn=run_ensemble_query,
        backbone=backbone,
        use_memory=False,
        doc_index=0,
        # Defaults (tool_budget_per_iter=8, agent_response_max_tokens=2000)
        # are calibrated for short numeric fields. CLAUDE.md's "Real
        # reconstruction run" already found a 125-char target needed
        # tool_budget_per_iter=20/agent_response_max_tokens=8000 -- our
        # 157-char sentence is in the same regime. The first real run of
        # this script used the defaults and failed to converge in all 5
        # iterations, every one cut off before the closing
        # <final_encoding> tag (confirmed: the agent's reasoning showed
        # real tool calls/strategy, just no room to finish) -- exactly
        # this already-documented failure mode, not a Haiku capability
        # gap.
        tool_budget_per_iter=20,
        agent_response_max_tokens=8000,
    )
finally:
    backbone.close()

print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))

encoding = result["best_encoding"]
if encoding is None:
    print("No encoding produced -- aborting verify tier.")
    sys.exit(1)

print(f"\nConverged encoding (search tier defended={result['success']}):")
print(repr(encoding))

# ── Verify tier: frontier adversary panel, graded against the FIELD ─────

verify_results = {}
for name in VERIFY_MODELS:
    model_cfg = config["api_models"][name]
    response = query_adversary(
        encoded_text=encoding,
        field_name=FIELD_NAME,
        model_id=model_cfg["model_id"],
        provider=model_cfg["provider"],
        max_tokens=model_cfg["max_tokens"],
    )
    verdict = check_extraction(response, TARGET_VALUE)
    verify_results[name] = {"response": response, **verdict}
    print(f"\n[{name}] response={response!r}")
    print(f"[{name}] verdict={verdict}")

print(
    "\nCAVEAT: claude_haiku appears both as the agent backbone (proposed "
    "this encoding) and as a verify-tier adversary above -- read that "
    "column with the same skepticism as a model grading its own homework."
)

out_path = os.path.join(
    REPO_ROOT, "results", "raw",
    f"verify_sentence_agent_haiku_{FIELD_NAME}.json",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(
        {
            "doc_id": DOC_ID, "field_name": FIELD_NAME, "target_value": TARGET_VALUE,
            "sentence": sentence_text, "agent_backbone": BACKBONE_NAME,
            "encoding": encoding,
            "search_result": {k: v for k, v in result.items() if k != "history"},
            "verify_results": verify_results,
        },
        f, indent=2,
    )
print(f"\nSaved to {out_path}")
