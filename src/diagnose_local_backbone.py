"""
Scratch diagnostic: isolate LocalReActBackbone's tool-call behavior on the
real deepseek_r1_32b model, WITHOUT loading the search-tier ensemble at
all (much lighter/faster than the full run_ghost_agent pipeline). Prints
the FULL raw model text at each step (ghost_agent.py's print statements
truncate to 300/500 chars, which hides exactly the detail needed to
diagnose why <final_encoding> tags aren't appearing).

Two probes:
  1. Iteration-1 turn: the real build_iteration_prompt() first-iteration
     prompt, same as the real pipeline sends.
  2. Iteration-2 turn: a synthetic "NOT YET DEFENDED" result message
     (fabricated, no real ensemble call) fed back, reproducing the
     iteration-2+ prompt shape from the real run without needing the
     ensemble loaded.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml
from agent_backbone import resolve_backbone
from ghost_agent_prompt import GHOST_AGENT_SYSTEM_PROMPT, build_iteration_prompt
from ghost_agent import extract_proposed_encoding

TARGET = "QLD-17"
FIELD_NAME = "jurisdiction_code"

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

print("Loading deepseek_r1_32b backbone (no ensemble)...", flush=True)
backbone = resolve_backbone("deepseek_r1_32b", config)
print("Loaded.\n", flush=True)


def dump_turn(label, final_text):
    print(f"\n{'='*70}\n{label}: FULL raw final_text ({len(final_text)} chars)\n{'='*70}")
    print(final_text)
    print(f"{'='*70}")
    tag_found = extract_proposed_encoding(final_text, TARGET)
    print(f"[{label}] extract_proposed_encoding -> {tag_found!r}")


# ── Probe 1: iteration 1 ─────────────────────────────────────────────────
user_msg_1 = build_iteration_prompt(
    target=TARGET, field_name=FIELD_NAME, reflection="", iteration=1,
    budget_remaining=8,
)
final_text_1 = backbone.run_turn(
    user_msg_1, GHOST_AGENT_SYSTEM_PROMPT,
    tool_budget_per_iter=8, response_max_tokens=4000,
)
dump_turn("ITERATION 1", final_text_1)

print(f"\n--- Full message transcript after iteration 1 ({len(backbone.messages)} messages) ---")
for i, m in enumerate(backbone.messages):
    content_preview = m["content"] if isinstance(m["content"], str) else str(m["content"])
    print(f"[{i}] role={m['role']!r} len={len(content_preview)} "
          f"preview={content_preview[:150]!r}")


# ── Probe 2: iteration 2, synthetic NOT YET DEFENDED feedback ───────────
synthetic_result_msg = (
    "Ensemble panel result: 2/3 valid members defeated.\n"
    "Consensus threshold: 3.\n"
    "NOT YET DEFENDED."
)
backbone.messages.append({"role": "user", "content": synthetic_result_msg})

reflection = (
    "Attempt 1: bidi_config=from_agent, vs_payload=from_agent, "
    "hamming_dist=0, defended=False. Panel: llama31_8b=EXTRACTED, "
    "mistral7b=EXTRACTED, deepseek_r1_14b=EXTRACTED."
)
user_msg_2 = build_iteration_prompt(
    target=TARGET, field_name=FIELD_NAME, reflection=reflection, iteration=2,
    budget_remaining=8,
)
final_text_2 = backbone.run_turn(
    user_msg_2, GHOST_AGENT_SYSTEM_PROMPT,
    tool_budget_per_iter=8, response_max_tokens=4000,
)
dump_turn("ITERATION 2 (synthetic feedback)", final_text_2)

print(f"\n--- Full message transcript after iteration 2 ({len(backbone.messages)} messages) ---")
for i, m in enumerate(backbone.messages):
    content_preview = m["content"] if isinstance(m["content"], str) else str(m["content"])
    print(f"[{i}] role={m['role']!r} len={len(content_preview)} "
          f"preview={content_preview[:150]!r}")

backbone.close()
print("\nDone.")
