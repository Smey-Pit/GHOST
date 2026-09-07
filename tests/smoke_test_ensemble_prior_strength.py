"""
tests/smoke_test_ensemble_prior_strength.py

Tests src/ensemble.py's run_ensemble_prior_strength WITHOUT a GPU, by
injecting fake loader/unloader/scorer functions -- same DI pattern
src/smoke_test_ensemble.py already uses for run_ensemble_query. The
real math (score_prior_strength itself) is already covered by
tests/smoke_test_prior_strength.py against deterministic fake
tokenizer/model objects; this test only needs to verify the
load-one-at-a-time/query/unload/average orchestration is correct.
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ensemble import load_ensemble_config, run_ensemble_prior_strength  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def fake_loader(hf_id, dtype):
    load_log.append(("load", hf_id))
    return (hf_id, hf_id)  # no real tokenizer/model needed


def fake_unloader(tokenizer, model):
    load_log.append(("unload", tokenizer))


def fake_scorer(tokenizer, model, target, context):
    # Deterministic per-member ratio keyed off the fake hf_id string, so
    # the test can check the mean and per-member values without any
    # real forward pass.
    ratio_by_hf_id = {
        "meta-llama/Llama-3.1-8B-Instruct": 0.2,
        "mistralai/Mistral-7B-Instruct-v0.3": 0.4,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 0.6,
        "deepseek-ai/deepseek-llm-7b-chat": 0.8,
    }
    ratio = ratio_by_hf_id[tokenizer]  # fake_loader returns (hf_id, hf_id)
    return {
        "nll_bits": ratio * 100,
        "compressed_bits": 100,
        "ratio": ratio,
        "min_k_bits": ratio * 50,
    }


load_log = []

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
resolved = load_ensemble_config(config)

# -- Test 1: averages the ratio (not raw bits) across all 4 members --
result = run_ensemble_prior_strength(
    target="123456789",
    context="Account Number: ",
    members=resolved["members"],
    loader=fake_loader,
    unloader=fake_unloader,
    scorer=fake_scorer,
)
assert len(result["per_member"]) == 4, result
expected_mean = (0.2 + 0.4 + 0.6 + 0.8) / 4
assert abs(result["mean_ratio"] - expected_mean) < 1e-9, result
print(f"Test 1 passed: mean_ratio={result['mean_ratio']} == {expected_mean}")

# -- Test 2: per-member dicts carry name/hf_id + the full score profile --
names = {m["name"] for m in result["per_member"]}
assert names == {"llama31_8b", "mistral7b", "deepseek_r1_14b", "deepseek_llm_7b"}, names
for m in result["per_member"]:
    for key in ("hf_id", "ratio", "nll_bits", "compressed_bits", "min_k_bits"):
        assert key in m, f"missing {key!r} in {m}"
print("Test 2 passed: per-member keys present")

# -- Test 3: each member is loaded exactly once and unloaded exactly once,
# in order (mirrors run_ensemble_query's one-at-a-time GPU-sharing rule) --
load_events = [e for e in load_log if e[0] == "load"]
unload_events = [e for e in load_log if e[0] == "unload"]
assert len(load_events) == 4, load_log
assert len(unload_events) == 4, load_log
# every load must be immediately followed by its own unload before the
# next member's load (one-at-a-time, not batch-loaded)
for i in range(4):
    load_idx = load_log.index(load_events[i])
    assert load_log[load_idx + 1][0] == "unload", (
        "expected the very next event after a load to be that member's "
        f"unload (one-at-a-time lifecycle), got: {load_log}"
    )
print("Test 3 passed: load-one-at-a-time-then-unload lifecycle respected")

# -- Test 4: empty members list doesn't crash (mean_ratio is nan, not an
# error) -- defensive, not a real call site today, but a callable/DI
# function should degrade rather than raise on an empty input.
import math  # noqa: E402

empty_result = run_ensemble_prior_strength(
    target="x", context="ctx", members=[],
    loader=fake_loader, unloader=fake_unloader, scorer=fake_scorer,
)
assert empty_result["per_member"] == []
assert math.isnan(empty_result["mean_ratio"])
print("Test 4 passed: empty members list degrades gracefully")

print("ENSEMBLE PRIOR-STRENGTH SMOKE TEST PASSED")
