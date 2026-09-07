"""
tests/smoke_test_ghost_agent_prior_strength.py

Verifies run_ghost_agent's prior_strength_fn wiring end-to-end (the
content_profile merge happening inside _run_ghost_agent_loop) WITHOUT a
real Anthropic API key, GPU, or touching any real results/ file --
same fake-client/fake-ensemble pattern src/smoke_test_task5.py already
uses. use_memory=False throughout (same reason smoke_test_task5.py uses
it: a real distillation call and real file writes are out of scope for
a plumbing test), so this checks the merge via the verbose print line
rather than inspecting a written experience_log.jsonl/strategy_memory.json.
"""

import contextlib
import io
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ghost_agent import run_ghost_agent  # noqa: E402
from ghost_tools import tool_combine  # noqa: E402
from ensemble import load_ensemble_config  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
resolved = load_ensemble_config(config)

target = "12345"
encoded = tool_combine(target, "full_rtl", "ghost")
texts = [f"Trying a reversal.\n<final_encoding>{encoded}</final_encoding>"]


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def create(self, **kwargs):
        text = self.texts[self.calls]
        self.calls += 1
        return FakeResponse([FakeTextBlock(text)])


class FakeClient:
    def __init__(self, texts):
        self.messages = FakeMessages(texts)


def fake_ensemble_query(**kwargs):
    per_member = [
        {"name": m["name"], "valid": True, "refusal": False, "extracted": False}
        for m in kwargs["members"]
    ]
    return {
        "defended": True,
        "n_valid": len(per_member),
        "n_failed": len(per_member),
        "per_member": per_member,
    }


# -- Test 1: prior_strength_fn is called with (target, "field_name: ")
# and its result reaches the verbose log via content_profile's merge --
prior_strength_calls = []


def fake_prior_strength_fn(target_arg, context_arg):
    prior_strength_calls.append((target_arg, context_arg))
    return {
        "mean_ratio": 0.4242,
        "per_member": [{"name": "fake_member", "ratio": 0.4242}],
    }


stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    result = run_ghost_agent(
        target=target,
        field_name="test_value",
        ensemble_members=resolved["members"],
        consensus_threshold=resolved["consensus_threshold"],
        clean_floor_check=resolved["clean_floor_check"],
        max_iterations=5,
        verbose=True,
        client=FakeClient(list(texts)),
        ensemble_query_fn=fake_ensemble_query,
        use_memory=False,
        prior_strength_fn=fake_prior_strength_fn,
    )
output = stdout.getvalue()

assert result["success"] is True, result
assert prior_strength_calls == [("12345", "test_value: ")], prior_strength_calls
assert "Prior-strength ratio (ensemble mean): 0.4242" in output, output
print("Test 1 passed: prior_strength_fn called once with (target, field-label "
      "context), result reaches verbose log")

# -- Test 2: prior_strength_fn=None (default) is a complete no-op -- no
# call, no print line, identical success/iteration outcome --
stdout2 = io.StringIO()
with contextlib.redirect_stdout(stdout2):
    result2 = run_ghost_agent(
        target=target,
        field_name="test_value",
        ensemble_members=resolved["members"],
        consensus_threshold=resolved["consensus_threshold"],
        clean_floor_check=resolved["clean_floor_check"],
        max_iterations=5,
        verbose=True,
        client=FakeClient(list(texts)),
        ensemble_query_fn=fake_ensemble_query,
        use_memory=False,
    )
output2 = stdout2.getvalue()
assert result2["success"] == result["success"]
assert "Prior-strength ratio" not in output2, output2
print("Test 2 passed: prior_strength_fn=None is a no-op, zero behavior change")


# -- Test 3: a raising prior_strength_fn is caught, not fatal --
def raising_prior_strength_fn(target_arg, context_arg):
    raise RuntimeError("simulated GPU load failure")


stdout3 = io.StringIO()
with contextlib.redirect_stdout(stdout3):
    result3 = run_ghost_agent(
        target=target,
        field_name="test_value",
        ensemble_members=resolved["members"],
        consensus_threshold=resolved["consensus_threshold"],
        clean_floor_check=resolved["clean_floor_check"],
        max_iterations=5,
        verbose=True,
        client=FakeClient(list(texts)),
        ensemble_query_fn=fake_ensemble_query,
        use_memory=False,
        prior_strength_fn=raising_prior_strength_fn,
    )
output3 = stdout3.getvalue()
assert result3["success"] is True, result3  # the run itself still succeeds
assert "Prior-strength scoring failed" in output3, output3
print("Test 3 passed: a raising prior_strength_fn is caught, run still succeeds")

print("GHOST_AGENT PRIOR-STRENGTH SMOKE TEST PASSED")
