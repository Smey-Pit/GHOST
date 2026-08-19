"""
Smoke test for the two "make the agent actually self-direct" fixes added
to src/ghost_agent.py / src/ghost_tools.py / src/agent_backbone.py:

1. require_unanimous: the stopping rule can now require EVERY valid
   ensemble member to fail, not just consensus_threshold -- the previous
   binary rule let the agent stop on iteration 1 essentially every real
   run tested (see CLAUDE.md's "Agentic-ness critique" section), with no
   margin and no reason to ever retry a seed or escalate scope.
2. widen_scope(): a new tool the agent can call mid-run to re-target its
   encoding from the original `target` to a caller-supplied
   `wider_context` (e.g. the sentence containing a field) -- previously
   scope was fixed for the whole run by whoever called run_ghost_agent,
   with no way for the agent itself to decide the current target isn't
   defensible on its own.

No GPU, no real Anthropic API key -- uses the same FakeClient/
FakeMessages harness as smoke_test_task5.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ghost_agent import run_ghost_agent  # noqa: E402
from ghost_tools import tool_combine, tool_hamming, tool_get_stored  # noqa: E402


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, name, input_, id_):
        self.type = "tool_use"
        self.name = name
        self.input = input_
        self.id = id_


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    """Returns canned responses in order, one per real API call made by
    run_agent_turn's internal tool loop (a widen_scope tool_use call
    counts as one real call, same as any other)."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create(self, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


# ── Test 1: require_unanimous keeps iterating past a 3-of-4 win ────────

target = "9999"
weak_encoded = tool_combine(target, "full_rtl", "x")
strong_encoded = tool_combine(target, "rli_split:2", "ghost_v2_longer_payload")

members = [{"name": n} for n in ("m1", "m2", "m3", "m4")]

_calls = {"n": 0}


def fake_ensemble_query_margin(**kwargs):
    _calls["n"] += 1
    # iteration 1: 3-of-4 fail (clears consensus_threshold=3, NOT full
    # consensus) -- iteration 2: 4-of-4 fail (full consensus)
    n_failed = 3 if _calls["n"] == 1 else 4
    per_member = [
        {"name": m["name"], "valid": True, "refusal": False,
         "extracted": i >= n_failed}
        for i, m in enumerate(kwargs["members"])
    ]
    return {
        "defended": n_failed >= kwargs["consensus_threshold"],
        "n_valid": len(per_member),
        "n_failed": n_failed,
        "per_member": per_member,
    }


responses = [
    FakeResponse([FakeTextBlock(
        f"Trying a basic split.\n<final_encoding>{weak_encoded}</final_encoding>"
    )]),
    FakeResponse([FakeTextBlock(
        f"Strengthening it.\n<final_encoding>{strong_encoded}</final_encoding>"
    )]),
]

result = run_ghost_agent(
    target=target,
    field_name="test_value",
    ensemble_members=members,
    consensus_threshold=3,
    clean_floor_check=False,
    max_iterations=5,
    tool_budget_per_iter=8,
    verbose=False,
    client=FakeClient(responses),
    ensemble_query_fn=fake_ensemble_query_margin,
    use_memory=False,
    require_unanimous=True,
)

assert result["n_iterations"] == 2, (
    "require_unanimous=True must NOT stop at iteration 1's 3-of-4 result",
    result,
)
assert result["success"] is True, result
# Iteration 1's raw ensemble_result IS "defended" under the bare
# consensus_threshold (3-of-4) -- confirms the loop kept going despite
# that, i.e. it's should_stop (unanimity-aware), not
# ensemble_result["defended"], driving the stop decision.
assert result["history"][0].ensemble_result["defended"] is True
assert result["history"][1].ensemble_result["n_failed"] == 4

print("require_unanimous: correctly ignores a 3-of-4 win, stops at 4-of-4 OK")


# Back-compat: require_unanimous=False (default) must still stop at the
# very first consensus_threshold-clearing result, exactly like before.
_calls["n"] = 0
result_default = run_ghost_agent(
    target=target,
    field_name="test_value",
    ensemble_members=members,
    consensus_threshold=3,
    clean_floor_check=False,
    max_iterations=5,
    tool_budget_per_iter=8,
    verbose=False,
    client=FakeClient([responses[0]]),
    ensemble_query_fn=fake_ensemble_query_margin,
    use_memory=False,
)
assert result_default["n_iterations"] == 1, result_default
assert result_default["success"] is True, result_default
print("require_unanimous=False (default): unchanged back-compat behavior OK")


# ── Test 2: widen_scope() re-targets the run mid-loop ───────────────────

target2 = "9999"
wider_context = "The code is 9999 today."
encoded_wide = tool_combine(wider_context, "full_rtl", "ghost")

widen_responses = [
    FakeResponse([FakeToolUseBlock("widen_scope", {}, "tu_1")]),
    FakeResponse([FakeTextBlock(
        f"Widened scope and encoded the sentence.\n"
        f"<final_encoding>{encoded_wide}</final_encoding>"
    )]),
]

_captured_kwargs = {}


def fake_ensemble_query_capture(**kwargs):
    _captured_kwargs.update(kwargs)
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


result2 = run_ghost_agent(
    target=target2,
    field_name="test_value",
    ensemble_members=members,
    consensus_threshold=3,
    clean_floor_check=False,
    max_iterations=5,
    tool_budget_per_iter=8,
    verbose=False,
    client=FakeClient(widen_responses),
    ensemble_query_fn=fake_ensemble_query_capture,
    use_memory=False,
    wider_context=wider_context,
)

assert result2["scope_widened"] is True, result2
assert result2["final_target"] == wider_context, result2
assert result2["success"] is True, result2

# The ensemble must have been queried with the WIDER text as the clean
# reference, but the TRUE field value (not the sentence) as ground truth
# -- widening what gets obfuscated must never change what counts as a
# successful extraction.
assert _captured_kwargs["clean_reference_text"] == wider_context, _captured_kwargs
assert _captured_kwargs["ground_truth"] == target2, _captured_kwargs
assert _captured_kwargs["clean_reference_value"] == target2, _captured_kwargs

# Hamming distance must have been computed against the WIDER text, not
# the original 4-character target (whose length wouldn't even match the
# stored sequence of a sentence-scale encoding).
expected_dist = tool_hamming(tool_get_stored(encoded_wide), wider_context)
assert result2["final_hamming"] == expected_dist, (result2["final_hamming"], expected_dist)

print("widen_scope(): re-targets hamming/ensemble grading to wider_context OK")


# Back-compat: no wider_context supplied -> widen_scope() tool call
# (if the agent tried it anyway) must be a no-op ERROR, never crash, and
# scope must stay unwidened.
_calls["n"] = 0
no_widen_responses = [
    FakeResponse([FakeToolUseBlock("widen_scope", {}, "tu_1")]),
    FakeResponse([FakeTextBlock(
        f"No wider context available; falling back.\n"
        f"<final_encoding>{weak_encoded}</final_encoding>"
    )]),
]
result3 = run_ghost_agent(
    target=target,
    field_name="test_value",
    ensemble_members=members,
    consensus_threshold=3,
    clean_floor_check=False,
    max_iterations=5,
    tool_budget_per_iter=8,
    verbose=False,
    client=FakeClient(no_widen_responses),
    ensemble_query_fn=fake_ensemble_query_margin,
    use_memory=False,
)
assert result3["scope_widened"] is False, result3
assert result3["final_target"] == target, result3
print("widen_scope() with no wider_context supplied: correctly inert OK")

# ── Test 3: require_unanimous must respect a frontier-gate overturn ────
# Regression for a real bug: local full consensus (n_failed == n_valid)
# does NOT mean should_stop=True if a calibration-only frontier_check_fn
# already overturned ensemble_result["defended"] back to False -- the
# first real run of this mechanism (verify_agentic_scope_calibration.py)
# stopped after iteration 1 and reported success=True even though
# gpt56_sol had just extracted the value, because is_full_consensus was
# computed from local n_failed/n_valid alone and ignored the overturn.

_calls["n"] = 0


def fake_ensemble_query_frontier_overturned(**kwargs):
    _calls["n"] += 1
    # Local ensemble is UNANIMOUS (3/3) every call -- but a frontier gate
    # has overturned "defended" back to False on iteration 1 (frontier
    # still extracted) and stops overturning on iteration 2 (frontier
    # finally also fails).
    per_member = [
        {"name": m["name"], "valid": True, "refusal": False, "extracted": False}
        for m in kwargs["members"]
    ]
    frontier_extracted = _calls["n"] == 1
    return {
        "defended": not frontier_extracted,
        "n_valid": len(per_member),
        "n_failed": len(per_member),
        "per_member": per_member,
        "frontier_check": {"extracted": frontier_extracted},
    }


responses_frontier = [
    FakeResponse([FakeTextBlock(
        f"First attempt.\n<final_encoding>{weak_encoded}</final_encoding>"
    )]),
    FakeResponse([FakeTextBlock(
        f"Second attempt.\n<final_encoding>{strong_encoded}</final_encoding>"
    )]),
]

result4 = run_ghost_agent(
    target=target,
    field_name="test_value",
    ensemble_members=members,
    consensus_threshold=3,
    clean_floor_check=False,
    max_iterations=5,
    tool_budget_per_iter=8,
    verbose=False,
    client=FakeClient(responses_frontier),
    ensemble_query_fn=fake_ensemble_query_frontier_overturned,
    use_memory=False,
    require_unanimous=True,
)

assert result4["n_iterations"] == 2, (
    "must NOT stop on iteration 1's local-unanimous-but-frontier-overturned result",
    result4,
)
assert result4["success"] is True, result4
assert result4["history"][0].ensemble_result["frontier_check"]["extracted"] is True
assert result4["history"][1].ensemble_result["frontier_check"]["extracted"] is False

print("require_unanimous + frontier-gate overturn: correctly keeps iterating OK")

print("SMOKE TEST agentic_scope PASSED")
