"""
smoke_test_task5.py

Verifies the ghost_agent.py orchestration loop (tool wiring, encoding
extraction, ensemble-consensus stopping condition, Attempt recording)
WITHOUT a real Anthropic API key or any GPU / model weights, by
injecting a fake Anthropic-SDK-shaped client and a fake
ensemble_query_fn. Real agent reasoning quality and real ensemble
behavior still need actual hardware/API access later -- this only
proves the loop's control flow is correct.
"""

import yaml

from ghost_agent import run_ghost_agent, extract_proposed_encoding
from ghost_tools import tool_combine
from ensemble import load_ensemble_config

with open("../config.yaml") as f:
    config = yaml.safe_load(f)
resolved = load_ensemble_config(config)


# ── extract_proposed_encoding: delimiter parsing ─────────────────────────

target = "12345"
encoded = tool_combine(target, "full_rtl", "x")

text_with_tag = f"Reasoning here.\n<final_encoding>{encoded}</final_encoding>\nDone."
extracted = extract_proposed_encoding(text_with_tag, target)
assert extracted == encoded, (extracted, encoded)

text_without_tag = "I think a simple reversal should work well here."
assert extract_proposed_encoding(text_without_tag, target) is None

# The whole point of the fix: a genuinely obfuscated encoding contains
# NONE of the target's plaintext digits, and extraction must still work.
assert target[:3] not in encoded, "test fixture should be genuinely obfuscated"
assert extracted is not None

# Empty tag should be treated as missing, not an empty-string "encoding"
text_empty_tag = "<final_encoding></final_encoding>"
assert extract_proposed_encoding(text_empty_tag, target) is None

print("extract_proposed_encoding: delimiter parsing OK")


# ── Fake Anthropic client (no tool_use blocks, plain text responses) ────

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


weak_encoded = tool_combine(target, "full_rtl", "x")
strong_encoded = tool_combine(target, "rli_split:2", "ghost_v2_longer_payload")

texts = [
    f"I'll try a simple reversal first.\n<final_encoding>{weak_encoded}</final_encoding>",
    f"Let's try a stronger split with more VS injection.\n<final_encoding>{strong_encoded}</final_encoding>",
]


# ── Fake ensemble: not defended on call 1, defended on call 2 ───────────

_call_count = {"n": 0}


def fake_ensemble_query(**kwargs):
    _call_count["n"] += 1
    extracted = _call_count["n"] == 1  # iteration 1: all extract correctly
    per_member = [
        {"name": m["name"], "valid": True, "refusal": False, "extracted": extracted}
        for m in kwargs["members"]
    ]
    n_failed = sum(1 for m in per_member if not m["extracted"])
    return {
        "defended": n_failed >= kwargs["consensus_threshold"],
        "n_valid": len(per_member),
        "n_failed": n_failed,
        "per_member": per_member,
    }


result = run_ghost_agent(
    target=target,
    field_name="test_value",
    ensemble_members=resolved["members"],
    consensus_threshold=resolved["consensus_threshold"],
    clean_floor_check=resolved["clean_floor_check"],
    max_iterations=5,
    tool_budget_per_iter=8,
    verbose=True,
    client=FakeClient(texts),
    ensemble_query_fn=fake_ensemble_query,
    use_memory=False,  # FakeClient is pre-scripted for exactly 2 calls;
                        # memory would add a 3rd (distillation) and touch
                        # real results/ files -- see ghost_agent.py's
                        # run_ghost_agent use_memory docstring
)

assert result["success"] is True, result
assert result["n_iterations"] == 2, result  # stopped early on iteration 2
assert result["best_encoding"] == strong_encoded, result
assert len(result["history"]) == 2
assert result["history"][0].ensemble_result["defended"] is False
assert result["history"][1].ensemble_result["defended"] is True
print(f"\nResult: success={result['success']}, iterations={result['n_iterations']}")
print("TASK 5 SMOKE TEST PASSED")


# ── Regression: multiple tool_use rounds within a single agent turn ─────
# The original run_agent_turn hardcoded exactly one continuation call
# after the first tool_use round. If the model called tools again in
# that continuation, those tool_use blocks were appended with no
# matching tool_result -- invisible until the NEXT api call, which the
# real Anthropic API rejects with a 400 ("tool_use ids were found
# without tool_result blocks"). This fake client reproduces that shape:
# round 1 tool_use, round 2 ALSO tool_use, round 3 finally plain text.

from ghost_agent import run_agent_turn, TOOL_DEFINITIONS  # noqa: E402


class FakeToolUseBlock:
    def __init__(self, name, input_, id_):
        self.type = "tool_use"
        self.name = name
        self.input = input_
        self.id = id_


class MultiRoundFakeMessages:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create(self, **kwargs):
        # every tool_use block must be answered by a tool_result block
        # in the IMMEDIATELY FOLLOWING message, mirroring the real
        # Anthropic API's requirement.
        msgs = kwargs["messages"]
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            content = msg["content"]
            if not isinstance(content, list):
                continue
            tool_use_ids = {
                b.id for b in content if getattr(b, "type", None) == "tool_use"
            }
            if not tool_use_ids:
                continue
            if i + 1 >= len(msgs):
                raise AssertionError(
                    "API would reject this call: unanswered tool_use "
                    "block with no following message"
                )
            next_content = msgs[i + 1]["content"]
            result_ids = {
                b["tool_use_id"] for b in next_content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            }
            if not tool_use_ids <= result_ids:
                raise AssertionError(
                    "API would reject this call: unanswered tool_use "
                    "block in prior assistant message"
                )
        response = self.responses[self.calls]
        self.calls += 1
        return response


class MultiRoundFakeClient:
    def __init__(self, responses):
        self.messages = MultiRoundFakeMessages(responses)


multi_round_responses = [
    FakeResponse([FakeToolUseBlock("render", {"encoding": weak_encoded}, "tu_1")]),
    FakeResponse([FakeToolUseBlock("hamming", {"s1": "a", "s2": "b"}, "tu_2")]),
    FakeResponse([FakeTextBlock(
        f"Done.\n<final_encoding>{strong_encoded}</final_encoding>"
    )]),
]

messages = []
messages, final_text = run_agent_turn(
    MultiRoundFakeClient(multi_round_responses),
    messages,
    "system prompt",
    max_tool_rounds=8,
)
assert "<final_encoding>" in final_text, final_text
assert extract_proposed_encoding(final_text, target) == strong_encoded

# Every tool_use block anywhere in messages must be immediately followed
# by a matching tool_result in the next message.
for i, msg in enumerate(messages):
    content = msg["content"]
    if not isinstance(content, list):
        continue
    tool_use_ids = {b.id for b in content if getattr(b, "type", None) == "tool_use"}
    if not tool_use_ids:
        continue
    next_msg = messages[i + 1]
    result_ids = {
        b["tool_use_id"] for b in next_msg["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    assert tool_use_ids <= result_ids, (tool_use_ids, result_ids)

print("Multi-round tool_use regression: no dangling tool_use blocks OK")
print("TASK 5 SMOKE TEST PASSED")
