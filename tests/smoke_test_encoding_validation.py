"""
tests/smoke_test_encoding_validation.py

Regression tests for three real bugs found 2026-08-19 during a real
10-example sentence-scope run (see CLAUDE.md's "Real sentence-scope
prior-strength run" section):

1. _run_ghost_agent_loop never checked that a proposed encoding
   actually renders back to the FULL target -- a length-mismatched (or,
   separately, a same-length-but-content-corrupted) encoding used to be
   silently accepted with a nonsensical hamming clamped to 0.
2. Even a length/content-matching encoding could be field-scope wearing
   sentence-scope's clothing: the agent copies the whole surrounding
   sentence as exact plaintext and only obfuscates the field itself.
   min_sentence_scope_hamming_fraction now rejects this.
3. distil_principle's output was never validated -- a refusal string
   could be persisted into strategy_memory.json as if it were a real
   principle. _is_valid_principle now filters this before
   strategy_memory.add_principle is ever called.

No GPU, no real Anthropic API key -- same FakeClient/FakeMessages
harness as smoke_test_task5.py / smoke_test_agentic_scope.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ghost_agent import run_ghost_agent, _is_valid_principle  # noqa: E402
from ghost_tools import tool_combine  # noqa: E402


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


def always_defended_ensemble_query(**kwargs):
    members = kwargs.get("members", [{"name": "m1"}])
    per_member = [
        {"name": m["name"], "valid": True, "refusal": False, "extracted": False}
        for m in members
    ]
    return {
        "defended": True,
        "n_valid": len(per_member),
        "n_failed": len(per_member),
        "per_member": per_member,
    }


MEMBERS = [{"name": "m1", "hf_id": "fake/model"}]


# ── Test 1: length-mismatched encoding (bare field, not full sentence)
# is rejected and retried, not silently accepted with hamming=0 ────────

sentence = "Company Type: PTE LTD"
field_value = "PTE LTD"

field_only_encoding = tool_combine(field_value, "full_rtl", "x")  # renders as just "PTE LTD"
full_sentence_encoding = tool_combine(sentence, "full_rtl", "y")  # renders as the whole sentence

texts_bad_then_good = [
    f"Encoding just the field.\n<final_encoding>{field_only_encoding}</final_encoding>",
    f"Encoding the full sentence this time.\n<final_encoding>{full_sentence_encoding}</final_encoding>",
]

result1 = run_ghost_agent(
    target=sentence,
    field_name="company_type",
    field_ground_truth=field_value,
    ensemble_members=MEMBERS,
    consensus_threshold=1,
    clean_floor_check=False,
    max_iterations=5,
    verbose=False,
    client=FakeClient(texts_bad_then_good),
    ensemble_query_fn=always_defended_ensemble_query,
    use_memory=False,
)

assert result1["success"] is True, result1
assert result1["n_iterations"] == 1, (
    f"the length-mismatched attempt must be rejected and NOT counted as "
    f"an iteration -- expected 1 (only the valid second attempt), got "
    f"{result1['n_iterations']}"
)
assert result1["final_hamming"] > 0, (
    "must not be the old bogus hamming=0 result from an accepted "
    "length-mismatched attempt"
)
assert result1["best_encoding"] == full_sentence_encoding, result1
print("Test 1 passed: length-mismatched (field-only) encoding rejected and retried")


# ── Test 2: same-length, content-matching encoding that only obfuscates
# the field and copies the rest of the sentence verbatim is rejected
# (min_sentence_scope_hamming_fraction), a genuine full attempt is not ──

sentence2 = (
    "Device Configuration Summary. This unit was inspected and logged "
    "under standard procedure. Serial Number: XQW-78921-UV"
)
field_value2 = "XQW-78921-UV"
prefix = sentence2[: sentence2.index(field_value2)]

encoded_field_only = tool_combine(field_value2, "full_rtl", "z")
verbatim_prefix_encoding = prefix + encoded_field_only  # renders back to the FULL sentence
full_sentence_encoding2 = tool_combine(sentence2, "full_rtl", "w")

texts_weak_then_strong = [
    f"Obfuscating the field.\n<final_encoding>{verbatim_prefix_encoding}</final_encoding>",
    f"Obfuscating the whole sentence.\n<final_encoding>{full_sentence_encoding2}</final_encoding>",
]

result2 = run_ghost_agent(
    target=sentence2,
    field_name="serial_number",
    field_ground_truth=field_value2,
    ensemble_members=MEMBERS,
    consensus_threshold=1,
    clean_floor_check=False,
    max_iterations=5,
    verbose=False,
    client=FakeClient(texts_weak_then_strong),
    ensemble_query_fn=always_defended_ensemble_query,
    use_memory=False,
)

assert result2["success"] is True, result2
assert result2["n_iterations"] == 1, (
    f"the verbatim-prefix attempt must be rejected (below "
    f"min_sentence_scope_hamming_fraction) and NOT counted -- expected 1 "
    f"(only the genuine second attempt), got {result2['n_iterations']}"
)
assert result2["best_encoding"] == full_sentence_encoding2, result2
weak_hamming_fraction = len(field_value2) / len(sentence2)  # what the rejected attempt would have scored
assert result2["final_hamming"] / len(sentence2) > weak_hamming_fraction, (
    "accepted attempt's hamming fraction should be well above the "
    "rejected weak attempt's"
)
print("Test 2 passed: verbatim-prefix-plus-field-only encoding rejected and retried")


# ── Test 3: _is_valid_principle rejects refusals and degenerate text,
# accepts a real-looking principle ──────────────────────────────────────

assert _is_valid_principle(
    "For short numeric strings, apply full RTL reversal combined with "
    "variation selector injection between digits."
) is True
assert _is_valid_principle("I can't help with this request.") is False
assert _is_valid_principle(
    "I appreciate the detailed scenario, but I need to be direct: "
    "I can't help with this."
) is False
assert _is_valid_principle("") is False
assert _is_valid_principle(None) is False
assert _is_valid_principle("too short") is False  # under _MIN_PRINCIPLE_LENGTH... actually check length
print("Test 3 passed: _is_valid_principle rejects refusals/degenerate text, accepts real principles")

print("SMOKE TEST encoding_validation PASSED")
