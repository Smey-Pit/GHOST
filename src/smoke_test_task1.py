"""
smoke_test_task1.py

Smoke test for src/ghost_tools.py (GHOST_AGENT_TASKS.md, Task 1).
Run from src/: python smoke_test_task1.py
"""
from ghost_tools import (
    tool_render, tool_get_stored, tool_hamming,
    tool_encode_vs, tool_encode_bidi, tool_combine,
)

target = "1068348"

# Test 1: full_rtl renders as target
enc = tool_encode_bidi(target, "full_rtl")
assert tool_render(enc) == target, \
    f"full_rtl failed: {tool_render(enc)} != {target}"
stored = tool_get_stored(enc)
assert stored == target[::-1], \
    f"stored wrong: {stored} != {target[::-1]}"
dist = tool_hamming(stored, target)
print(f"full_rtl: stored={stored} dist={dist}")
assert dist > 0

# Test 2: rli_split renders correctly
enc2 = tool_encode_bidi(target, "rli_split:3")
rendered2 = tool_render(enc2)
print(f"rli_split:3 renders as: '{rendered2}'")
assert len(rendered2) > 0

# Test 3: VS injection is invisible to render
text = "hello"
vs_enc = tool_encode_vs(text, "test_payload")
assert len(vs_enc) > len(text), "VS chars not added"
readable = ''.join(c for c in vs_enc
                    if ord(c) < 0xFE00 or ord(c) > 0xE01FF)
assert readable == text, f"VS changed visible text: {readable}"

# Test 4: combine produces non-empty result
combined = tool_combine(target, "full_rtl", "ghost")
assert len(combined) > 0
assert tool_render(combined) == target, \
    f"combine render failed: {tool_render(combined)}"

# Test 5: hamming distance
assert tool_hamming("12345", "12345") == 0
assert tool_hamming("12345", "54321") == 4
assert tool_hamming("12345", "123") == -1  # different length

# Test 6: entropy fix regression.
# A repeating character with a short payload must NOT produce a
# periodic VS pattern (the bug the entropy fix removes: cycling a
# short literal payload by position gives every character at the
# same offset mod len(payload) an identical VS suffix).
repeated = "a" * 12
short_payload = "ab"  # length 2 -> old cycling period would be 2
vs_repeated = tool_encode_vs(repeated, short_payload)
vs_suffixes = []
current = []
for c in vs_repeated:
    if ord(c) < 0xFE00:
        if current:
            vs_suffixes.append(tuple(current))
        current = []
    else:
        current.append(c)
if current:
    vs_suffixes.append(tuple(current))
# Under the old (buggy) design, suffixes would alternate between
# exactly 2 distinct values with period 2. Confirm we see more
# variety than that across 12 characters.
distinct_suffixes = set(vs_suffixes)
print(f"VS suffixes for repeated char: {len(distinct_suffixes)} distinct "
      f"out of {len(vs_suffixes)} characters (payload length 2)")
assert len(distinct_suffixes) > 2, \
    "VS injection still periodic in payload length -- entropy fix not applied"

# Test 7: longer payload injects more VS chars per character (strength).
short = tool_encode_vs("x", "a")
long = tool_encode_vs("x", "abcdefghij")
n_vs_short = sum(1 for c in short if ord(c) >= 0xFE00)
n_vs_long = sum(1 for c in long if ord(c) >= 0xFE00)
assert n_vs_long > n_vs_short, \
    f"longer payload should inject more VS chars: {n_vs_long} <= {n_vs_short}"

print("TASK 1 SMOKE TEST PASSED")
