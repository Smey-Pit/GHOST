"""
Smoke test for the three new proxy-model-aware GHOST-Agent tools
(bidi_permute, encode_vs_logprob, combine_permute) added to
src/ghost_tools.py, plus their TOOL_DEFINITIONS/TOOL_FUNCTIONS/
extract_config_from_tool_calls wiring in src/agent_backbone.py.

No GPU, no real proxy model -- encode_vs_logprob/combine_permute are
exercised against a FAKE proxy object matching encode.ProxyModel's
.query_logprob(target_char, context) interface.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ghost_tools  # noqa: E402
from agent_backbone import (  # noqa: E402
    TOOL_DEFINITIONS, TOOL_FUNCTIONS, execute_tool_call,
    extract_config_from_tool_calls,
)


# ── 1. tool_bidi_permute (no proxy needed) ──────────────────────────────

text = "12345"
encoded = ghost_tools.tool_bidi_permute(text, trials=200)
assert encoded, "tool_bidi_permute returned empty for a trivially permutable string"
assert ghost_tools.tool_render(encoded) == text, "must render back to the original"
stored = ghost_tools.tool_get_stored(encoded)
assert len(stored) == len(text)
print(f"tool_bidi_permute: OK (stored={stored!r}, "
      f"hamming={ghost_tools.tool_hamming(stored, text)})")


# ── 2. encode_vs_logprob / combine_permute without a proxy loaded ───────

ghost_tools.clear_proxy_model()
result = execute_tool_call("encode_vs_logprob", {
    "text": "12345", "payload": "ghost", "threshold_tau": -8.0,
})
assert result.startswith("ERROR"), f"expected ERROR without a proxy, got {result!r}"
print(f"encode_vs_logprob without proxy: correctly errors ({result!r})")

result = execute_tool_call("combine_permute", {
    "text": "12345", "trials": 200, "payload": "ghost", "threshold_tau": -8.0,
})
assert result.startswith("ERROR"), f"expected ERROR without a proxy, got {result!r}"
print(f"combine_permute without proxy: correctly errors ({result!r})")


# ── 3. encode_vs_logprob / combine_permute with a fake proxy ────────────

class FakeProxy:
    """Mimics encode.ProxyModel.query_logprob -- deterministic, no GPU.

    Logprob starts high (unconfident VS did anything) and drops by a
    fixed amount per injected VS char, so encode_char_with_vs_logprob's
    stopping loop actually terminates before its 64-iteration cap for a
    reasonable threshold_tau, exercising the adaptive-depth behavior
    (different characters can still take a different number of
    iterations depending on candidate length parity below).
    """

    def query_logprob(self, target_char, context):
        n_vs = len(context) - len(target_char)
        return -1.0 - 2.0 * n_vs


ghost_tools.set_proxy_model(FakeProxy())
try:
    out = ghost_tools.tool_encode_vs_logprob("12", "ghost", -5.0)
    assert ghost_tools.tool_render(out) == "12", "must still render back to original"
    stored = ghost_tools.tool_get_stored(out)
    assert stored == "12", "get_stored must strip VS, leaving the plain characters"
    assert len(out) > len("12"), "VS characters must actually have been injected"
    print(f"tool_encode_vs_logprob with fake proxy: OK (encoded len={len(out)})")

    combined = ghost_tools.tool_combine_permute("12345", 200, "ghost", -5.0)
    assert combined, "combine_permute returned empty with a proxy loaded"
    assert ghost_tools.tool_render(combined) == "12345"
    print(f"tool_combine_permute with fake proxy: OK "
          f"(stored={ghost_tools.tool_get_stored(combined)!r})")
finally:
    ghost_tools.clear_proxy_model()

# Re-confirm clearing actually took effect (no stale reference leaking
# into a later run without a proxy).
result = execute_tool_call("encode_vs_logprob", {
    "text": "12345", "payload": "ghost", "threshold_tau": -8.0,
})
assert result.startswith("ERROR")
print("clear_proxy_model: correctly disables the tools again OK")


# ── 4. TOOL_DEFINITIONS / TOOL_FUNCTIONS wiring ─────────────────────────

names_in_definitions = {t["name"] for t in TOOL_DEFINITIONS}
for name in ("bidi_permute", "encode_vs_logprob", "combine_permute"):
    assert name in names_in_definitions, f"{name} missing from TOOL_DEFINITIONS"
    assert name in TOOL_FUNCTIONS, f"{name} missing from TOOL_FUNCTIONS"
print("TOOL_DEFINITIONS/TOOL_FUNCTIONS: all three new tools registered OK")


# ── 5. extract_config_from_tool_calls recognizes the new tools ─────────

bidi_config, vs_payload = extract_config_from_tool_calls([
    ("bidi_permute", {"text": "12345", "trials": 777}),
])
assert bidi_config == "bidi_permute(trials=777,seed=0)"
assert vs_payload == "unknown"

bidi_config, vs_payload = extract_config_from_tool_calls([
    ("combine_permute", {"text": "12345", "trials": 500, "payload": "xyz", "threshold_tau": -8.0}),
])
assert bidi_config == "bidi_permute(trials=500,seed=0)"
assert vs_payload == "xyz"

bidi_config, vs_payload = extract_config_from_tool_calls([
    ("bidi_permute", {"text": "12345", "trials": 100}),
    ("encode_vs_logprob", {"text": "whatever", "payload": "abc", "threshold_tau": -8.0}),
])
assert bidi_config == "bidi_permute(trials=100,seed=0)"
assert vs_payload == "abc"
print("extract_config_from_tool_calls: recognizes bidi_permute/encode_vs_logprob/combine_permute OK")

print("SMOKE TEST ghost_tools_permute PASSED")
