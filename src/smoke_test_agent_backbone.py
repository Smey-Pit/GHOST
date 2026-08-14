"""
smoke_test_agent_backbone.py

Verifies agent_backbone.py's LocalReActBackbone tool-call round-trip
(text-protocol parsing, execute_tool_call dispatch, reset() clearing
conversation state) WITHOUT loading any real GPU weights, by monkey-
patching LocalReActBackbone._generate to a scripted sequence of
responses -- mirrors smoke_test_task5.py's FakeClient approach for the
Anthropic path.
"""

from agent_backbone import LocalReActBackbone, _parse_tool_calls, resolve_backbone


# ── _parse_tool_calls: text-protocol parsing ─────────────────────────────

text = (
    'Let me check.\n'
    '<tool_call>{"name": "hamming", "input": {"s1": "12345", "s2": "54321"}}</tool_call>\n'
    'and also\n'
    '<tool_call>{"name": "render", "input": {"encoding": "abc"}}</tool_call>'
)
calls = _parse_tool_calls(text)
assert calls == [
    ("hamming", {"s1": "12345", "s2": "54321"}),
    ("render", {"encoding": "abc"}),
], calls

# Malformed JSON in a tag must be skipped, not raise.
bad_text = '<tool_call>{not valid json}</tool_call>'
assert _parse_tool_calls(bad_text) == []

# No tags -> no calls (this is the "final answer" signal).
assert _parse_tool_calls("just a final answer, no tools") == []

# Real deepseek_r1_32b output (diagnose_local_backbone.py) used markdown
# ```json fences instead of the instructed <tool_call> tag -- must be
# recognised too, or a model that reliably prefers this format never
# gets its tool calls parsed at all (confirmed: this exact failure
# silently turned a genuine multi-tool-call response into a single-turn
# "final answer" on a real run).
fence_text = (
    '1. Verify rendering:\n'
    '```json\n{"name": "render", "input": {"encoding": "QLD-17"}}\n```\n'
    '2. Check hamming:\n'
    '```json\n{"name": "hamming", "input": {"s1": "a", "s2": "b"}}\n```'
)
assert _parse_tool_calls(fence_text) == [
    ("render", {"encoding": "QLD-17"}),
    ("hamming", {"s1": "a", "s2": "b"}),
], _parse_tool_calls(fence_text)

# Mixed tag + fence in the same response, in appearance order.
mixed_text = (
    '<tool_call>{"name": "render", "input": {"encoding": "x"}}</tool_call>\n'
    'then\n'
    '```json\n{"name": "hamming", "input": {"s1": "a", "s2": "b"}}\n```'
)
assert _parse_tool_calls(mixed_text) == [
    ("render", {"encoding": "x"}),
    ("hamming", {"s1": "a", "s2": "b"}),
], _parse_tool_calls(mixed_text)

# A ```json fence with no "name" key (e.g. an unrelated example block in
# the model's prose) must NOT be misparsed as a real tool call.
non_call_fence = '```json\n{"foo": "bar"}\n```'
assert _parse_tool_calls(non_call_fence) == []

print("_parse_tool_calls: OK")


# ── LocalReActBackbone.run_turn: scripted multi-round tool loop ─────────

class FakeLocalBackbone(LocalReActBackbone):
    """Skips real model loading; _generate returns a scripted sequence."""

    def __init__(self, responses):
        self.tokenizer = None
        self.model = None
        self.max_new_tokens = 100
        self.strip_think = False
        self.messages = []
        self._responses = responses
        self._calls = 0

    def _generate(self, messages, max_new_tokens=None):
        text = self._responses[self._calls]
        self._calls += 1
        return text


responses = [
    'Trying full_rtl.\n<tool_call>{"name": "hamming", "input": {"s1": "a", "s2": "b"}}</tool_call>',
    'Checking again.\n<tool_call>{"name": "render", "input": {"encoding": "12345"}}</tool_call>',
    'Done.\n<final_encoding>SOME_ENCODING</final_encoding>',
]
backbone = FakeLocalBackbone(responses)
final_text = backbone.run_turn(
    user_msg="encode 12345",
    system_prompt="system prompt",
    tool_budget_per_iter=8,
    response_max_tokens=100,
)
assert "<final_encoding>SOME_ENCODING</final_encoding>" in final_text, final_text
assert backbone._calls == 3, backbone._calls

# First message must be the seeded system+tool-protocol message.
assert backbone.messages[0]["role"] == "system"
assert "hamming(s1, s2)" in backbone.messages[0]["content"]

# Tool results were fed back as user turns between assistant turns.
roles = [m["role"] for m in backbone.messages]
assert roles == [
    "system", "user", "assistant", "user", "assistant", "user", "assistant",
], roles
assert "Tool `hamming` result" in backbone.messages[3]["content"]
assert "Tool `render` result" in backbone.messages[5]["content"]

print("LocalReActBackbone.run_turn: multi-round tool loop OK")

# reset() clears conversation but not the (here, fake) loaded weights.
backbone.reset()
assert backbone.messages == []
print("LocalReActBackbone.reset: OK")


# ── generate_text: standalone call for principle distillation ──────────

class FakeGenTextBackbone(FakeLocalBackbone):
    def _generate(self, messages, max_new_tokens=None):
        assert messages == [{"role": "user", "content": "distil this"}]
        return "the distilled principle"


gen_backbone = FakeGenTextBackbone([])
result = gen_backbone.generate_text("distil this", max_tokens=150)
assert result == "the distilled principle", result
print("LocalReActBackbone.generate_text: OK")


# ── resolve_backbone: config-driven construction (provider dispatch only) ─

config = {
    "agent_backbones": {
        "default": "claude_sonnet",
        "options": {
            "claude_sonnet": {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-6",
            },
        },
    },
}


class FakeAnthropicClient:
    pass


backbone = resolve_backbone("claude_sonnet", config, client=FakeAnthropicClient())
assert backbone.model_id == "claude-sonnet-4-6"
assert isinstance(backbone.client, FakeAnthropicClient)
print("resolve_backbone (anthropic provider): OK")

try:
    resolve_backbone("nonexistent", config)
    raise AssertionError("expected ValueError for unknown backbone name")
except ValueError:
    pass
print("resolve_backbone: unknown name raises OK")

print("SMOKE TEST agent_backbone.py PASSED")
