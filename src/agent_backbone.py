"""
src/agent_backbone.py

Configurable agent backbone for GHOST-Agent (extends Task 5 of
GHOST_AGENT_TASKS.md). The agent role was originally hardcoded to
claude-sonnet-4-6 (CRITICAL RULE 5 in the task doc) -- this module
replaces that hardcode with a swappable Backbone so the paper can run
backbone ablations (Claude Sonnet/Haiku/Opus vs. local DeepSeek-R1
distills) without changing ghost_agent.py's orchestration loop.

Two implementations, one interface (reset/run_turn/generate_text/close):

  AnthropicBackbone   -- wraps the real Anthropic tool-use API (native
                         tool_use/tool_result content blocks). This is
                         a thin wrapper around run_agent_turn below,
                         which is unchanged from the original
                         ghost_agent.py implementation (still exported
                         here for the existing smoke test's direct
                         import).
  LocalReActBackbone  -- local HF causal LMs (e.g. DeepSeek-R1 distills)
                         have no Anthropic-style structured tool use, so
                         tool calls are requested via a text protocol
                         instead: the model emits
                         <tool_call>{"name": ..., "input": {...}}</tool_call>
                         and this backbone regexes/json-parses it,
                         executes the tool, and feeds the result back as
                         a plain chat turn. Same TOOL_FUNCTIONS dispatch
                         (execute_tool_call) as the Anthropic path --
                         only the calling *protocol* differs, not the
                         tools themselves.

Both backbones expose the SAME four methods so ghost_agent.py's
run_ghost_agent() never needs to know which one it's holding:
  reset()                 -- clear conversation state (NOT the loaded
                              weights) between fields/documents. Local
                              models are loaded ONCE and reused across
                              many run_ghost_agent() calls in a
                              convergence run -- reloading per field
                              would reintroduce the same load-time-
                              dominates-cost problem CLAUDE.md already
                              documents for the search-tier ensemble.
  run_turn(user_msg, system_prompt, tool_budget_per_iter,
           response_max_tokens) -> str
                          -- one agent iteration: send user_msg, run
                             the tool-call round-trip loop, return the
                             final text response.
  generate_text(prompt, max_tokens) -> str
                          -- one-off, history-free call. Used by
                             ghost_agent.py's distil_principle so
                             principle distillation goes through
                             whichever backbone ran the field, not a
                             separate hardcoded model.
  close()                 -- free GPU memory (no-op for API backbones).

resolve_backbone(name, config) reads config.yaml's new agent_backbones
section and constructs the right one.
"""

import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from ghost_tools import TOOL_FUNCTIONS  # noqa: E402
from unicode_utils import strip_think_tags  # noqa: E402


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ── Anthropic tool definitions (moved from ghost_agent.py, unchanged) ───

TOOL_DEFINITIONS = [
    {
        "name": "render",
        "description": (
            "Render a bidi-encoded string to see what a human "
            "reader sees. Always call this to verify your "
            "encoding before reporting it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "encoding": {
                    "type": "string",
                    "description": "The encoded string to render",
                },
            },
            "required": ["encoding"],
        },
    },
    {
        "name": "get_stored",
        "description": (
            "Get the stored character sequence from an encoded "
            "string. This is what an LLM tokenizer ingests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "encoding": {
                    "type": "string",
                    "description": "The encoded string",
                },
            },
            "required": ["encoding"],
        },
    },
    {
        "name": "hamming",
        "description": (
            "Compute Hamming distance between two strings. "
            "Higher = more different = better obfuscation. "
            "Use this to evaluate your encoding quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "s1": {
                    "type": "string",
                    "description": "First string (stored sequence)",
                },
                "s2": {
                    "type": "string",
                    "description": "Second string (original target)",
                },
            },
            "required": ["s1", "s2"],
        },
    },
    {
        "name": "encode_bidi",
        "description": (
            "Apply a bidi encoding configuration to text. "
            "Available configs: full_rtl, rli_split:N, "
            "rli_split_v2:N, three_block:N:M, nested:N:M, "
            "double_rli:N"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to encode",
                },
                "config": {
                    "type": "string",
                    "description": (
                        "Config string e.g. 'full_rtl' or "
                        "'rli_split:3' or 'three_block:2:5'"
                    ),
                },
            },
            "required": ["text", "config"],
        },
    },
    {
        "name": "encode_vs",
        "description": (
            "Inject invisible Variation Selector characters "
            "into text. Fragments LLM tokenization without "
            "affecting human readability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to inject VS into",
                },
                "payload": {
                    "type": "string",
                    "description": (
                        "Payload string determining VS codepoints. "
                        "Try 'ghost', 'ghost_v2', 'aaaaaa', etc."
                    ),
                },
            },
            "required": ["text", "payload"],
        },
    },
    {
        "name": "combine",
        "description": (
            "Apply BOTH bidi encoding and VS injection. "
            "This is the strongest obfuscation -- use this "
            "as your primary tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to encode",
                },
                "bidi_config": {
                    "type": "string",
                    "description": "Bidi configuration string",
                },
                "vs_payload": {
                    "type": "string",
                    "description": "VS payload string",
                },
            },
            "required": ["text", "bidi_config", "vs_payload"],
        },
    },
]


def execute_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a single tool call and return string result."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return f"ERROR: unknown tool {tool_name}"
    try:
        result = fn(**tool_input)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


# ── Anthropic backbone (native tool_use) ────────────────────────────────

def run_agent_turn(
    client, messages: list, system_prompt: str, max_tool_rounds: int = 8,
    response_max_tokens: int = 2000, model_id: str = DEFAULT_ANTHROPIC_MODEL,
) -> tuple:
    """
    Run one agent turn against a real (or fake, for tests) Anthropic-SDK-
    shaped client: call the model with tools, execute all tool calls,
    feed back the results, and repeat until the model stops calling
    tools (stop_reason != "tool_use") or max_tool_rounds is hit.

    Unchanged from the original ghost_agent.py implementation except
    `model_id` is now a parameter instead of a hardcoded module constant
    -- this is what lets AnthropicBackbone below select Sonnet/Haiku/Opus
    per config instead of always claude-sonnet-4-6.

    Every `tool_use` block MUST be answered by a `tool_result` block in
    the very next message -- the Anthropic API rejects a subsequent call
    otherwise ("tool_use ids were found without tool_result blocks"). A
    one-continuation-round version of this function previously left a
    second round of tool_use blocks unanswered whenever the model chose
    to call tools again after its first round; that corrupted `messages`
    for the rest of the run, only surfacing as a 400 on the NEXT api
    call (confirmed against the real API). Looping until no tool_use
    blocks remain (capped by max_tool_rounds) fixes this.

    Returns:
        (updated_messages, agent_final_text)
    """
    final_text = ""

    for _ in range(max_tool_rounds):
        response = client.messages.create(
            model=model_id,
            max_tokens=response_max_tokens,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text":
                final_text = block.text
            elif block.type == "tool_use":
                result = execute_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})

    return messages, final_text


class AnthropicBackbone:
    """
    Agent backbone over a real Anthropic model (Sonnet/Haiku/Opus/etc),
    using native tool_use. `model_id` and `client` are both injectable
    so tests can supply a fake client without a real API key, and so
    config.yaml's agent_backbones.options can select any Anthropic model
    string without code changes.
    """

    def __init__(self, model_id: str = DEFAULT_ANTHROPIC_MODEL, client=None):
        self.model_id = model_id
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.messages = []

    def reset(self) -> None:
        """Clear conversation state between fields. No GPU weights to keep warm."""
        self.messages = []

    def run_turn(
        self, user_msg: str, system_prompt: str,
        tool_budget_per_iter: int, response_max_tokens: int,
    ) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        self.messages, final_text = run_agent_turn(
            self.client, self.messages, system_prompt,
            max_tool_rounds=tool_budget_per_iter,
            response_max_tokens=response_max_tokens,
            model_id=self.model_id,
        )
        return final_text

    def generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """One-off, history-free call -- used for principle distillation."""
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def close(self) -> None:
        pass  # no GPU resource to free

    def release_gpu(self) -> None:
        pass  # no GPU resource to free

    def reacquire_gpu(self) -> None:
        pass


# ── Local backbone (text-protocol tool calling) ─────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Real deepseek_r1_32b output (diagnose_local_backbone.py, iteration 2)
# used markdown ```json fences instead of the instructed <tool_call> tag
# even though the protocol text explicitly specifies the tag -- a real,
# observed model preference (```json is an extremely common training
# pattern; the custom tag is not), not a hypothetical to guard against.
# Without this second pattern, _parse_tool_calls returned zero matches on
# a response that was clearly attempting three tool calls, so run_turn
# treated it as a finished final answer after a single turn.
_TOOL_CALL_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_tool_calls(text: str) -> list:
    """
    Parse tool-call requests out of a local model's raw text response.
    Two accepted forms, in the order they appear in the text:
      <tool_call>{"name": ..., "input": {...}}</tool_call>   (instructed)
      ```json\n{"name": ..., "input": {...}}\n```            (observed
                                                                fallback)
    A JSON object without a "name" key (KeyError below) is silently NOT
    treated as a tool call -- this is what keeps an unrelated ```json
    example block in a model's prose from being misparsed as a real
    call. Malformed JSON in either form is skipped rather than raising --
    a local model emitting broken JSON is a real failure mode to expect,
    not something that should crash the whole agent loop; the model just
    gets no tool result for that call and can try again next round.
    """
    matches = [
        (m.start(), m.group(1))
        for pattern in (_TOOL_CALL_RE, _TOOL_CALL_FENCE_RE)
        for m in pattern.finditer(text)
    ]
    matches.sort(key=lambda pair: pair[0])

    calls = []
    for _, raw_json in matches:
        try:
            payload = json.loads(raw_json)
            calls.append((payload["name"], payload.get("input", {})))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return calls


def _build_tool_protocol_text() -> str:
    """
    Text-protocol description of TOOL_DEFINITIONS for models without
    native structured tool use. Built programmatically from
    TOOL_DEFINITIONS so the two calling conventions (Anthropic native
    vs. this text protocol) can never drift out of sync on which tools
    exist or what arguments they take.
    """
    lines = [
        "You have access to the following tools. To call a tool, output "
        "a line in exactly this format (one tool call per line; you may "
        "call several across multiple turns):",
        '  <tool_call>{"name": "TOOL_NAME", "input": {"arg": "value"}}</tool_call>',
        "",
        "When you are finished and ready to give your final answer, do "
        "NOT emit any <tool_call> tag in that response -- just write "
        "your final answer (with the <final_encoding> tag as instructed "
        "above). Any <tool_call> tag in a response means you are still "
        "working, not reporting a final answer.",
        "",
        "Available tools:",
    ]
    for tool in TOOL_DEFINITIONS:
        args = ", ".join(tool["input_schema"]["properties"].keys())
        lines.append(f"- {tool['name']}({args}): {tool['description'].strip()}")
    return "\n".join(lines)


_TOOL_PROTOCOL_TEXT = _build_tool_protocol_text()


def _load_backbone_model(hf_id: str, dtype: str = "bfloat16", quantization: Optional[str] = None):
    """
    Load a HF causal LM + tokenizer for use as the agent backbone.

    Deliberately separate from ensemble.py's load_local_model (used for
    search-tier ensemble members) rather than reused with a new param --
    quantization is an agent-backbone-only concern (see
    LocalReActBackbone's docstring on why 8-bit was adopted: it fixes
    both a GPU VRAM OOM and a host-RAM staging-spike SIGKILL hit on a
    real run) and must never leak into the ensemble's own loading path,
    which stays plain bf16/dtype-as-configured.

    quantization=None: identical to ensemble.py's load_local_model.
    quantization="8bit": bitsandbytes BitsAndBytesConfig(load_in_8bit=True)
        -- ~half the VRAM AND host-RAM-staging footprint of bf16.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    if quantization == "8bit":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
    elif quantization is None:
        torch_dtype = getattr(torch, dtype)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map="auto", torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(f"Unknown quantization mode: {quantization!r}")

    return tokenizer, model


class LocalReActBackbone:
    """
    Agent backbone over an already-loaded-once local HF causal LM (e.g.
    a DeepSeek-R1 distill). Loaded ONCE per backbone instance and reused
    across every field/document the caller runs through it -- reset()
    clears only the conversation, never the weights. Caller MUST call
    close() when finished with the whole run to free GPU memory; unlike
    ensemble.py's per-query load/unload (which amortizes cost across
    many SHORT queries), the agent role needs the model resident across
    an entire multi-iteration, multi-field convergence run or reloading
    would reintroduce the same load-time-dominates-cost problem CLAUDE.md
    already flags for the search-tier ensemble.

    Tool calls use the text protocol above (no native tool_use for local
    models) -- see _parse_tool_calls/_build_tool_protocol_text.

    strip_think: strip <think>...</think> before parsing tool calls AND
    before returning final text, same convention as check_extraction /
    query_adversary_local for reasoning-distill models (a stray
    <tool_call>-shaped string inside a hidden reasoning block must not
    be parsed as a real tool call).

    release_gpu()/reacquire_gpu(): a large UNQUANTIZED backbone (e.g. the
    32B distill at bf16, ~64GB) staying resident for the whole run WILL
    collide with the search-tier ensemble's own per-member GPU residency
    during the ensemble_query_fn call each iteration -- confirmed for
    real, not theoretical: a first live run of this exact combination hit
    `torch.OutOfMemoryError` on iteration 2 (backbone + one ensemble
    member briefly both resident exceeded the 80GB card). ghost_agent.py's
    loop calls release_gpu() immediately before ensemble_query_fn and
    reacquire_gpu() immediately after, for every backbone -- a no-op for
    AnthropicBackbone, and for an UNQUANTIZED local backbone a plain
    `.to("cpu")`/`.to(the original device)` round-trip. This keeps the
    "loaded once" property (no re-download, no re-init) while never
    holding both the backbone and an ensemble member on GPU at once.

    quantization="8bit": these are ALSO no-ops here, for two reasons, not
    one -- both confirmed on real runs, not hypothetical:
      1. A quantized backbone's footprint (~32-34GB for the 32B distill)
         already coexists with an ensemble member's peak (~28GB for
         deepseek_r1_14b) inside 80GB VRAM without needing to release
         anything.
      2. bitsandbytes' Linear8bitLt layers hold GPU-specific quantization
         state (scales/CB tensors) that is NOT guaranteed to round-trip
         safely through an arbitrary `.to("cpu")`/`.to("cuda")` move the
         way plain bf16 tensors do -- attempting the same offload dance
         on a quantized model is a real risk of corrupting or crashing,
         not just wasted work, so this class skips it entirely rather
         than assume it's safe.
      This class exists BECAUSE the unquantized 32B backbone OOM'd the
      GPU (release_gpu/reacquire_gpu was the first fix) and then a
      63GB+ host-RAM shard-staging spike during from_pretrained SIGKILLed
      the process outright under a 60GB SLURM --mem allocation (exit
      137) -- 8-bit quantization was adopted specifically because it
      fixes BOTH the VRAM contention and the host-RAM staging spike at
      once, at the cost of quantization noise as an uncontrolled variable
      in the encoding-search results (not yet characterised).
    """

    def __init__(
        self, hf_id: str, dtype: str = "bfloat16",
        max_new_tokens: int = 4000, strip_think: bool = False,
        quantization: Optional[str] = None,
    ):
        self.hf_id = hf_id
        self.quantization = quantization
        self.tokenizer, self.model = _load_backbone_model(hf_id, dtype, quantization)
        self._device = self.model.device
        self.max_new_tokens = max_new_tokens
        self.strip_think = strip_think
        self.messages: list = []

    def reset(self) -> None:
        self.messages = []

    def release_gpu(self) -> None:
        """Move backbone weights to CPU so the search-tier ensemble has
        the full GPU to itself during ensemble_query_fn -- see class
        docstring; the OOM this fixes was hit on a real run, not
        hypothetical. No-op when quantized -- see class docstring for
        why a quantized model's footprint doesn't need this AND why
        moving it wouldn't be safe to attempt anyway."""
        if self.quantization is not None:
            return
        import torch
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def reacquire_gpu(self) -> None:
        """Move backbone weights back to GPU after the ensemble query
        that just ran (with the backbone off-GPU) has finished and
        unloaded its own member. No-op when quantized (see release_gpu)."""
        if self.quantization is not None:
            return
        self.model.to(self._device)

    def _generate(self, messages: list, max_new_tokens: Optional[int] = None) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if self.strip_think:
            text = strip_think_tags(text)
        return text

    def run_turn(
        self, user_msg: str, system_prompt: str,
        tool_budget_per_iter: int, response_max_tokens: int,
    ) -> str:
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": system_prompt + "\n\n" + _TOOL_PROTOCOL_TEXT,
            })
        self.messages.append({"role": "user", "content": user_msg})

        # self.max_new_tokens (config.yaml's per-backbone max_new_tokens,
        # e.g. 4000 for deepseek_r1_32b) is a FLOOR, not just a default --
        # a caller-supplied response_max_tokens smaller than that (e.g.
        # run_ghost_agent's own default of 2000, calibrated for short
        # numeric-field targets against Claude) would silently starve a
        # reasoning-heavy local model of the budget config.yaml explicitly
        # configured for it. Confirmed on a real run: with the caller's
        # 2000-token default winning, deepseek_r1_32b spent every
        # iteration after the first narrating a strategy in its reasoning
        # but never reached its own <final_encoding> closing tag before
        # generation cut off -- extract_proposed_encoding correctly
        # returned None every time, and the whole run failed to converge
        # for a budget reason, not a capability reason.
        effective_max_tokens = max(response_max_tokens, self.max_new_tokens)

        final_text = ""
        for _ in range(tool_budget_per_iter):
            text = self._generate(self.messages, max_new_tokens=effective_max_tokens)
            self.messages.append({"role": "assistant", "content": text})
            final_text = text

            calls = _parse_tool_calls(text)
            if not calls:
                break

            results = [
                f"Tool `{name}` result: {execute_tool_call(name, tool_input)}"
                for name, tool_input in calls
            ]
            self.messages.append({"role": "user", "content": "\n".join(results)})

        return final_text

    def generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """One-off, history-free call -- used for principle distillation.

        Same floor as run_turn: a reasoning model can burn most/all of a
        small max_tokens budget on hidden deliberation before any visible
        output (same failure mode as gpt56_sol/gemini_31_pro/
        deepseek_r1_14b already documented in CLAUDE.md), so the caller's
        max_tokens is not allowed to go below self.max_new_tokens."""
        effective_max_tokens = max(max_tokens, self.max_new_tokens)
        return self._generate(
            [{"role": "user", "content": prompt}], max_new_tokens=effective_max_tokens,
        )

    def close(self) -> None:
        from ensemble import unload_local_model
        unload_local_model(self.tokenizer, self.model)


# ── Config resolution ────────────────────────────────────────────────────

def resolve_backbone(name: str, config: dict, client=None):
    """
    Construct the agent backbone named `name` from config.yaml's
    agent_backbones.options section.

    Args:
        name: key into config["agent_backbones"]["options"]
        config: parsed config.yaml dict
        client: only used for provider="anthropic" entries -- an
            injectable Anthropic-SDK-shaped client, same seam as
            AnthropicBackbone's own `client` param (tests only; real
            runs should omit this and let AnthropicBackbone construct
            a real anthropic.Anthropic()).

    Returns:
        AnthropicBackbone or LocalReActBackbone instance.
    """
    options = config["agent_backbones"]["options"]
    if name not in options:
        raise ValueError(
            f"Unknown agent backbone '{name}'. Available: {list(options)}"
        )
    spec = options[name]
    provider = spec["provider"]

    if provider == "anthropic":
        return AnthropicBackbone(
            model_id=spec["model_id"], client=client,
        )
    if provider == "local":
        return LocalReActBackbone(
            hf_id=spec["hf_id"],
            dtype=spec.get("dtype", "bfloat16"),
            max_new_tokens=spec.get("max_new_tokens", 4000),
            strip_think=spec.get("strip_think_tags", False),
            quantization=spec.get("quantization"),
        )
    raise ValueError(f"Unknown agent backbone provider: {provider}")
