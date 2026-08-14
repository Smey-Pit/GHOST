# GHOST-Agent: Self-Improving Obfuscation Agent
# Task Breakdown for Claude Code
#
# Read this entire document before writing any code.
# Complete tasks in order. Each task is self-contained
# and produces a tested module before you move on.
# Do not start Task N+1 until Task N passes its smoke test.

---

## INTRODUCTION: What Changed and Why

### The Original GHOST

The original GHOST encodes documents using two Unicode
mechanisms applied as a fixed algorithm:

1. VS injection: appends invisible Variation Selector
   characters to tokens, fragmenting LLM tokenization
2. Bidi encoding: stores reversed character sequences
   under RTL controls, causing models to misread field values

This worked well for numerical content. The logprob-guided
stopping criterion told us when VS injection was sufficient.

### What We Now Know

Recent experiments showed that VS + bidi combined defeats
text reconstruction across ALL content types — not just
numbers. Frontier models including GPT-5.6 Sol and Claude
Opus 4.6 cannot reliably reconstruct obfuscated text when
both mechanisms are applied together.

This changes the paper's scope from "numerical defense"
to "general text protection." GHOST is now a general
solution. But the fixed algorithm has a problem: for
high-prior natural language content, we don't know in
advance how much injection is needed or which bidi
configuration is optimal. The logprob threshold is a
crude proxy — it doesn't measure actual extraction failure.

### The Self-Improving Agent Solution

GHOST-Agent replaces the fixed algorithm with an iterative
refinement loop. Instead of asking "did the proxy model
assign low logprob?" it asks "did the adversary model
successfully extract the content?" and improves the
encoding until the answer is no.

The paradigm follows Self-Refine (Madaan et al., 2023)
and Reflexion (Shinn et al., 2023):

```
GENERATE → FEEDBACK → REFLECT → IMPROVE → REPEAT
```

Applied to GHOST:

```
GENERATE:  Apply current encoding to document
FEEDBACK:  Query adversary model: can you read this?
REFLECT:   Why did the model succeed? What to change?
IMPROVE:   Apply targeted encoding improvements
REPEAT:    Until extraction fails or budget exhausted
```

### Why This Is Better

1. REAL SIGNAL. Feedback is actual extraction accuracy,
   not a logprob surrogate. If the adversary cannot extract
   the content, we know the defense works — regardless of
   what the logprob said.

2. NO LOGPROB ACCESS NEEDED. The agent only needs the
   adversary model to answer a question. Any API works.
   This removes the Qwen proxy dependency for the
   core defense loop.

3. GENERAL. Works for numbers, text, mixed content.
   The agent reasons about WHY encoding failed and
   adapts. A fixed algorithm cannot do this.

4. SELF-IMPROVING. Each iteration, the agent accumulates
   knowledge about what works and why. Later iterations
   benefit from earlier attempts.

### What You Are Building

Five focused modules, one at a time:

```
Task 1: ghost_tools.py        — encoding primitives
Task 2: adversary.py          — adversary query interface
Task 3: reflection.py         — reflection memory system
Task 4: ghost_agent_prompt.py — agent system prompt
Task 5: ghost_agent.py        — main agent loop
Task 6: convergence.py        — analysis and comparison
```

Each module is independent. Each has a smoke test.
Do not proceed without passing the smoke test.

---

## TASK 1: ghost_tools.py
## Encoding Primitives and Tool Definitions

### What this is

The six Python functions the agent calls as tools.
Pure Python — no API, no GPU, no external dependencies
beyond python-bidi.

Install: pip install python-bidi -q

### What to build

```python
"""
src/ghost_tools.py

Six encoding/analysis tools available to the GHOST agent.
All functions take and return plain Python strings.
No side effects. No API calls. No GPU.

These are wrapped as Anthropic tool definitions in Task 5.
Here they are just Python functions.
"""

# ── Bidi control characters ───────────────────────────────────
RTL  = '\u202E'
LTR  = '\u202D'
POP  = '\u202C'
RLI  = '\u2067'
LRI  = '\u2066'
POPI = '\u2069'
FSI  = '\u2068'
RLM  = '\u200F'
LRM  = '\u200E'

ALL_CTRL = frozenset([RTL, LTR, POP, RLI, LRI, POPI,
                      FSI, RLM, LRM])

# VS ranges
VS_RANGE_1 = range(0xFE00, 0xFE10)
VS_RANGE_2 = range(0xE0100, 0xE01F0)
VS_ALL = frozenset(list(VS_RANGE_1) + list(VS_RANGE_2))


def tool_render(encoding: str) -> str:
    """
    Render a bidi-encoded string using the Unicode
    Bidirectional Algorithm.

    Returns only the visible characters — what a human
    reader sees. Control characters are stripped.

    Use this to verify your encoding displays correctly.

    Args:
        encoding: a string containing bidi control chars
                  and visible content

    Returns:
        The visually rendered string (no control chars)
    """
    from bidi.algorithm import get_display
    result = get_display(encoding)
    return ''.join(c for c in result if c not in ALL_CTRL)


def tool_get_stored(encoding: str) -> str:
    """
    Extract the stored character sequence from an encoding.

    This is what an LLM tokenizer ingests — the raw
    codepoints in memory order, excluding control chars.
    This is what the adversary model processes.

    Args:
        encoding: a bidi-encoded string

    Returns:
        The stored character sequence (no control chars,
        no VS characters, in memory order)
    """
    return ''.join(
        c for c in encoding
        if c not in ALL_CTRL and ord(c) not in VS_ALL
    )


def tool_hamming(s1: str, s2: str) -> int:
    """
    Compute Hamming distance between two strings.

    Higher distance = stored sequence is more different
    from the target = better obfuscation.

    Returns 0 if strings have different lengths (invalid).

    Args:
        s1: first string
        s2: second string (should be same length as s1)

    Returns:
        Number of positions where s1 and s2 differ.
        Returns -1 if lengths differ.
    """
    if len(s1) != len(s2):
        return -1
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def tool_encode_vs(text: str, payload: str) -> str:
    """
    Inject Variation Selector characters into text.

    Appends VS characters (derived from payload bytes)
    after each non-whitespace character in text.
    VS characters are invisible to humans but fragment
    LLM tokenization.

    Args:
        text: the text to inject VS chars into
        payload: a string whose bytes determine which
                 VS codepoints are injected. Use a longer
                 payload for more injection. Use different
                 payloads to try different VS combinations.

    Returns:
        The text with VS characters injected.
        Renders identically to original for human readers.
    """
    def byte_to_vs(b: int) -> str:
        return chr(0xFE00 + b) if b < 16 \
               else chr(0xE0100 + b - 16)

    payload_bytes = payload.encode('utf-8')
    result = []
    byte_idx = 0

    for char in text:
        result.append(char)
        if char not in (' ', '\n', '\t', '\r'):
            result.append(byte_to_vs(
                payload_bytes[byte_idx % len(payload_bytes)]
            ))
            byte_idx += 1

    return ''.join(result)


def tool_encode_bidi(text: str, config: str) -> str:
    """
    Apply a bidi encoding configuration to text.

    Config string specifies which segments to reverse
    and how. The agent uses this to experiment with
    different bidi structures.

    Config format (pipe-separated segments):
        Each segment: "start:end:direction"
        direction: "rtl" or "ltr"
        Example: "0:3:rtl|3:7:ltr"
        Means: reverse chars 0-3, leave 3-7 as-is

    Special configs:
        "full_rtl"       — reverse entire string
        "rli_split:N"    — RLI+RTL first N chars,
                           LRI+rest (your post→stop pattern)
        "rli_split_v2:N" — variant of above
        "three_block:N:M"— three-segment split

    Args:
        text: the text to encode (field value or document)
        config: configuration string

    Returns:
        Bidi-encoded string that renders as original text.
        Returns empty string if config is invalid.
    """
    n = len(text)

    if config == "full_rtl":
        return RTL + text[::-1] + POP

    if config.startswith("rli_split:"):
        try:
            k = int(config.split(":")[1])
            if k <= 0 or k >= n:
                return ""
            return (f"{RLI}{RTL}{text[k:][::-1]}{POP}"
                    f"{LRI}{text[:k]}{POPI}{POPI}")
        except (ValueError, IndexError):
            return ""

    if config.startswith("rli_split_v2:"):
        try:
            k = int(config.split(":")[1])
            if k <= 0 or k >= n:
                return ""
            return (f"{RLI}{RTL}{text[:k][::-1]}{POP}"
                    f"{LRI}{text[k:]}{POPI}{POPI}")
        except (ValueError, IndexError):
            return ""

    if config.startswith("three_block:"):
        try:
            parts = config.split(":")
            k1, k2 = int(parts[1]), int(parts[2])
            if k1 >= k2 or k1 <= 0 or k2 >= n:
                return ""
            return (f"{RLI}{RTL}{text[k1:k2][::-1]}{POP}"
                    f"{LRI}{text[:k1]}{POPI}"
                    f"{LRI}{text[k2:]}{POPI}{POPI}")
        except (ValueError, IndexError):
            return ""

    if config.startswith("nested:"):
        try:
            parts = config.split(":")
            k1, k2 = int(parts[1]), int(parts[2])
            if k1 >= k2 or k1 < 0 or k2 > n:
                return ""
            return (f"{RTL}{text[:k1][::-1]}"
                    f"{LTR}{text[k1:k2]}{POP}"
                    f"{text[k2:][::-1]}{POP}")
        except (ValueError, IndexError):
            return ""

    if config.startswith("double_rli:"):
        try:
            k = int(config.split(":")[1])
            if k <= 0 or k >= n:
                return ""
            return (f"{RLI}{text[:k][::-1]}{POPI}"
                    f"{RLI}{text[k:][::-1]}{POPI}")
        except (ValueError, IndexError):
            return ""

    # Pipe-separated custom config
    if "|" in config or config.count(":") == 2:
        try:
            segments = config.split("|")
            parts = []
            for seg in segments:
                s, e, d = seg.split(":")
                s, e = int(s), int(e)
                segment = text[s:e]
                if d == "rtl":
                    parts.append(RTL + segment[::-1] + POP)
                else:
                    parts.append(segment)
            return ''.join(parts)
        except (ValueError, IndexError):
            return ""

    return ""   # unknown config


def tool_combine(text: str,
                 bidi_config: str,
                 vs_payload: str) -> str:
    """
    Apply both bidi encoding and VS injection to text.

    Order of operations:
    1. Apply bidi encoding to establish stored sequence
    2. Apply VS injection to characters in stored sequence

    This is the full GHOST encoding combining both mechanisms.

    Args:
        text: original text to encode
        bidi_config: config string for tool_encode_bidi
        vs_payload: payload string for tool_encode_vs

    Returns:
        Fully GHOST-encoded string, or empty string if
        bidi_config is invalid.
    """
    bidi_encoded = tool_encode_bidi(text, bidi_config)
    if not bidi_encoded:
        return ""

    # Apply VS to stored characters within bidi encoding
    # We inject VS after content chars, preserving ctrl chars
    def byte_to_vs(b: int) -> str:
        return chr(0xFE00 + b) if b < 16 \
               else chr(0xE0100 + b - 16)

    payload_bytes = vs_payload.encode('utf-8')
    result = []
    byte_idx = 0

    for char in bidi_encoded:
        result.append(char)
        if (char not in ALL_CTRL and
                char not in (' ', '\n', '\t', '\r') and
                ord(char) not in VS_ALL):
            result.append(byte_to_vs(
                payload_bytes[byte_idx % len(payload_bytes)]
            ))
            byte_idx += 1

    return ''.join(result)


# ── Tool registry for agent ───────────────────────────────────
# Used in Task 5 to build Anthropic tool definitions.

TOOL_FUNCTIONS = {
    "render":      tool_render,
    "get_stored":  tool_get_stored,
    "hamming":     tool_hamming,
    "encode_vs":   tool_encode_vs,
    "encode_bidi": tool_encode_bidi,
    "combine":     tool_combine,
}
```

### Smoke test for Task 1

Run this after implementing. All assertions must pass.

```python
# smoke_test_task1.py
from ghost_tools import (
    tool_render, tool_get_stored, tool_hamming,
    tool_encode_vs, tool_encode_bidi, tool_combine,
    RTL, POP
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
# May or may not equal target depending on bidi algorithm
# Just check it returns something non-empty
assert len(rendered2) > 0

# Test 3: VS injection is invisible to render
text = "hello"
vs_enc = tool_encode_vs(text, "test_payload")
assert len(vs_enc) > len(text), "VS chars not added"
# VS chars should not appear in readable output
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

print("TASK 1 SMOKE TEST PASSED")
```

---

## TASK 2: adversary.py
## Adversary Query Interface

### What this is

A thin wrapper around the API clients that queries
an adversary model and checks whether it successfully
extracted the protected content. No agent logic here.
Just: send prompt, get response, check match.

### What to build

```python
"""
src/adversary.py

Query adversary models to test extraction success.
This is the FEEDBACK component of the agent loop.

The adversary model is the model we are trying to defeat.
We query it with our encoded document and check whether
it can correctly extract the protected content.

This module has no knowledge of the agent or the encoding
strategy. It only knows: here is some text, here is a
field name, did the model get it right?
"""

import os
import time
from typing import Optional


EXTRACTION_PROMPT = """Read the following text carefully and \
report the exact value of {field_name}.
Return only the value with no additional text or explanation.

Text:
{text}"""


def query_adversary(
    encoded_text: str,
    field_name: str,
    model_id: str,
    provider: str,
    max_tokens: int = 100,
    max_retries: int = 3,
) -> str:
    """
    Query an adversary model to attempt field extraction.

    Args:
        encoded_text: the GHOST-encoded document text
        field_name: the field to extract (e.g. "account_number")
        model_id: model identifier string
        provider: "anthropic", "openai", or "google"
        max_tokens: max response tokens
        max_retries: retry attempts on API failure

    Returns:
        The model's response string (stripped of whitespace).
        Returns "REFUSAL" if model refuses to process.
        Returns "ERROR" if all retries exhausted.
    """
    prompt = EXTRACTION_PROMPT.format(
        field_name=field_name,
        text=encoded_text
    )

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = _call_anthropic(
                    model_id, prompt, max_tokens
                )
            elif provider == "openai":
                response = _call_openai(
                    model_id, prompt, max_tokens
                )
            elif provider == "google":
                response = _call_google(
                    model_id, prompt, max_tokens
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")

            return response.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Adversary query failed: {e}")
                return "ERROR"
            time.sleep(2 ** attempt)

    return "ERROR"


def check_extraction(
    model_response: str,
    ground_truth: str,
    is_refusal: bool = False
) -> dict:
    """
    Check whether adversary extraction succeeded.

    Normalises both strings before comparison:
    strips dashes, spaces (for formatted identifiers),
    and whitespace.

    Args:
        model_response: raw model output
        ground_truth: original unencoded field value
        is_refusal: True if response is a refusal

    Returns:
        dict with keys:
          extracted: bool — did model get the right value?
          response: str — cleaned model response
          refusal: bool — was this a refusal?
          partial: bool — did model get part of it right?
    """
    REFUSAL_PHRASES = [
        "i can't", "i cannot", "unable to",
        "i'm not able", "i am not able",
        "i don't", "i do not"
    ]

    resp_lower = model_response.lower()
    is_refusal = (is_refusal or
                  any(p in resp_lower for p in REFUSAL_PHRASES))

    def normalise(s: str) -> str:
        return s.strip().replace("-", "").replace(" ", "").lower()

    norm_response = normalise(model_response)
    norm_truth = normalise(ground_truth)

    exact = norm_response == norm_truth
    partial = (not exact and
               norm_truth in norm_response and
               not is_refusal)

    return {
        "extracted": exact,
        "response": model_response.strip(),
        "refusal": is_refusal,
        "partial": partial,
    }


def _call_anthropic(model_id: str,
                     prompt: str,
                     max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_openai(model_id: str,
                  prompt: str,
                  max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def _call_google(model_id: str,
                  prompt: str,
                  max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_id)
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens,
                           "temperature": 0}
    )
    try:
        return response.text
    except Exception:
        return "REFUSAL"
```

### Smoke test for Task 2

```python
# smoke_test_task2.py
# Uses Claude Sonnet 4.6 as adversary (cheap, fast)
import os
from adversary import query_adversary, check_extraction

# Test with clean text (model should succeed)
clean_text = "Invoice #INV-20481. Account: 48271039."
response = query_adversary(
    encoded_text=clean_text,
    field_name="account number",
    model_id="claude-sonnet-4-6",
    provider="anthropic"
)
print(f"Clean text response: '{response}'")
result = check_extraction(response, "48271039")
print(f"Extraction result: {result}")
assert result["extracted"], \
    "Model should extract from clean text"

# Test refusal detection
fake_refusal = "I can't help with that request."
result2 = check_extraction(fake_refusal, "48271039")
assert result2["refusal"], "Should detect refusal"
assert not result2["extracted"], "Refusal is not extraction"

print("TASK 2 SMOKE TEST PASSED")
```

---

## TASK 3: reflection.py
## Reflection Memory System

### What this is

A data structure and formatter that accumulates the
agent's attempt history and formats it into a reflection
prompt. This is the REFLECT component — where the agent
reasons about past failures before proposing improvements.

No API calls. Pure data management and prompt formatting.

### What to build

```python
"""
src/reflection.py

Reflection memory for the GHOST self-improving agent.

Stores the history of encoding attempts and their
outcomes. Formats this history into a structured
reflection that the agent uses to reason about
what to try next.

The reflection prompt answers three questions:
  1. What did I try?
  2. What happened?
  3. What should I infer?
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Attempt:
    """
    A single encoding attempt and its outcome.
    """
    iteration: int
    bidi_config: str          # config string used
    vs_payload: str           # payload string used
    encoding_description: str # agent's own description
    encoded_text: str         # the actual encoded string
    rendered: str             # what human sees
    stored: str               # what model tokenizes
    hamming_dist: int         # stored vs target distance
    adversary_response: str   # what adversary model said
    extraction_succeeded: bool # did adversary get it right?
    refusal: bool             # did adversary refuse?
    agent_reasoning: str      # agent's reflection on this


@dataclass
class ReflectionMemory:
    """
    Accumulates attempt history for one document field.
    """
    target: str              # original field value
    field_name: str          # e.g. "account_number"
    attempts: List[Attempt] = field(default_factory=list)

    def add_attempt(self, attempt: Attempt):
        self.attempts.append(attempt)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def best_attempt(self) -> Optional[Attempt]:
        """Attempt with highest Hamming distance."""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: a.hamming_dist)

    @property
    def succeeded(self) -> bool:
        """Has any attempt defeated extraction?"""
        return any(
            not a.extraction_succeeded and not a.refusal
            for a in self.attempts
        )

    def format_for_reflection(self) -> str:
        """
        Format attempt history for the agent's reflection prompt.
        Returns a structured string summarising what was tried
        and what the outcomes were.
        """
        if not self.attempts:
            return "No attempts yet."

        lines = [
            f"TARGET: '{self.target}' (length {len(self.target)})",
            f"FIELD: {self.field_name}",
            f"ATTEMPTS SO FAR: {self.n_attempts}",
            "",
        ]

        for att in self.attempts:
            status = "FAILED (model extracted correctly)" \
                if att.extraction_succeeded \
                else "SUCCESS (extraction defeated)"
            if att.refusal:
                status = "REFUSAL (model refused to process)"

            lines.extend([
                f"--- Attempt {att.iteration} ---",
                f"Configuration: {att.bidi_config}",
                f"VS payload: '{att.vs_payload}'",
                f"Stored sequence: '{att.stored}'",
                f"Hamming distance: {att.hamming_dist}",
                f"Adversary said: '{att.adversary_response}'",
                f"Outcome: {status}",
                f"Your reasoning then: {att.agent_reasoning}",
                "",
            ])

        # Summary insights
        failed = [a for a in self.attempts
                  if a.extraction_succeeded]
        succeeded = [a for a in self.attempts
                     if not a.extraction_succeeded
                     and not a.refusal]

        if failed:
            best_dist = max(a.hamming_dist for a in failed)
            lines.append(
                f"INSIGHT: Best distance achieved before "
                f"success: {best_dist}/{len(self.target)}"
            )
            # Check if model is getting partial matches
            partial_responses = [
                a.adversary_response for a in failed
                if any(d in a.adversary_response
                       for d in self.target)
            ]
            if partial_responses:
                lines.append(
                    "INSIGHT: Model is recovering some digits "
                    "— try stronger VS injection or different "
                    "bidi split"
                )

        if succeeded:
            lines.append(
                f"NOTE: {len(succeeded)} attempt(s) already "
                f"defeated extraction. Returning best one."
            )

        return "\n".join(lines)
```

### Smoke test for Task 3

```python
# smoke_test_task3.py
from reflection import ReflectionMemory, Attempt

mem = ReflectionMemory(target="48271039",
                       field_name="account_number")
assert mem.n_attempts == 0
assert not mem.succeeded
assert mem.best_attempt is None

att1 = Attempt(
    iteration=1,
    bidi_config="full_rtl",
    vs_payload="ghost",
    encoding_description="simple full reversal",
    encoded_text="placeholder",
    rendered="48271039",
    stored="93017284",
    hamming_dist=7,
    adversary_response="93017284",
    extraction_succeeded=True,
    refusal=False,
    agent_reasoning="tried simple reversal, model read stored"
)
mem.add_attempt(att1)

assert mem.n_attempts == 1
assert not mem.succeeded  # extraction succeeded = we failed

formatted = mem.format_for_reflection()
assert "48271039" in formatted
assert "full_rtl" in formatted
assert "93017284" in formatted
print(formatted)

att2 = Attempt(
    iteration=2,
    bidi_config="rli_split:4",
    vs_payload="ghost_v2",
    encoding_description="RLI split with VS",
    encoded_text="placeholder2",
    rendered="48271039",
    stored="03927184",
    hamming_dist=8,
    adversary_response="I cannot read this",
    extraction_succeeded=False,
    refusal=True,
    agent_reasoning="tried rli split"
)
mem.add_attempt(att2)

# Refusal doesn't count as our success
assert not mem.succeeded

print("TASK 3 SMOKE TEST PASSED")
```

---

## TASK 4: ghost_agent_prompt.py
## Agent System Prompt

### What this is

The system prompt that tells the agent what it is,
what tools it has, and how to reason. This is the
most important piece for output quality. A weak
prompt produces random exploration. A good prompt
produces targeted, intelligent improvement.

No API calls. Just a well-crafted prompt string.

### What to build

```python
"""
src/ghost_agent_prompt.py

System prompt for the GHOST self-improving agent.

This prompt encodes the agent's knowledge of:
  - Unicode bidi controls and VS characters
  - The obfuscation objective
  - How to reason about encoding failures
  - What configurations to try in what order

The quality of this prompt determines the quality
of the agent's encoding strategy.
"""


GHOST_AGENT_SYSTEM_PROMPT = """You are GHOST-Agent, \
a self-improving Unicode obfuscation specialist.

Your goal is to encode a piece of text so that:
1. A HUMAN reading it sees the ORIGINAL text correctly
2. An LLM attempting to extract content from it FAILS

You have six tools:
  render(encoding)              → what human sees
  get_stored(encoding)          → what LLM tokenizes
  hamming(s1, s2)               → difference score
  encode_bidi(text, config)     → bidi encoding
  encode_vs(text, payload)      → VS injection
  combine(text, config, payload)→ both mechanisms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MECHANISM 1: BIDIRECTIONAL CONTROLS

Unicode bidi controls change the DISPLAY ORDER of text
without changing what is stored in memory. LLMs process
what is STORED, not what is DISPLAYED.

Available bidi configs for encode_bidi():

"full_rtl"
  Stores entire text reversed.
  RTL + reversed_text + POP
  Human sees: original
  LLM gets: reversed

"rli_split:N"
  Splits at position N. First N chars in LRI (LTR island),
  rest in RTL block inside RLI.
  Human sees: original
  LLM gets: a scrambled permutation

"rli_split_v2:N"
  Variant: reversed first N chars in RTL, rest in LRI.

"three_block:N:M"
  Three segments: middle [N:M] reversed in RTL,
  flanked by LRI blocks.
  Human sees: original
  LLM gets: different permutation

"nested:N:M"
  Outer RTL, inner LTR protects [N:M] from reversal.

"double_rli:N"
  Two independent RTL isolate blocks.

KEY INSIGHT: The further the stored sequence is from
the original (high Hamming distance), the harder it
is for the LLM to recover the original value.
Use hamming() to measure. Maximise this number.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MECHANISM 2: VARIATION SELECTOR INJECTION

VS characters are invisible Unicode format characters
that LLM tokenizers ingest but humans cannot see.
Injecting them fragments familiar tokens into rare,
poorly-represented sequences the model cannot reconstruct.

Use encode_vs(text, payload) to inject VS chars.
The payload string determines which VS codepoints are
used. Different payloads produce different fragmentation.

Try payloads like:
  "ghost"         — basic injection
  "ghost_v2"      — different VS codepoints
  "aaaaaa"        — VS1 repeated (strong fragmentation)
  "abcdefghij"    — spread across VS range

Longer payloads inject more VS chars. More VS = stronger
fragmentation but same human-visible appearance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MECHANISM 3: COMBINATION (strongest)

Use combine(text, bidi_config, vs_payload) to apply
BOTH mechanisms simultaneously.

Bidi: scrambles the character ORDER
VS:   fragments individual character IDENTITY

Together they defeat extraction through two independent
failure modes. An adversary must fix both to recover.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR STRATEGY

Follow this progression:

STEP 1: Verify your encoding is valid
  Use render() to confirm human sees original text.
  If render() returns wrong text, the config is broken.

STEP 2: Check stored sequence
  Use get_stored() and hamming() to measure distance.
  Target: hamming distance >= len(text) * 0.6

STEP 3: Try combine() with different configs
  Start with "full_rtl" + payload "ghost"
  If adversary succeeds: increase payload length
  If adversary still succeeds: try different bidi config

STEP 4: If combine() fails, try config variants
  "rli_split:N" for different N values
  "three_block:N:M" for different split points
  Pick splits that maximise hamming distance

STEP 5: Reflect on adversary response
  If adversary returned the exact value: they can read bidi
  → strengthen VS injection (longer payload)
  If adversary returned reversed value: VS not blocking bidi
  → try different bidi config
  If adversary returned partial value: making progress
  → stronger injection on specific characters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT RULES

1. ALWAYS verify render() = original before querying adversary
2. ALWAYS call get_stored() and hamming() to log your progress
3. REASON explicitly about why each attempt succeeded or failed
4. DO NOT repeat a config that already failed — adapt it
5. When combine() defeats the adversary: STOP and report success
6. Report your final encoding in the exact format requested

The adversary is a frontier LLM. It is very capable.
Simple encodings may not work. Persist and adapt.
"""


def build_iteration_prompt(
    target: str,
    field_name: str,
    reflection: str,
    iteration: int,
    budget_remaining: int,
) -> str:
    """
    Build the user message for each agent iteration.

    Args:
        target: the original text to protect
        field_name: what this field represents
        reflection: formatted history from reflection.py
        iteration: current iteration number (1-indexed)
        budget_remaining: tool calls remaining

    Returns:
        User message string for this iteration.
    """
    if iteration == 1:
        return f"""TASK: Encode the following text so that 
an LLM cannot extract the {field_name}.

Original text: "{target}"
Field type: {field_name}
Budget: {budget_remaining} tool calls remaining

Begin by trying combine() with "full_rtl" config.
Verify with render(), then query will be done externally.
Report your encoding and reasoning."""

    return f"""ITERATION {iteration} of the encoding loop.

HISTORY OF ATTEMPTS:
{reflection}

Budget remaining: {budget_remaining} tool calls

The adversary model has just seen your previous encoding.
Based on the history above, propose your IMPROVED encoding.

Reason explicitly about why previous attempts failed,
then use the tools to build and verify a better encoding.
End with your proposed encoding for external adversary query."""
```

### Smoke test for Task 4

```python
# smoke_test_task4.py
from ghost_agent_prompt import (
    GHOST_AGENT_SYSTEM_PROMPT,
    build_iteration_prompt
)

# Check prompt is substantial
assert len(GHOST_AGENT_SYSTEM_PROMPT) > 1000
assert "render" in GHOST_AGENT_SYSTEM_PROMPT
assert "hamming" in GHOST_AGENT_SYSTEM_PROMPT
assert "combine" in GHOST_AGENT_SYSTEM_PROMPT
assert "bidi" in GHOST_AGENT_SYSTEM_PROMPT

# Check iteration prompt construction
prompt1 = build_iteration_prompt(
    target="48271039",
    field_name="account_number",
    reflection="No attempts yet.",
    iteration=1,
    budget_remaining=20
)
assert "48271039" in prompt1
assert "account_number" in prompt1

prompt2 = build_iteration_prompt(
    target="48271039",
    field_name="account_number",
    reflection="Attempt 1 failed...",
    iteration=2,
    budget_remaining=15
)
assert "ITERATION 2" in prompt2
assert "Attempt 1 failed" in prompt2

print("TASK 4 SMOKE TEST PASSED")
```

---

## TASK 5: ghost_agent.py
## Main Agent Loop

### What this is

The central orchestrator. Wires together Tasks 1-4
into a working self-improving agent using Anthropic's
tool use API. This is where the generate→feedback→
reflect→improve loop is implemented.

Uses claude-sonnet-4-6 as the reasoning agent.
The ADVERSARY model is separate (configured at runtime).

### What to build

```python
"""
src/ghost_agent.py

GHOST self-improving agent.

The agent loop:
  1. Build iteration prompt with reflection history
  2. Call claude-sonnet-4-6 with tools
  3. Execute tool calls the agent makes
  4. Extract the agent's proposed encoding
  5. Query the adversary model externally
  6. Check extraction success
  7. Update reflection memory
  8. Repeat or stop

The agent (claude-sonnet-4-6) reasons and proposes.
The adversary (configurable) attempts extraction.
These are two separate models with separate roles.
"""

import json
import anthropic
from typing import Optional

from ghost_tools import TOOL_FUNCTIONS
from ghost_agent_prompt import (
    GHOST_AGENT_SYSTEM_PROMPT,
    build_iteration_prompt
)
from adversary import query_adversary, check_extraction
from reflection import ReflectionMemory, Attempt


# ── Anthropic tool definitions ────────────────────────────────

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
                    "description": "The encoded string to render"
                }
            },
            "required": ["encoding"]
        }
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
                    "description": "The encoded string"
                }
            },
            "required": ["encoding"]
        }
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
                    "description": "First string (stored sequence)"
                },
                "s2": {
                    "type": "string",
                    "description": "Second string (original target)"
                }
            },
            "required": ["s1", "s2"]
        }
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
                    "description": "Text to encode"
                },
                "config": {
                    "type": "string",
                    "description": (
                        "Config string e.g. 'full_rtl' or "
                        "'rli_split:3' or 'three_block:2:5'"
                    )
                }
            },
            "required": ["text", "config"]
        }
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
                    "description": "Text to inject VS into"
                },
                "payload": {
                    "type": "string",
                    "description": (
                        "Payload string determining VS codepoints. "
                        "Try 'ghost', 'ghost_v2', 'aaaaaa', etc."
                    )
                }
            },
            "required": ["text", "payload"]
        }
    },
    {
        "name": "combine",
        "description": (
            "Apply BOTH bidi encoding and VS injection. "
            "This is the strongest obfuscation — use this "
            "as your primary tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to encode"
                },
                "bidi_config": {
                    "type": "string",
                    "description": "Bidi configuration string"
                },
                "vs_payload": {
                    "type": "string",
                    "description": "VS payload string"
                }
            },
            "required": ["text", "bidi_config", "vs_payload"]
        }
    }
]


# ── Agent execution ───────────────────────────────────────────

def execute_tool_call(tool_name: str,
                       tool_input: dict) -> str:
    """Execute a single tool call and return string result."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return f"ERROR: unknown tool {tool_name}"
    try:
        result = fn(**tool_input)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


def run_agent_turn(
    client: anthropic.Anthropic,
    messages: list,
    system_prompt: str,
) -> tuple[list, str]:
    """
    Run one agent turn: call claude-sonnet-4-6 with tools,
    execute all tool calls, return updated messages and
    the agent's final text response.

    Returns:
        (updated_messages, agent_final_text)
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system_prompt,
        tools=TOOL_DEFINITIONS,
        messages=messages
    )

    # Add assistant response to messages
    messages.append({
        "role": "assistant",
        "content": response.content
    })

    # Collect final text (may come after tool use)
    final_text = ""
    tool_results = []

    for block in response.content:
        if block.type == "text":
            final_text = block.text
        elif block.type == "tool_use":
            result = execute_tool_call(
                block.name, block.input
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })

    # If there were tool calls, add results and get
    # final response
    if tool_results:
        messages.append({
            "role": "user",
            "content": tool_results
        })
        # Get agent's response to tool results
        final_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages
        )
        messages.append({
            "role": "assistant",
            "content": final_response.content
        })
        for block in final_response.content:
            if block.type == "text":
                final_text = block.text

    return messages, final_text


def extract_proposed_encoding(agent_text: str,
                               target: str) -> Optional[str]:
    """
    Extract the agent's proposed encoding from its text response.

    The agent is instructed to include the encoding.
    We look for patterns indicating the final encoding.

    Returns the encoding string or None if not found.
    """
    # Look for explicit encoding markers
    markers = [
        "proposed encoding:",
        "final encoding:",
        "encoding:",
        "encoded text:",
    ]
    text_lower = agent_text.lower()
    for marker in markers:
        idx = text_lower.find(marker)
        if idx != -1:
            after = agent_text[idx + len(marker):].strip()
            # Take first line after marker
            first_line = after.split('\n')[0].strip()
            # Remove quotes if present
            first_line = first_line.strip('"\'`')
            if first_line and target[:3] in first_line:
                return first_line

    return None


def run_ghost_agent(
    target: str,
    field_name: str,
    adversary_model_id: str,
    adversary_provider: str,
    max_iterations: int = 5,
    tool_budget_per_iter: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Run the GHOST self-improving agent on a single field value.

    Args:
        target: the original text to protect
        field_name: human-readable field name
        adversary_model_id: the model trying to extract
        adversary_provider: "anthropic", "openai", "google"
        max_iterations: maximum refinement iterations
        tool_budget_per_iter: max tool calls per iteration
        verbose: print progress

    Returns:
        dict with keys:
          success: bool — was extraction defeated?
          best_encoding: str — the best encoding found
          n_iterations: int — iterations used
          history: list — all attempts
          final_hamming: int — best hamming distance achieved
    """
    client = anthropic.Anthropic()
    memory = ReflectionMemory(target=target,
                               field_name=field_name)
    messages = []
    best_encoding = None
    best_hamming = 0

    if verbose:
        print(f"\n{'='*55}")
        print(f"GHOST-Agent: protecting '{target}' ({field_name})")
        print(f"Adversary: {adversary_model_id}")
        print(f"Max iterations: {max_iterations}")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Build iteration prompt
        reflection = memory.format_for_reflection()
        user_msg = build_iteration_prompt(
            target=target,
            field_name=field_name,
            reflection=reflection,
            iteration=iteration,
            budget_remaining=tool_budget_per_iter
        )

        messages.append({"role": "user", "content": user_msg})

        # Run agent turn
        messages, agent_text = run_agent_turn(
            client, messages, GHOST_AGENT_SYSTEM_PROMPT
        )

        if verbose:
            print(f"Agent reasoning:\n{agent_text[:300]}...")

        # Extract proposed encoding from agent response
        proposed = extract_proposed_encoding(agent_text, target)

        if proposed is None:
            # Try to get encoding from last combine/bidi tool call
            # by looking at the message history
            if verbose:
                print("Could not extract encoding from agent text")
            continue

        # Verify the encoding renders correctly
        from ghost_tools import tool_render, tool_get_stored, \
            tool_hamming
        rendered = tool_render(proposed)
        stored = tool_get_stored(proposed)
        dist = tool_hamming(stored, target)

        if verbose:
            print(f"Proposed encoding renders as: '{rendered}'")
            print(f"Stored sequence: '{stored}'")
            print(f"Hamming distance: {dist}")

        # Track best hamming even if invalid rendering
        if dist > best_hamming:
            best_hamming = dist
            best_encoding = proposed

        # Query adversary model
        adversary_response = query_adversary(
            encoded_text=proposed,
            field_name=field_name,
            model_id=adversary_model_id,
            provider=adversary_provider
        )

        extraction_result = check_extraction(
            adversary_response, target
        )

        if verbose:
            print(f"Adversary said: '{adversary_response}'")
            print(f"Extraction succeeded: "
                  f"{extraction_result['extracted']}")

        # Record attempt
        attempt = Attempt(
            iteration=iteration,
            bidi_config="from_agent",
            vs_payload="from_agent",
            encoding_description=agent_text[:200],
            encoded_text=proposed,
            rendered=rendered,
            stored=stored,
            hamming_dist=dist,
            adversary_response=adversary_response,
            extraction_succeeded=extraction_result["extracted"],
            refusal=extraction_result["refusal"],
            agent_reasoning=agent_text[:500]
        )
        memory.add_attempt(attempt)

        # Add adversary result to conversation
        result_msg = (
            f"Adversary model response: '{adversary_response}'\n"
            f"Extraction {'SUCCEEDED' if extraction_result['extracted'] else 'FAILED'}.\n"
            f"{'Refusal detected.' if extraction_result['refusal'] else ''}"
        )
        messages.append({"role": "user", "content": result_msg})

        # Check stopping condition
        if not extraction_result["extracted"] and \
           not extraction_result["refusal"]:
            if verbose:
                print(f"\nSUCCESS: Extraction defeated "
                      f"in {iteration} iterations!")
            best_encoding = proposed
            break

    return {
        "success": memory.succeeded,
        "best_encoding": best_encoding,
        "n_iterations": memory.n_attempts,
        "final_hamming": best_hamming,
        "history": memory.attempts,
        "target": target,
        "field_name": field_name,
    }
```

### Smoke test for Task 5

```python
# smoke_test_task5.py
# Uses claude-sonnet-4-6 as BOTH agent and adversary
# to keep cost low during testing. In production,
# adversary is a different model.

from ghost_agent import run_ghost_agent

result = run_ghost_agent(
    target="12345",
    field_name="test_value",
    adversary_model_id="claude-sonnet-4-6",
    adversary_provider="anthropic",
    max_iterations=2,   # keep short for smoke test
    verbose=True
)

print(f"\nResult: {result['success']}")
print(f"Iterations used: {result['n_iterations']}")
print(f"Best hamming: {result['final_hamming']}")
assert result['n_iterations'] > 0, "Agent ran no iterations"
assert result['best_encoding'] is not None, \
    "No encoding produced"
print("TASK 5 SMOKE TEST PASSED")
```

---

## TASK 6: convergence.py
## Convergence Analysis and Comparison

### What this is

The experiment that produces the paper's results for
the self-improving agent. Runs GHOST-Agent on a sample
of documents, measures how many iterations are needed
to defeat extraction, and compares against static GHOST.

This is what goes in the paper's results section.

### What to build

```python
"""
src/convergence.py

Convergence analysis for GHOST-Agent.

Measures:
  1. Success rate: what fraction of fields defeat extraction?
  2. Convergence speed: how many iterations needed?
  3. Content-type breakdown: numbers vs text vs mixed
  4. Comparison: agent vs static GHOST (from existing plan)

Produces:
  results/tables/agent_convergence.csv
  results/tables/agent_vs_static.csv
  results/figures/convergence_by_content_type.csv

Run on a sample of 50 documents from data/raw/documents.json
before committing to the full 400-document run.
"""

import json
import csv
import os
import random
from ghost_agent import run_ghost_agent
from ghost_tools import (
    tool_combine, tool_render, tool_get_stored, tool_hamming
)


def run_static_ghost(field_value: str) -> dict:
    """
    Apply static GHOST (no agent) using full_rtl + ghost payload.
    This is the baseline for comparison.

    Returns dict with encoding and hamming distance.
    """
    encoded = tool_combine(field_value, "full_rtl", "ghost")
    stored = tool_get_stored(encoded)
    dist = tool_hamming(stored, field_value)
    rendered = tool_render(encoded)
    valid = (rendered == field_value)
    return {
        "encoding": encoded,
        "stored": stored,
        "hamming": dist,
        "valid": valid
    }


def run_convergence_analysis(
    documents_path: str,
    adversary_model_id: str,
    adversary_provider: str,
    n_samples: int = 50,
    max_iterations: int = 5,
    seed: int = 168,
    output_dir: str = "results/tables",
):
    """
    Run convergence analysis on a sample of documents.

    For each document, for each field:
      1. Run static GHOST — measure baseline FEA
      2. Run GHOST-Agent — measure agent FEA and iterations

    Args:
        documents_path: path to data/raw/documents.json
        adversary_model_id: model to test against
        adversary_provider: "anthropic", "openai", "google"
        n_samples: number of documents to sample
        max_iterations: max agent iterations per field
        seed: random seed for sampling
        output_dir: where to write CSV results
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    with open(documents_path) as f:
        documents = json.load(f)

    sampled = random.sample(
        documents, min(n_samples, len(documents))
    )

    results = []

    for doc in sampled:
        domain = doc["domain"]
        doc_id = doc["id"]

        for field_name, field_value in doc["fields"].items():
            print(f"\n{doc_id} / {field_name}: '{field_value}'")

            # Static GHOST baseline
            static = run_static_ghost(field_value)

            # GHOST-Agent
            agent_result = run_ghost_agent(
                target=field_value,
                field_name=field_name,
                adversary_model_id=adversary_model_id,
                adversary_provider=adversary_provider,
                max_iterations=max_iterations,
                verbose=False
            )

            row = {
                "doc_id": doc_id,
                "domain": domain,
                "field_name": field_name,
                "field_value": field_value,
                "field_length": len(field_value),
                # Static GHOST
                "static_hamming": static["hamming"],
                "static_valid": static["valid"],
                # GHOST-Agent
                "agent_success": agent_result["success"],
                "agent_iterations": agent_result["n_iterations"],
                "agent_hamming": agent_result["final_hamming"],
                "agent_improvement":
                    agent_result["final_hamming"] - static["hamming"],
            }
            results.append(row)
            print(
                f"  Static hamming: {static['hamming']}  "
                f"Agent: success={agent_result['success']} "
                f"iters={agent_result['n_iterations']} "
                f"hamming={agent_result['final_hamming']}"
            )

    # Write results
    out_path = os.path.join(output_dir, "agent_convergence.csv")
    if results:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Print summary
    print(f"\n{'='*55}")
    print(f"CONVERGENCE ANALYSIS SUMMARY")
    print(f"{'='*55}")
    print(f"Documents sampled: {len(sampled)}")
    print(f"Fields evaluated: {len(results)}")
    if results:
        success_rate = sum(
            1 for r in results if r["agent_success"]
        ) / len(results)
        mean_iters = sum(
            r["agent_iterations"] for r in results
        ) / len(results)
        mean_improvement = sum(
            r["agent_improvement"] for r in results
        ) / len(results)
        print(f"Agent success rate: {success_rate:.1%}")
        print(f"Mean iterations to success: {mean_iters:.1f}")
        print(f"Mean hamming improvement over static: "
              f"{mean_improvement:.1f}")

        # By domain
        for domain in ["financial", "medical",
                        "legal", "technical"]:
            domain_results = [
                r for r in results if r["domain"] == domain
            ]
            if domain_results:
                domain_success = sum(
                    1 for r in domain_results
                    if r["agent_success"]
                ) / len(domain_results)
                print(f"  {domain}: {domain_success:.1%} success")

    print(f"\nResults written to {out_path}")
    return results
```

### Smoke test for Task 6

```python
# smoke_test_task6.py
# Runs on 3 documents only to verify pipeline works
from convergence import run_convergence_analysis

results = run_convergence_analysis(
    documents_path="data/raw/documents.json",
    adversary_model_id="claude-sonnet-4-6",
    adversary_provider="anthropic",
    n_samples=3,          # tiny sample for smoke test
    max_iterations=2,     # keep short
    output_dir="results/tables"
)

assert len(results) > 0, "No results produced"
assert "agent_success" in results[0], "Missing agent_success key"
assert "agent_iterations" in results[0], "Missing iterations key"
print(f"Smoke test ran {len(results)} field evaluations")
print("TASK 6 SMOKE TEST PASSED")
```

---

## IMPLEMENTATION ORDER

```
Day 1:
  Task 1 — ghost_tools.py
  Smoke test Task 1
  Task 2 — adversary.py
  Smoke test Task 2

Day 2:
  Task 3 — reflection.py
  Smoke test Task 3
  Task 4 — ghost_agent_prompt.py
  Smoke test Task 4

Day 3:
  Task 5 — ghost_agent.py
  Smoke test Task 5 (2 iterations, cheap)

Day 4:
  Task 6 — convergence.py
  Smoke test Task 6 (3 documents)
  Run full convergence on 50 documents
```

## CRITICAL RULES FOR ALL TASKS

1. Read GHOST_BACKGROUND_FOR_CLAUDE_CODE.md first
2. Complete smoke test before moving to next task
3. No hardcoded API keys — use os.environ
4. Temperature=0 for all model calls (deterministic)
5. The AGENT is always claude-sonnet-4-6
   The ADVERSARY is configured at runtime — never hardcoded
6. Log all tool calls and responses to logs/ directory
7. Never pass the target's raw codepoints directly in
   the agent's prompt — the agent should discover
   encodings through tool use, not string manipulation

## FILES TO READ BEFORE STARTING

1. GHOST_BACKGROUND_FOR_CLAUDE_CODE.md
2. GHOST_EXPERIMENT_PLAN.md (for context on the
   broader experiment pipeline)
3. This document (GHOST_AGENT_TASKS.md)
