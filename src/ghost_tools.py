"""
src/ghost_tools.py

Six encoding/analysis tools available to the GHOST-Agent (see
GHOST_AGENT_TASKS.md, Task 1). All functions take and return plain
Python strings. No side effects. No API calls. No GPU.

These are wrapped as Anthropic tool definitions in Task 5.
Here they are just Python functions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from unicode_utils import byte_to_vs, derive_payload_bytes  # noqa: E402

# ── Bidi control characters ───────────────────────────────────
RTL = '‮'
LTR = '‭'
POP = '‬'
RLI = '⁧'
LRI = '⁦'
POPI = '⁩'
FSI = '⁨'
RLM = '‏'
LRM = '‎'

ALL_CTRL = frozenset([RTL, LTR, POP, RLI, LRI, POPI,
                       FSI, RLM, LRM])

# VS ranges
VS_RANGE_1 = range(0xFE00, 0xFE10)
VS_RANGE_2 = range(0xE0100, 0xE01F0)
VS_ALL = frozenset(list(VS_RANGE_1) + list(VS_RANGE_2))

# Namespaces this module's derived byte streams away from encode.py's
# (which keys on the pipeline's config seed/disruption_payload instead).
_VS_SALT = "ghost_tools.inject_vs"


def _inject_vs(text, payload, skip):
    """
    Shared VS-injection core for tool_encode_vs() and tool_combine().

    Appends VS characters after every character in `text` that isn't in
    `skip` (and isn't itself a VS codepoint). The bytes that pick which
    VS codepoints get used come from a per-character pseudorandom stream
    keyed on (payload, char_index) -- see unicode_utils.derive_payload_bytes
    -- NOT from cycling payload's literal UTF-8 bytes by position.

    Cycling a short literal payload gives every character at the same
    (position mod len(payload)) a shared, repeating byte, which is a
    periodic, pattern-matchable signature across the whole text -- the
    exact kind of low-entropy signal that a strong prior (or an attacker
    who's noticed the period) can see through, and strictly weaker than
    the strip_vs ablation this project already evaluates elsewhere.
    Keying per character index means every occurrence gets an
    independent-looking stream, while staying deterministic for the
    same (text, payload) pair -- required so the agent's render()/
    hamming() checks on a given attempt stay reproducible.

    Payload length controls injection strength (how many VS characters
    get appended per character), not which fixed bytes get reused.
    """
    n_vs_per_char = max(1, len(payload))
    result = []
    char_index = 0
    for char in text:
        result.append(char)
        if char not in skip and ord(char) not in VS_ALL:
            per_char_bytes = derive_payload_bytes(
                payload, _VS_SALT, char_index, n_bytes=n_vs_per_char
            )
            result.extend(byte_to_vs(b) for b in per_char_bytes)
            char_index += 1
    return ''.join(result)


def tool_render(encoding: str) -> str:
    """
    Render a bidi-encoded string using the Unicode
    Bidirectional Algorithm.

    Returns only the visible characters — what a human
    reader sees. Control and Variation Selector characters
    are stripped (VS codepoints render as no glyph at all,
    same as a control character, so they must be stripped
    here too or this wouldn't actually return what a human
    sees whenever VS injection is present).

    Use this to verify your encoding displays correctly.

    Args:
        encoding: a string containing bidi control chars,
                  VS characters, and visible content

    Returns:
        The visually rendered string (no control or VS chars)
    """
    # bidi.algorithm.get_display is the legacy pure-Python engine and
    # doesn't support isolate characters (RLI/LRI/PDI, used by the
    # rli_split*/three_block/double_rli configs below) -- it asserts on
    # them. `bidi.get_display` is the real UAX#9 implementation (wraps
    # the Rust unicode-bidi crate), same as bidi_permute.py already uses.
    from bidi import get_display
    result = get_display(encoding)
    return ''.join(c for c in result
                    if c not in ALL_CTRL and ord(c) not in VS_ALL)


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

    Returns -1 if strings have different lengths (invalid).

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

    Appends VS characters after each non-whitespace character in text.
    VS characters are invisible to humans but fragment LLM tokenization.
    Each character gets its own independent, non-repeating VS sequence
    derived from `payload` (see _inject_vs) rather than a literal payload
    cycled by position, so there is no periodic signature across the text.

    Args:
        text: the text to inject VS chars into
        payload: seeds the per-character derived VS byte streams. Use a
                 longer payload for more VS characters per character.
                 Use a different payload to get a different (but still
                 deterministic) injection.

    Returns:
        The text with VS characters injected.
        Renders identically to original for human readers.
    """
    return _inject_vs(text, payload, skip={' ', '\n', '\t', '\r'})


def tool_encode_bidi(text: str, config: str) -> str:
    """
    Apply a bidi encoding configuration to text.

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
        "nested:N:M"     — outer RTL, inner LTR protects [N:M]
        "double_rli:N"   — two independent RTL isolate blocks

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

    return ""  # unknown config


def tool_combine(text: str,
                  bidi_config: str,
                  vs_payload: str) -> str:
    """
    Apply both bidi encoding and VS injection to text.

    Order of operations:
    1. Apply bidi encoding to establish stored sequence
    2. Apply VS injection to characters in stored sequence
       (skipping bidi control characters), via the same
       non-repeating per-character derivation as tool_encode_vs

    This is the full GHOST encoding combining both mechanisms.

    Args:
        text: original text to encode
        bidi_config: config string for tool_encode_bidi
        vs_payload: seeds the per-character derived VS byte streams
                    (see tool_encode_vs)

    Returns:
        Fully GHOST-encoded string, or empty string if
        bidi_config is invalid.
    """
    bidi_encoded = tool_encode_bidi(text, bidi_config)
    if not bidi_encoded:
        return ""

    skip = ALL_CTRL | {' ', '\n', '\t', '\r'}
    return _inject_vs(bidi_encoded, vs_payload, skip=skip)


# ── Tool registry for agent ───────────────────────────────────
# Used in Task 5 to build Anthropic tool definitions.

TOOL_FUNCTIONS = {
    "render": tool_render,
    "get_stored": tool_get_stored,
    "hamming": tool_hamming,
    "encode_vs": tool_encode_vs,
    "encode_bidi": tool_encode_bidi,
    "combine": tool_combine,
}
