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


# ── Optional proxy-model-backed tools (permutation search + real
#    logprob-guided VS injection) ────────────────────────────────────
#
# Unlike every tool above, these need a live language model resident on
# GPU -- src/encode.py's actual production mechanism for both halves of
# GHOST (bidi_permute's searched separable permutation, and
# encode_char_with_vs_logprob's adaptive per-character VS depth) is
# strictly more powerful than the manually-parameterized configs and
# fixed-length payloads the tools above expose, but neither is reachable
# through them. This is a real, structural capability gap: an agent
# reasoning as well as possible still cannot propose an encoding whose
# building blocks were never given to it as tools.
#
# The proxy model is caller-owned, not loaded here -- set_proxy_model()
# must be called (with an encode.ProxyModel-shaped object: a
# .query_logprob(target_char, context) method) before either tool below
# is usable; execute_tool_call's existing try/except turns the
# RuntimeError otherwise raised into a normal "ERROR: ..." tool result,
# so a run that forgets to load the proxy degrades gracefully instead of
# crashing the whole agent loop. GPU-sharing with the search-tier
# ensemble/agent backbone (the release_gpu()/reacquire_gpu() dance
# LocalReActBackbone already does) is NOT yet wired up for the proxy --
# whoever loads it is responsible for not colliding with either, same as
# src/verify_ghost_permute_sentence_track_a.py's own explicit load/
# del/empty_cache lifecycle.

_proxy_model = None

_VS_LOGPROB_SALT = "ghost_tools.inject_vs_logprob"


def set_proxy_model(proxy) -> None:
    """Register a loaded proxy model (encode.ProxyModel-shaped) for
    tool_encode_vs_logprob/tool_combine_permute. Call clear_proxy_model()
    when done so a later run without a proxy fails loudly, not silently
    against a stale reference."""
    global _proxy_model
    _proxy_model = proxy


def clear_proxy_model() -> None:
    global _proxy_model
    _proxy_model = None


def _inject_vs_logprob(text, payload, threshold_tau, skip):
    """
    Shared logprob-guided VS-injection core for tool_encode_vs_logprob()
    and tool_combine_permute() -- mirrors _inject_vs()'s structure
    exactly, but each character's injection depth is now ADAPTIVE (driven
    by a real proxy-model logprob check via
    encode.encode_char_with_vs_logprob) instead of fixed by payload
    length. `payload` still seeds the per-character derived byte stream
    the same way _inject_vs's payload does -- a different payload string
    still gives a different (but reproducible) encoding -- it just no
    longer controls HOW MANY VS characters get appended per character;
    that's now whatever the real model's logprob trajectory demands, up
    to encode_char_with_vs_logprob's own 64-iteration safety cap.
    """
    if _proxy_model is None:
        raise RuntimeError(
            "no proxy model loaded -- call ghost_tools.set_proxy_model() "
            "with a loaded encode.ProxyModel before using this tool"
        )
    from encode import encode_char_with_vs_logprob

    result = []
    char_index = 0
    for char in text:
        if char in skip or ord(char) in VS_ALL:
            result.append(char)
            continue
        char_bytes = derive_payload_bytes(
            payload, _VS_LOGPROB_SALT, char_index, n_bytes=64,
        )
        encoded_char, _iters = encode_char_with_vs_logprob(
            char, char_bytes, _proxy_model, threshold_tau,
        )
        result.append(encoded_char)
        char_index += 1
    return ''.join(result)


def tool_bidi_permute(text: str, trials: int = 2000, seed: int = 0) -> str:
    """
    Search for a separable permutation of text maximally displaced (by
    Hamming distance) from the original, subject to round-trip
    correctness, then encode it via nested bidi isolates/overrides.

    This is src/bidi_permute.py's real searched mechanism -- strictly
    stronger than manually guessing a tool_encode_bidi config, since it
    tries `trials` random candidates and keeps the best one that still
    renders back to the original text. No proxy model needed (pure
    search + a real bidi-rendering check), so this tool is always
    available, unlike the two below.

    Args:
        text: the text to permute (field value or whole sentence)
        trials: number of random candidate permutations to try (more =
                better chance of a highly-displaced result, at more
                compute cost). 2000 is a reasonable default; the
                production pipeline uses 500 for cost reasons.
        seed: seeds the random search. `trials` only samples a fraction
              of the full permutation space, so the starting seed can
              matter as much as the trial budget -- if a search comes
              back with a disappointing Hamming distance, call this
              again with a DIFFERENT seed rather than assuming the
              result is the best available. Defaults to 0 for
              backward compatibility with earlier calls that didn't
              specify one.

    Returns:
        Bidi-encoded string that renders as the original text, using
        the best round-trip-valid permutation found. Empty string if no
        candidate round-tripped within `trials` (rare -- see
        bidi_permute.find_target_permutation's docstring).
    """
    from bidi_permute import decompose, encode as _bidi_permute_encode, find_target_permutation

    _dist, perm = find_target_permutation(text, trials=trials, seed=seed)
    if perm is None:
        return ""
    tree = decompose(tuple(perm))
    return _bidi_permute_encode(tree, text)


def tool_encode_vs_logprob(text: str, payload: str, threshold_tau: float) -> str:
    """
    Inject Variation Selector characters into text, with each
    character's injection depth chosen ADAPTIVELY by querying a real
    proxy model's logprob for the true character after each VS byte
    appended -- stopping once that logprob drops below threshold_tau (or
    a 64-iteration safety cap). This is src/encode.py's actual
    production VS mechanism (what the paper's `ghost`/`ghost_permute`
    conditions measure), not the fixed-length approximation
    tool_encode_vs provides.

    REQUIRES set_proxy_model() to have been called first -- raises if
    not (caught by execute_tool_call, surfaced as a normal tool error).

    Args:
        text: the text to inject VS chars into
        payload: seeds the per-character derived byte stream (same role
                 as tool_encode_vs's payload) -- does NOT control how
                 many VS characters get appended anymore; that's now
                 adaptive per character.
        threshold_tau: stop injecting once the proxy's logprob for the
                 true character drops to or below this value (more
                 negative = harder to satisfy = more VS characters
                 typically needed). The calibrated production value is
                 -8.0 (config.yaml's threshold_tau).

    Returns:
        The text with adaptively-injected VS characters. Renders
        identically to the original for human readers.
    """
    return _inject_vs_logprob(
        text, payload, threshold_tau, skip={' ', '\n', '\t', '\r'},
    )


def tool_combine_permute(text: str, trials: int, payload: str,
                          threshold_tau: float, seed: int = 0) -> str:
    """
    Apply BOTH the searched separable-permutation bidi mechanism
    (tool_bidi_permute) AND real logprob-guided VS injection
    (tool_encode_vs_logprob) -- the strongest available combination,
    matching src/encode.py's production encode_ghost_permute/
    encode_ghost_permute_sentence mechanism, instead of `combine`'s
    manually-specified config + fixed-length payload.

    REQUIRES set_proxy_model() to have been called first.

    Order of operations (same rule as `combine`): bidi permutation is
    applied FIRST to establish the stored sequence, THEN VS injection
    runs on that already-permuted sequence (skipping bidi control
    characters) -- never the other way around.

    Args:
        text: original text to encode
        trials: passed to tool_bidi_permute
        payload: seeds the per-character derived VS byte streams (see
                 tool_encode_vs_logprob)
        threshold_tau: logprob stopping threshold (see
                 tool_encode_vs_logprob)
        seed: passed to tool_bidi_permute -- try a different seed if a
              previous attempt's Hamming distance was disappointing for
              this trial budget.

    Returns:
        Fully searched-permutation + logprob-guided-VS encoded string,
        or empty string if the permutation search found no round-trip-
        valid candidate.
    """
    bidi_encoded = tool_bidi_permute(text, trials=trials, seed=seed)
    if not bidi_encoded:
        return ""
    skip = ALL_CTRL | {' ', '\n', '\t', '\r'}
    return _inject_vs_logprob(bidi_encoded, payload, threshold_tau, skip=skip)


_wider_context = None


def set_wider_context(text) -> None:
    """Register a wider text (e.g. the full sentence containing the
    current field) that tool_widen_scope() can hand to the agent when
    its current target isn't defensible on its own. Caller-owned, same
    registration pattern as set_proxy_model() -- call clear_wider_context()
    when the run ends so a later run without one fails loudly, not
    silently against a stale reference."""
    global _wider_context
    _wider_context = text


def clear_wider_context() -> None:
    global _wider_context
    _wider_context = None


def tool_widen_scope() -> str:
    """
    Request a WIDER piece of text to obfuscate than your current target
    (e.g. the full sentence containing the field, instead of just the
    field's bare value) -- for when your best attempt on the current
    target still can't clear a full-consensus margin against the panel.

    Only use this AFTER at least one combine_permute/combine attempt,
    including a seed retry, has already failed to reach full consensus
    on the current target -- widening trades a much larger encoded
    payload for more obfuscation surface, it is not a free first move.

    Returns the wider text if the caller supplied one for this run, or
    an ERROR string if none is available (in which case keep working
    with the current target -- widening isn't possible this run).

    IMPORTANT: after calling this, build your next attempt by calling
    combine_permute/combine on the TEXT THIS TOOL RETURNS, not on your
    original target -- and report that new encoding as your
    <final_encoding>.
    """
    if _wider_context is None:
        return (
            "ERROR: no wider context available for this field -- "
            "keep working with the current target"
        )
    return _wider_context


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
    "bidi_permute": tool_bidi_permute,
    "encode_vs_logprob": tool_encode_vs_logprob,
    "combine_permute": tool_combine_permute,
    "widen_scope": tool_widen_scope,
}
