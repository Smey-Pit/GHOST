"""
src/unicode_utils.py

All Unicode encoding/decoding primitives for GHOST.
Import this module everywhere. Never reimplement these functions.
"""

import hashlib
import unicodedata

# -- Keyed pseudorandom payload derivation -----------------------

def derive_payload_bytes(seed, salt, *context, n_bytes=64):
    """
    Deterministic pseudorandom byte stream, unique per (seed, salt, context).

    Used in place of a single fixed disruption payload reused for every
    character: a fixed payload means every VS-injected character across
    the whole corpus shares the exact same byte prefix, which is a
    repeating substring a generic sanitizer can pattern-match and strip
    without even knowing it's GHOST specifically -- strictly weaker than
    the strip_vs ablation the paper already evaluates, since that attack
    doesn't need to know the payload at all.

    Same inputs always reproduce the same bytes (required for
    checkpointing and reproducibility across reruns), but different
    context (e.g. a different character/field/document) gets an
    independent-looking stream, so there is no fixed signature spanning
    the corpus. `context` should be enough to make each character
    occurrence unique, e.g. (doc_id, field_name, char_index).
    """
    key = "|".join(str(c) for c in (seed, salt, *context))
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out += hashlib.sha256(f"{key}#{counter}".encode("utf-8")).digest()
        counter += 1
    return bytes(out[:n_bytes])


# -- Unicode control characters ---------------------------------
RTL = '\u202e'   # Right-to-Left Override
PDF = '\u202c'   # Pop Directional Formatting
BIDI_CHARS = frozenset([
    RTL, PDF,
    '\u202a', '\u202b', '\u202d',
    '\u2066', '\u2067', '\u2068', '\u2069',
])

# -- Variation Selector ranges ------------------------------------
VS_RANGE_1 = range(0xFE00, 0xFE10)    # VS1-VS16
VS_RANGE_2 = range(0xE0100, 0xE01F0)  # VS17-VS256
VS_CODEPOINTS = frozenset(list(VS_RANGE_1) + list(VS_RANGE_2))


def byte_to_vs(byte: int) -> str:
    """Map a single byte (0-255) to a Variation Selector codepoint."""
    if byte < 16:
        return chr(0xFE00 + byte)
    else:
        return chr(0xE0100 + byte - 16)


def encode_bidi(field_value: str) -> str:
    """
    Apply bidi reversal encoding to a field value.
    Stores reversed sequence under RTL override.
    Human renderer displays original order.
    Model tokenizer ingests reversed sequence + control chars.
    """
    return RTL + field_value[::-1] + PDF


def strip_vs(text: str) -> str:
    """Remove all Variation Selector characters from text."""
    return ''.join(c for c in text if ord(c) not in VS_CODEPOINTS)


def strip_bidi(text: str) -> str:
    """Remove all bidi control characters from text."""
    return ''.join(c for c in text if c not in BIDI_CHARS)


def strip_all(text: str) -> str:
    """Remove both VS and bidi control characters."""
    return strip_bidi(strip_vs(text))


def apply_nfkc(text: str) -> str:
    """Apply NFKC normalization (normalization attack)."""
    return unicodedata.normalize('NFKC', text)


def apply_nfc(text: str) -> str:
    """Apply NFC normalization (normalization attack)."""
    return unicodedata.normalize('NFC', text)


def apply_nfd(text: str) -> str:
    """Apply NFD normalization (normalization attack)."""
    return unicodedata.normalize('NFD', text)


def apply_nfkd(text: str) -> str:
    """Apply NFKD normalization (normalization attack)."""
    return unicodedata.normalize('NFKD', text)


def strip_think_tags(text: str) -> str:
    """
    Strip <think>...</think> blocks from DeepSeek-R1 outputs.
    Returns only the final answer after reasoning.

    DeepSeek-R1-Distill's chat template appends the opening "<think>\n"
    itself as part of the generation prompt (confirmed via
    tokenizer.apply_chat_template(..., tokenize=False)) -- it is NOT
    part of the model's generated text. So a raw decoded response
    typically contains only the closing "</think>", with no matching
    opening tag. The original regex required both tags literally
    present and silently left the entire reasoning block (plus
    whatever markdown/commentary follows it) in `cleaned` whenever the
    opening tag was missing -- breaking every downstream JSON/exact-match
    parse that assumed reasoning had been stripped.
    """
    import re
    if '<think>' in text and '</think>' in text:
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        cleaned = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    else:
        cleaned = text
    return cleaned.strip()


def show_codepoints(text: str, max_n: int = 30) -> str:
    """
    Return human-readable codepoint representation for debugging.
    Use this in logs and smoke tests, never in production paths.
    """
    entries = []
    for c in text[:max_n]:
        cp = ord(c)
        if cp in VS_CODEPOINTS:
            label = "VS"
        elif c in BIDI_CHARS:
            label = "BIDI"
        else:
            label = repr(c)
        entries.append(f"U+{cp:05X}[{label}]")
    if len(text) > max_n:
        entries.append(f"...+{len(text) - max_n}")
    return ' '.join(entries)
