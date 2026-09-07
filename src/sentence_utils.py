"""
src/sentence_utils.py

Sentence-boundary location for sentence-level GHOST obfuscation (see the
sentence-level obfuscation plan, Phase 2). Every existing GHOST condition
(bidi_only/vs_only/ghost/bidi_permute/ghost_permute, src/encode.py's
_apply_field_transform) obfuscates only the isolated field value; this
module locates the SENTENCE containing a field so encode.py can obfuscate
that instead, removing the surrounding contextual scaffolding (field
labels, expected format, prose) an adversary could otherwise use to
reconstruct a garbled value.

Two dataset shapes are supported by one function:
  - data/raw/documents.json (src/dataset.py): text = " ".join(sentences),
    one phrase-template sentence per field. No char_span available --
    the field is located via text.find(field_value), same approach
    _apply_field_transform already uses for splicing.
  - data/track_a/*/pilot_fields.jsonl (LLM-generated narrative prose):
    an exact char_span per field is already known and MUST be used
    instead of text.find -- a field value can repeat elsewhere in a
    longer document, and text.find would silently pick the wrong
    occurrence.
"""
import re

# Split on whitespace immediately following a sentence-ending punctuation
# mark. Deliberately NOT triggered by [.!?] alone -- real sentence
# boundaries in this project's documents always have a space (or end of
# string) after the terminal punctuation, while an in-field decimal
# ("1234.56") or code-style period ("A12.3", dataset.py's _gen_icd_code)
# never does: there is no whitespace between the digits on either side.
# This sidesteps needing digit-adjacency lookarounds entirely -- verified
# directly against both real generators' value shapes in
# smoke_test_sentence_utils.py.
#
# ALSO split on any run of newlines, independent of preceding punctuation
# (real bug found 2026-08-19: on tabular/semi-structured Track A documents
# like "Docket Number: DK-2023-12345\nCase Filing Number: ...\n..." with
# no sentence-ending punctuation anywhere nearby -- or in one real case,
# anywhere in the ENTIRE document except its final line -- the old regex
# never found a boundary near the field at all, so find_field_sentence
# returned the whole multi-hundred-character document as "the sentence."
# A tabular key:value line is a natural, human-perceived unit on its own,
# same as a real sentence is -- splitting on newlines treats it as one.
# Confirmed harmless for the two existing supported shapes: data/raw/
# documents.json's text has no newlines at all (space-joined sentences),
# and the real Track A prose doc smoke_test_sentence_utils.py's
# multi-field-one-sentence case depends on (regfiling_0000, pilot split)
# is a single unbroken paragraph with no newlines either.
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def split_sentences_with_spans(text):
    """
    Split text into sentences, returning [(sentence_text, start, end), ...]
    covering the whole string (no gaps, no dropped characters -- the
    final sentence's end is always len(text)).
    """
    spans = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        spans.append((text[start:m.start()], start, m.start()))
        start = m.end()
    spans.append((text[start:], start, len(text)))
    return spans


def find_field_sentence(text, field_value, char_span=None):
    """
    Locate the sentence containing a field's value.

    Args:
        text: full document/carrier text
        field_value: the field's value string (used for text.find only
            when char_span is not given)
        char_span: optional (start, end) character offsets already known
            for this field (Track A's pilot_fields.jsonl format). ALWAYS
            prefer this over letting the function re-derive the location
            via text.find -- a repeated field value elsewhere in a longer
            document would make text.find silently pick the wrong
            occurrence.

    Returns:
        (sentence_text, sentence_start, sentence_end), or None if the
        field/span cannot be located in text at all (e.g. field_value is
        empty, or genuinely absent from text).
    """
    if char_span is not None:
        f_start, f_end = char_span
    else:
        if not field_value:
            return None
        f_start = text.find(field_value)
        if f_start == -1:
            return None
        f_end = f_start + len(field_value)

    for sentence_text, s_start, s_end in split_sentences_with_spans(text):
        if s_start <= f_start and f_end <= s_end:
            return sentence_text, s_start, s_end
    return None
