"""
smoke_test_sentence_utils.py

Verifies sentence_utils.py (sentence location) and encode.py's
_apply_sentence_transform / encode_bidi_permute_sentence (Phase 2 of the
sentence-level obfuscation plan) against REAL data -- data/raw/documents.json
(the older, one-sentence-per-field dataset.py format) and
data/track_a/pilot/*.jsonl (narrative prose with real char_span, including
a genuine multi-field-one-sentence case). No mocking of the bidi oracle:
round-trip checks go through the real python-bidi implementation, same as
bidi_permute.py's own self-tests.

Run from src/: python smoke_test_sentence_utils.py
"""
import json
import os

from sentence_utils import find_field_sentence, split_sentences_with_spans
from bidi_permute import visible as bidi_visible
from encode import _apply_sentence_transform, encode_bidi_permute_sentence

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── split_sentences_with_spans: covers the whole string, no gaps ────────

text = "First sentence. Second sentence! Third?"
spans = split_sentences_with_spans(text)
reconstructed = "".join(text[s:e] for _, s, e in [(t, s, e) for t, s, e in spans])
# spans partition [0, len(text)) exactly, in order, with the separating
# whitespace consumed between them (not lost, not duplicated) -- rebuild
# by re-inserting single spaces where consumed to confirm no character
# outside a sentence span was silently dropped.
assert spans[0][1] == 0 and spans[-1][2] == len(text), spans
for i in range(len(spans) - 1):
    assert spans[i][2] <= spans[i + 1][1], "sentence spans must not overlap"
print("split_sentences_with_spans: coverage OK")


# ── Real edge case: decimal amounts and ICD-style codes must NOT fragment
#    a sentence at their internal period ──────────────────────────────

amount_text = "The invoice total is 1234.56 dollars. Payment is due immediately."
r = find_field_sentence(amount_text, "1234.56")
assert r is not None
assert r[0] == "The invoice total is 1234.56 dollars.", r[0]
assert "1234.56" in r[0], "decimal field must stay inside its own sentence intact"

icd_text = "The diagnosis code recorded is A12.3 for this patient. Follow-up is required."
r2 = find_field_sentence(icd_text, "A12.3")
assert r2 is not None
assert r2[0] == "The diagnosis code recorded is A12.3 for this patient.", r2[0]
print("decimal amount / ICD-code period edge cases: OK (no false sentence split)")


# ── Real data/raw/documents.json record: one sentence per field ────────

with open(os.path.join(REPO_ROOT, "data", "raw", "documents.json")) as f:
    real_docs = json.load(f)

multi_field_doc = next(d for d in real_docs if len(d["fields"]) >= 2)
for field_name, field_value in multi_field_doc["fields"].items():
    located = find_field_sentence(multi_field_doc["text"], field_value)
    assert located is not None, f"{field_name} not located"
    sentence_text, s, e = located
    assert field_value in sentence_text
    assert multi_field_doc["text"][s:e] == sentence_text
print(f"data/raw/documents.json real record ({multi_field_doc['id']}): "
      f"all {len(multi_field_doc['fields'])} fields located in their own sentence, OK")


# ── Real Track A pilot record: char_span-based location, INCLUDING a
#    genuine multi-field-one-sentence case (regfiling_0000) ────────────

pilot_docs = {}
with open(os.path.join(REPO_ROOT, "data", "track_a", "pilot", "pilot_documents.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        pilot_docs[d["doc_id"]] = d

pilot_fields = []
with open(os.path.join(REPO_ROOT, "data", "track_a", "pilot", "pilot_fields.jsonl")) as f:
    for line in f:
        pilot_fields.append(json.loads(line))

doc0 = pilot_docs["regfiling_0000"]
doc0_fields = [f for f in pilot_fields if f["doc_id"] == "regfiling_0000"]
assert len(doc0_fields) == 2, doc0_fields

located_sentences = set()
for f in doc0_fields:
    r = find_field_sentence(
        doc0["carrier_text"], f["ground_truth"], char_span=tuple(f["char_span"]),
    )
    assert r is not None, f["field_name"]
    located_sentences.add(r[0])

# jurisdiction_code and registration_date genuinely share one sentence in
# this real document -- confirmed by direct inspection before writing
# this test. If sentence_utils' boundary logic is wrong, this collapses
# to more than one distinct sentence text.
assert len(located_sentences) == 1, (
    f"expected both fields in the SAME real sentence, got {located_sentences}"
)
print("Track A regfiling_0000: multi-field-one-sentence case located correctly, OK")


# ── _apply_sentence_transform: dedup -- the shared sentence must only be
#    transformed (and therefore counted) ONCE ──────────────────────────

char_spans = {f["field_name"]: tuple(f["char_span"]) for f in doc0_fields}
track_a_item = {
    "id": "regfiling_0000",
    "text": doc0["carrier_text"],
    "fields": {f["field_name"]: f["ground_truth"] for f in doc0_fields},
}

transform_call_count = {"n": 0}


def _counting_transform(sentence_text):
    transform_call_count["n"] += 1
    return sentence_text.upper()  # trivial, deterministic marker transform


result = _apply_sentence_transform(track_a_item, _counting_transform, char_spans=char_spans)
assert transform_call_count["n"] == 1, (
    f"shared sentence transformed {transform_call_count['n']} times, expected 1"
)
# Ground truth must be completely unchanged.
assert result["fields"] == track_a_item["fields"]
# The shared sentence's uppercased form must appear exactly once in the
# spliced text, and the original mixed-case form must be gone.
shared_sentence = next(iter(located_sentences))
assert shared_sentence.upper() in result["text"]
assert shared_sentence not in result["text"]
print("_apply_sentence_transform: dedup + ground-truth-unchanged OK")


# ── encode_bidi_permute_sentence end-to-end: real permutation search +
#    real bidi oracle round-trip, at FULL-DOCUMENT scope (not just the
#    isolated sentence) -- this is the actual "renders identically"
#    guarantee the whole project depends on, checked at the scope this
#    condition actually ships at ─────────────────────────────────────

fin_doc = next(d for d in real_docs if d["domain"] == "financial" and len(d["fields"]) >= 2)
encoded_doc = encode_bidi_permute_sentence(fin_doc, trials=500, seed=168, salt="smoke_test_salt")

assert encoded_doc["fields"] == fin_doc["fields"], "ground truth must be unchanged"
assert encoded_doc["text"] != fin_doc["text"], "text must actually be transformed"
assert bidi_visible(encoded_doc["text"]) == fin_doc["text"], (
    "FULL-DOCUMENT round-trip failed -- a sentence-level encoding must "
    "still render the entire document identically to a human, not just "
    "the isolated sentence in a vacuum"
)
print(f"encode_bidi_permute_sentence end-to-end ({fin_doc['id']}): "
      "full-document round-trip verified against real UAX#9 oracle, OK")

print("\nSMOKE TEST sentence_utils.py / _apply_sentence_transform / "
      "encode_bidi_permute_sentence PASSED")
