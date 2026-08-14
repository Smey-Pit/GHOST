"""
src/test_normalization_forms.py

Quick, GPU-free check of what NFC/NFD/NFKC/NFKD do to an already-encoded
composed-GHOST string, compared against the strip_vs/strip_bidi/strip_all
ablations. Uses the real composed-GHOST example on hand
(results/raw/verify_gemini_account_number.json, account_number =
5926847003, converged 1 iteration, Hamming 10/10) rather than a synthetic
string.

Pure unicodedata.normalize + unicode_utils primitives -- no model load,
no GPU. Run with: python src/test_normalization_forms.py
"""

import json
from pathlib import Path

from unicode_utils import (
    apply_nfc, apply_nfd, apply_nfkc, apply_nfkd,
    strip_vs, strip_bidi, strip_all,
    show_codepoints,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "results" / "raw" / "verify_gemini_account_number.json"


def describe(label, text, target):
    n_bidi = sum(1 for c in text if c in {'‪', '‫', '‬', '‭', '‮',
                                           '⁦', '⁧', '⁨', '⁩'})
    n_vs = sum(1 for c in text if 0xFE00 <= ord(c) < 0xFE10 or 0xE0100 <= ord(c) < 0xE01F0)
    print(f"--- {label} ---")
    print(f"  length={len(text)}  bidi_controls={n_bidi}  vs_chars={n_vs}")
    print(f"  codepoints: {show_codepoints(text, max_n=20)}")
    print(f"  digits_survive_as_stored_order: {text == target}")
    print()


def main():
    data = json.loads(EXAMPLE_PATH.read_text())
    field = data["field"]
    target = data["target"]
    ghost_encoded = data["encoding"]

    print(f"Field: {field}   Target: {target}")
    print(f"Source: {EXAMPLE_PATH.relative_to(REPO_ROOT)}\n")

    describe("composed GHOST (raw, no attack)", ghost_encoded, target)

    # Existing ablations for reference
    describe("strip_vs (bidi-only survives)", strip_vs(ghost_encoded), target)
    describe("strip_bidi (VS-only survives)", strip_bidi(ghost_encoded), target)
    describe("strip_all / strip_both", strip_all(ghost_encoded), target)

    # The four normalization forms
    describe("NFC", apply_nfc(ghost_encoded), target)
    describe("NFD", apply_nfd(ghost_encoded), target)
    describe("NFKC", apply_nfkc(ghost_encoded), target)
    describe("NFKD", apply_nfkd(ghost_encoded), target)


if __name__ == "__main__":
    main()
