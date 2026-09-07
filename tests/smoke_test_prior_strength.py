"""
tests/smoke_test_prior_strength.py

No GPU/API needed. Validates the actual math in src/prior_strength.py
(not just plumbing) against deterministic fake tokenizer/model objects:
  - compressed_bits: real zlib, no mocking needed.
  - _per_token_bits / score_target_nll_bits: a "confident" fake model
    (near-certain on the true next token everywhere) must score near-
    zero bits; a "uniform" fake model (no information) must score
    ~log2(vocab_size) bits per token.
  - score_min_k_bits: a model confident everywhere except one specific
    target position must have its spike surface in min_k_bits even
    though the mean over all tokens would mostly hide it.
  - score_prior_strength: ratio ordering (confident < uniform for the
    same target/context) and required keys.
  - _per_token_bits: empty context raises ValueError (there's no anchor
    position to predict the target's first token from).
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import prior_strength  # noqa: E402

VOCAB_SIZE = 256


class FakeTokenizer:
    """Char-level tokenizer: one token per character, id = ord(c) % VOCAB_SIZE."""

    def __call__(self, text, return_tensors="pt"):
        ids = [ord(c) % VOCAB_SIZE for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class ConfidentFakeModel:
    """Puts near-all probability mass on the actual next token, everywhere."""

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, VOCAB_SIZE)
        for pos in range(seq_len - 1):
            correct_next = input_ids[0, pos + 1].item()
            logits[0, pos, correct_next] = 50.0
        return SimpleNamespace(logits=logits)


class UniformFakeModel:
    """No information: uniform distribution over the vocabulary everywhere."""

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, VOCAB_SIZE)
        return SimpleNamespace(logits=logits)


class SpikeFakeModel:
    """Confident everywhere except at one absolute sequence position."""

    def __init__(self, spike_abs_position):
        self.spike_abs_position = spike_abs_position

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, VOCAB_SIZE)
        for pos in range(seq_len - 1):
            if pos + 1 == self.spike_abs_position:
                continue  # leave uniform -> high surprisal predicting this token
            correct_next = input_ids[0, pos + 1].item()
            logits[0, pos, correct_next] = 50.0
        return SimpleNamespace(logits=logits)


def main():
    tokenizer = FakeTokenizer()
    context = "Account Number: "
    target = "123456789"

    # -- compressed_bits: pure zlib, deterministic --
    repetitive = "aaaaaaaaaaaaaaaaaaaa"
    random_ish = "q7Zx9!kLp2@wR4vN8mYs"
    compressed = prior_strength.compressed_bits(repetitive)
    assert compressed > 0
    assert prior_strength.compressed_bits(random_ish) > compressed, (
        "a repetitive string must compress to fewer bits than a "
        "high-entropy string of the same length"
    )
    assert prior_strength.compressed_bits("") == 0.0
    print("compressed_bits: repetitive < high-entropy, empty=0 OK")

    # -- confident model: near-zero total surprisal --
    confident_bits = prior_strength.score_target_nll_bits(
        tokenizer, ConfidentFakeModel(), target, context
    )
    assert confident_bits < 1.0, (
        f"confident model should score near-zero total bits, got {confident_bits}"
    )
    print(f"score_target_nll_bits (confident model): {confident_bits:.4f} bits OK")

    # -- uniform model: ~log2(vocab_size) bits PER TOKEN --
    uniform_bits = prior_strength.score_target_nll_bits(
        tokenizer, UniformFakeModel(), target, context
    )
    expected_per_token = torch.log2(torch.tensor(float(VOCAB_SIZE))).item()
    expected_total = expected_per_token * len(target)
    assert abs(uniform_bits - expected_total) < 1e-3, (
        f"uniform model should score ~{expected_total:.4f} bits, got {uniform_bits}"
    )
    print(f"score_target_nll_bits (uniform model): {uniform_bits:.4f} bits "
          f"(expected {expected_total:.4f}) OK")

    assert confident_bits < uniform_bits, (
        "a model that knows the continuation must score fewer bits than "
        "one with no information about it"
    )
    print("confident < uniform total surprisal OK")

    # -- score_prior_strength: ratio ordering + required keys --
    confident_profile = prior_strength.score_prior_strength(
        tokenizer, ConfidentFakeModel(), target, context
    )
    uniform_profile = prior_strength.score_prior_strength(
        tokenizer, UniformFakeModel(), target, context
    )
    for profile in (confident_profile, uniform_profile):
        for key in ("nll_bits", "compressed_bits", "ratio", "min_k_bits"):
            assert key in profile, f"missing key {key!r} in score_prior_strength output"
    assert confident_profile["ratio"] < uniform_profile["ratio"], (
        "a model that already 'knows' the target must have a lower "
        "prior-strength ratio than one with no information about it, "
        "for the SAME target/context (same compressed_bits denominator)"
    )
    print(
        f"score_prior_strength ratios: confident={confident_profile['ratio']:.4f} "
        f"< uniform={uniform_profile['ratio']:.4f} OK"
    )

    # -- score_min_k_bits: a single spike must surface even though the --
    # -- mean over all (mostly confident) tokens would mostly hide it. --
    n_context_tokens = len(context)  # char-level fake tokenizer: 1 token/char
    spike_target_index = 3  # 4th character of target
    spike_abs_position = n_context_tokens + spike_target_index
    spike_model = SpikeFakeModel(spike_abs_position)

    all_bits = prior_strength._per_token_bits(tokenizer, spike_model, target, context)
    mean_bits = sum(all_bits) / len(all_bits)
    min_k_bits = prior_strength.score_min_k_bits(tokenizer, spike_model, target, context, k=0.2)
    assert min_k_bits > mean_bits, (
        f"min_k_bits ({min_k_bits:.4f}) should exceed the plain mean "
        f"({mean_bits:.4f}) when the spike is isolated to one token"
    )
    assert all_bits[spike_target_index] == max(all_bits), (
        "the spike position should be the single highest-surprisal token"
    )
    print(f"score_min_k_bits: spike surfaced (min_k={min_k_bits:.4f} > "
          f"mean={mean_bits:.4f}) OK")

    # -- empty context must raise, not silently misbehave --
    try:
        prior_strength._per_token_bits(tokenizer, ConfidentFakeModel(), target, "")
        raise AssertionError("expected ValueError for empty context")
    except ValueError:
        print("_per_token_bits: empty context raises ValueError OK")

    print("SMOKE TEST prior_strength.py PASSED")


if __name__ == "__main__":
    main()
