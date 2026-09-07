"""
src/prior_strength.py

Scores how much of a target string an LLM can recover from its own
prior (context + training-time familiarity) rather than from actually
reading its characters -- the "semantic prior strength" signal GHOST-
Agent needs to eventually learn things like "a brand name needs more
Hamming/VS depth than a bank account number of the same length."

WHY NOT PER-CHARACTER AVERAGE LOGPROB: averaging surprisal per
character/token erases exactly the signal we want. A long, memorized
passage (a famous lyric) should score as LOW total surprisal despite
its length -- the model already "has" the whole thing. A short random
account number should score HIGH despite being short -- the model has
no prior pull toward those specific digits. Total (summed) surprisal is
the right unit for "bits the model needs to be told," not a per-unit
rate.

WHY NOT RAW TOTAL SURPRISAL ALONE: it still conflates "short" with
"predictable," and conflates "genuinely memorized" with "merely fluent"
(ordinary grammatical English is low-perplexity to an LLM regardless of
whether any specific passage is memorized). The fix, following the
training-data-extraction / membership-inference literature (Carlini et
al., "Extracting Training Data from Large Language Models"; Shi et al.,
"Min-K% Prob"), is to normalize the model's surprisal against a generic,
model-agnostic, tokenizer-agnostic reference compressor (zlib) computed
on the same raw string:

    ratio = model_NLL_bits(target | context) / zlib_compressed_bits(target)

ratio near 0: the model needs far fewer bits than a generic compressor
would -- it already "knows" this beyond what's explained by the string
being short or compressible (memorized/familiar; e.g. a well-known
brand name, a famous quote).
ratio near 1: the model has no edge over a generic compressor -- opaque/
arbitrary content (e.g. a bank account number, a random alphanumeric
ID).

This ratio is comparable across wildly different lengths and domains
(a 6-character field vs. a 150-character excerpt), which per-character
average logprob is not, and it is the same quantity for every ensemble
member regardless of tokenizer family, which raw per-token logprob is
not.

Computed ONCE per field, on the RAW (unencoded) target -- this
characterizes the content itself, not any particular encoding attempt.
See src/ensemble.py's run_ensemble_prior_strength for the load-once/
query-many/unload wrapper that runs this across the search-tier
ensemble's tokenizer-diverse members and averages the ratio.
"""

import math
import zlib

_LN2 = math.log(2)


def compressed_bits(text: str) -> float:
    """
    Reference baseline: bits needed by a generic, model-agnostic
    compressor to represent `text`. zlib operates on raw UTF-8 bytes,
    not model tokens, so this is comparable across tokenizer families
    and independent of any model's training data.
    """
    if not text:
        return 0.0
    return len(zlib.compress(text.encode("utf-8"))) * 8


def _per_token_bits(tokenizer, model, target: str, context: str) -> list:
    """
    Teacher-forced per-token surprisal (in bits) of `target`'s own
    tokens, conditioned on `context` and (for tokens after the first)
    the preceding target tokens -- one forward pass, no sampling.
    Generalizes encode.py's ProxyModel.query_logprob (single next-token
    logprob) to a multi-token continuation.

    KNOWN APPROXIMATION: `context` and `context + target` are tokenized
    separately to locate the target's token span; because tokenization
    is not always separable at that boundary (a subword can merge
    across it), the token split is an approximation, not a guaranteed
    exact alignment. This mirrors the same kind of string-concatenation
    approximation encode.py's char-level logprob queries already make.

    Args:
        tokenizer: a loaded HF tokenizer.
        model: a loaded HF causal LM on some device.
        target: the string being scored (e.g. a field value or sentence).
        context: non-empty preceding text (e.g. "Account Number: ", or
            the sentence/document text preceding the target). Required
            non-empty so there is always at least one real anchor
            position to predict the target's first token from.

    Returns:
        list of per-target-token surprisal values in bits, in order.
    """
    import torch

    if not context:
        raise ValueError(
            "context must be non-empty -- it anchors the first "
            "predicted position; pass at least a field label or "
            "leading document text."
        )

    context_ids = tokenizer(context, return_tensors="pt").input_ids
    full_ids = tokenizer(context + target, return_tensors="pt").input_ids
    n_context_tokens = context_ids.shape[1]
    n_total_tokens = full_ids.shape[1]
    if n_total_tokens <= n_context_tokens:
        # target contributed no distinguishable tokens once merged with
        # context (degenerate/empty target) -- nothing to score.
        return []

    device = next(model.parameters()).device
    full_ids = full_ids.to(device)

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

    bits = []
    for t in range(n_context_tokens, n_total_tokens):
        token_id = full_ids[0, t].item()
        logprob_nats = log_probs[t - 1, token_id].item()
        bits.append(-logprob_nats / _LN2)
    return bits


def score_target_nll_bits(tokenizer, model, target: str, context: str) -> float:
    """Total (summed) surprisal of `target` given `context`, in bits."""
    return sum(_per_token_bits(tokenizer, model, target, context))


def score_min_k_bits(
    tokenizer, model, target: str, context: str, k: float = 0.2,
) -> float:
    """
    Mean surprisal of the least-predictable k fraction of target
    tokens (Shi et al.'s Min-K% Prob idea) -- catches a passage that is
    mostly fluent/generic but contains one or two specific,
    hard-to-guess tokens (a real date, a named entity), which a plain
    mean over all tokens can wash out. k=0.2 means the least predictable
    20% of tokens.
    """
    per_token = _per_token_bits(tokenizer, model, target, context)
    if not per_token:
        return 0.0
    per_token_sorted = sorted(per_token, reverse=True)  # highest surprisal first
    n_k = max(1, round(len(per_token_sorted) * k))
    return sum(per_token_sorted[:n_k]) / n_k


def score_prior_strength(
    tokenizer, model, target: str, context: str, min_k: float = 0.2,
) -> dict:
    """
    Full prior-strength profile for one (target, context, model) triple.

    Returns:
        dict with keys:
          nll_bits: total model surprisal of target given context (bits)
          compressed_bits: zlib reference baseline (bits)
          ratio: nll_bits / compressed_bits -- the core, length- and
              tokenizer-comparable prior-strength score. Near 0 = model
              already knows this beyond generic compressibility
              (memorized/familiar). Near 1 = no edge over a generic
              compressor (opaque/arbitrary).
          min_k_bits: mean surprisal of the least-predictable min_k
              fraction of target tokens (see score_min_k_bits).
    """
    per_token = _per_token_bits(tokenizer, model, target, context)
    nll_bits = sum(per_token)
    ref_bits = compressed_bits(target)
    ratio = (nll_bits / ref_bits) if ref_bits > 0 else float("inf")

    if per_token:
        per_token_sorted = sorted(per_token, reverse=True)
        n_k = max(1, round(len(per_token_sorted) * min_k))
        min_k_bits = sum(per_token_sorted[:n_k]) / n_k
    else:
        min_k_bits = 0.0

    return {
        "nll_bits": nll_bits,
        "compressed_bits": ref_bits,
        "ratio": ratio,
        "min_k_bits": min_k_bits,
    }
