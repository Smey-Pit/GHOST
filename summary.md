# GHOST — Summary

## What it does
GHOST is a defense that stops LLMs from correctly extracting sensitive numeric
fields (account numbers, patient IDs, case numbers, etc.) from scraped
documents — while the document still looks completely normal to a human
reader. It works by exploiting two separate weaknesses in how LLMs tokenize
Unicode text, rather than by changing the visible content.

## The algorithm
GHOST combines two mechanisms, applied in sequence to each sensitive field:

1. **Bidirectional reversal.** The field value is reversed and wrapped in
   Unicode right-to-left override characters. A human's renderer displays it
   back in the correct order, but the LLM tokenizes the raw (reversed)
   character sequence — so the model "reads" the digits backwards.

2. **Variation Selector (VS) injection.** Invisible Unicode Variation
   Selector characters are inserted after each digit. These characters
   produce no visible glyph, but they force the tokenizer to split familiar
   digit tokens into rare token sequences the model has never learned to
   interpret. The number of VS characters added per digit is decided
   automatically: a smaller "proxy" model is queried after each addition, and
   injection stops once that model's confidence in the correct digit drops
   below a set threshold — so just enough noise is added, no more.

Applying VS injection to the *already-reversed* digits (not the original
value) is the key step: it means an attacker who strips only one layer
(only the VS characters, or only the reversal) still fails to recover the
original number, because the other layer is still in effect. Only removing
both layers, in the right order, recovers the true value — and that requires
knowing the exact scheme was used.

This works specifically on numbers and identifiers because they have no
learnable pattern the model can use to "guess" past the corruption. Natural
language is not affected in the same way, because the model's language prior
lets it self-correct.

## Status
Confirmed implemented and matches the intended design (`src/unicode_utils.py`,
`src/encode.py`), including the correct order of operations (reverse, then
inject). The codebase also includes a couple of enhancements beyond the
original plan (per-character randomized injection payloads instead of one
fixed payload, and an additional permutation-based variant of the reversal
mechanism), which strengthen the defense against pattern-matching attacks.
