# GHOST Paper — Background for Claude Code
# Read this entire document before touching any code.
# Every implementation decision in the experiment plan
# follows from the concepts described here.

---

## What This Paper Is About

GHOST (Generated Hidden Obfuscation STrategy) is a defense
system that prevents large language models from correctly
extracting sensitive numerical content from structured
documents. The core problem: when documents containing
account numbers, invoice amounts, patient IDs, or legal
case numbers are scraped from the web and fed to LLMs,
those models can extract the sensitive field values
accurately. GHOST makes that extraction fail while leaving
the document visually identical to a human reader.

The paper makes two distinct technical contributions:

1. A dual-mechanism obfuscation system (GHOST itself)
   exploiting two separate failure modes in LLM processing
   of Unicode text.

2. An empirical finding about LLM cognition: frontier
   models systematically misreport the visible content of
   bidirectional Unicode-manipulated numerical sequences,
   and this failure is bounded by a property called
   semantic prior strength.

These two contributions are connected. The second explains
WHY the first works, and WHERE it works and does not work.

---

## The Threat Model

The defender is a content owner who publishes documents
online. The adversary is an automated LLM pipeline that
scrapes raw Unicode text and uses it for training or
inference-time field extraction.

Critical assumption: the adversary does NOT apply Unicode
normalization (specifically NFKC) before ingesting text.
Most real scraping libraries (BeautifulSoup, Scrapy,
CommonCrawl, LangChain loaders) preserve raw Unicode
codepoints by default. This assumption is justified
empirically in the paper and is the reason the defense
works at all. A normalization-aware adversary can partially
defeat GHOST; the paper measures this honestly.

The defender's constraint: the published document must
render identically to the original for any human using
a standard Unicode renderer. This is a HARD constraint.
The encoded document cannot look different, have extra
visible characters, or display incorrect values. A human
reading the GHOST-encoded document must see exactly the
same text as the original.

---

## Mechanism 1: Variation Selector Injection

### What Variation Selectors are

Unicode Variation Selectors (VS) are format characters
in two ranges:
  VS1-VS16:   U+FE00 through U+FE0F
  VS17-VS256: U+E0100 through U+E01EF

Their official purpose is to modify glyph rendering of
a preceding character (used for CJK ideographs and emoji
variants). Their key properties for GHOST:

- They produce NO visible glyph on their own
- They render as completely invisible to humans
- They are NOT stripped by common scraping pipelines
- LLM tokenizers ingest them as raw codepoints
- They cause familiar digit tokens to be split into
  rare, low-frequency sub-token sequences

### Why VS injection works

When a digit like "4" appears in training data, the LLM
learns a strong token representation for it. When "4"
is followed by VS characters (e.g., U+FE04, U+E0100),
the tokenizer produces a sequence that almost never
appeared in training data. The model has no learned
representation for "4-followed-by-VS-chars" and cannot
reliably reconstruct that the underlying digit was "4".

For natural language words, this matters less because
the model's prior over word sequences is strong enough
to self-correct. For digit sequences, there is no prior:
"48271039" is no more or less likely than "93017284",
so the model cannot correct through prior-driven inference.
This is the Prior Strength Hypothesis.

### How VS injection is implemented

We use a byte-encoding scheme to map VS characters
systematically. Each byte b of a disruption payload maps
to a specific VS codepoint:

  if b < 16:  chr(0xFE00 + b)      ← VS1-VS16 range
  else:       chr(0xE0100 + b - 16) ← VS17+ range

This gives us 256 distinct VS codepoints (one per byte),
allowing us to encode arbitrary payloads as VS sequences.
For GHOST the payload is a fixed disruption string chosen
to maximise spread across both VS ranges.

### Logprob-guided injection (the key algorithm)

Rather than injecting a fixed number of VS characters
per digit, GHOST uses a proxy model to determine when
enough VS characters have been injected. The procedure:

  For each character in the field value:
    1. Start with the base character (e.g. "4")
    2. Append one VS character
    3. Query proxy model (Qwen2.5-7B) for logprob of
       the ORIGINAL character given the obfuscated version
    4. If logprob <= threshold τ: stop, move to next char
    5. Otherwise: append another VS character, repeat

The proxy model is Qwen2.5-7B-Instruct. It is used ONLY
during encoding, not during evaluation. This model exposes
token-level logprobabilities via HuggingFace transformers,
which is required for the stopping criterion.

The threshold τ needs to be calibrated. Start with -20.0
and run a pilot on 5 documents. Inspect how many iterations
each digit requires. If most digits reach threshold in 3-5
iterations, the threshold is reasonable. If digits routinely
need 15+ iterations, raise the threshold (e.g. -10.0).
If no digits ever reach threshold, lower it (e.g. -30.0).

The transferability claim: obfuscation computed against
Qwen2.5-7B transfers to ALL other models without
re-optimisation. This is because VS fragmentation
produces low-frequency token sequences across ALL model
families, not just Qwen. This is the paper's key
practical claim and must be verified experimentally.

---

## Mechanism 2: Bidirectional Rendering Gap

### The empirical discovery

When numerical content is encoded using Unicode
bidirectional controls, frontier LLMs systematically
misreport what the text says. Specifically:

  Human sees:  1068348   (correctly rendered by bidi algorithm)
  Model says:  8438601   (or some variant — the stored sequence)

This is NOT the model reading a hidden steganographic
channel. The model is misreading the VISIBLE layer —
the layer the human reads. The human and the model are
looking at the same string and seeing different things.

This has been empirically confirmed across GPT-5.6 Sol,
Gemini 3.1 Pro Preview, Grok, and Perplexity. Gemini
sometimes refuses to process the string at all, which
is also a defense success (extraction failure).

### How bidi encoding works

Unicode bidi controls:
  RTL = U+202E  Right-to-Left Override
  PDF = U+202C  Pop Directional Formatting

When a renderer encounters RTL, it reverses the display
order of everything until PDF. So:

  Stored in memory:  RTL + "84382601" + PDF
  Human sees:        "10628348"  (reversed display)

The LLM tokenizer processes raw codepoints in memory
order, not rendered order. So it sees:
  RTL_token, "8", "4", "3", "8", "2", "6", "0", "1", PDF_token

The human sees "10628348" but the model tokenizes the
reversed sequence "84382601" surrounded by control chars.
Without a strong semantic prior to correct from, the model
reports the stored sequence, not the rendered sequence.

### Prior strength as the boundary condition

This failure does NOT affect natural language text.
If you bidi-encode "hello world", frontier models report
"hello world" correctly because their prior over English
word sequences is strong enough to override the corrupted
input. They effectively self-correct.

Digit sequences have no such prior. "48271039" and
"93017284" are equally plausible to the model. No
self-correction is possible. The model reports what
it tokenizes.

This gives us the Prior Strength Hypothesis:
  The effectiveness of Unicode obfuscation is a
  monotonically increasing function of the WEAKNESS
  of the semantic prior. Strong prior → obfuscation
  fails. Weak prior (numerical content) → obfuscation
  succeeds.

The gradient experiment (Phase 6) tests this across
five content types from random digits (weakest prior)
to natural language phrases (strongest prior). The
expected result is a clear gradient where obfuscation
effectiveness decreases as prior strength increases.

---

## The Combined GHOST Algorithm

GHOST applies BOTH mechanisms in sequence:

Step 1: Bidi encoding
  For field value "48271039":
  reversed = "93017284"
  bidi_encoded = RTL + "93017284" + PDF
  Human sees: "48271039" (correct)
  Model tokenizes: RTL, 9, 3, 0, 1, 7, 2, 8, 4, PDF

Step 2: VS injection on the STORED sequence
  For each character in "93017284" (the stored/reversed chars):
  Apply logprob-guided VS injection
  
  After VS injection the stored sequence becomes:
  RTL + 9[VS...] + 3[VS...] + 0[VS...] + ... + PDF

CRITICAL: VS injection applies to the REVERSED characters,
not the original field value. This is because the VS
characters must be adjacent to the characters the model
tokenizes (the stored sequence), not the characters the
human sees (the rendered sequence). Getting this order
wrong would mean VS characters are adjacent to the wrong
digits in the model's token stream.

### Why the combination is stronger than either alone

VS injection alone: model cannot reconstruct individual
digits but may still recover approximate values through
context.

Bidi alone: model tokenizes reversed sequence but may
self-correct or partially recover if field value is
short.

Combined: model receives reversed digits (wrong order)
AND fragmented tokens (wrong identity). Two independent
failure modes. An adversary must:
  1. Strip VS characters
  2. Strip bidi controls
  3. Reverse the exposed stored sequence
  4. Know that all three steps were necessary

Critically, strip-VS-only leaves bidi reversal intact.
Strip-bidi-only leaves VS fragmentation intact. Neither
single-layer attack recovers the original value. Only
strip-both (+ reversal) recovers the original, and
that requires knowing the full encoding scheme.

Note on NFKC: NFKC normalization strips VS characters
BUT PRESERVES bidi controls. So an adversary running
NFKC gets: bidi-encoded field without VS fragmentation.
The model still reads the wrong digit order. This is
NOT the same as stripping both — bidi is still active.
The paper reports both NFKC and strip-both as separate
attack conditions.

---

## The Dataset

We evaluate on 400 synthetic structured documents across
four domains. Documents are short (2-4 sentences) and
contain 2-5 named numerical fields each.

Domains and representative fields:
  Financial:  invoice numbers, account numbers, BSB codes,
              transaction amounts, dates
  Medical:    patient IDs, ICD-10-AM procedure codes,
              dosage values, prescription references
  Legal:      case numbers, TFNs, ABNs, hearing dates
  Technical:  IP addresses, port numbers, serial numbers,
              firmware versions, API key prefixes

These domains were chosen because they represent
realistic high-value extraction targets — the kind of
documents an adversary would actually want to scrape.
They also all contain numerical identifiers with weak
semantic prior, which is where GHOST works.

We also have a 100-instance gradient subset spanning
five content types to test the Prior Strength Hypothesis.
This subset is used ONLY in the gradient experiment
(Phase 6), not in the main evaluation.

---

## The Models

### Proxy model (encoding only)
Qwen2.5-7B-Instruct:
  Used for logprob-guided VS injection stopping criterion.
  Needs HuggingFace access and local GPU deployment.
  Also evaluated as a TARGET model, but its FEA results
  are reported separately (labelled "proxy-optimised")
  and excluded from the transferability analysis.

### Open-source target models (local inference, sequential)
  Llama3.2-3B-Instruct       3B    ~7GB BF16
  Llama-3.1-8B-Instruct      8B    ~18GB BF16
  Qwen2.5-3B-Instruct        3B    ~7GB BF16
  Qwen2.5-7B-Instruct        7B    ~16GB BF16  ← also proxy
  DeepSeek-R1-Distill-Qwen-14B 14B ~32GB BF16  ← reasoning model

  Load one at a time. Unload and clear CUDA cache before
  loading the next. The H100 80GB can handle any of these
  individually with room to spare.

  DeepSeek-R1-Distill-Qwen-14B generates chain-of-thought
  reasoning in <think>...</think> tags before its final
  answer. Strip these tags before exact match evaluation.
  Set max_new_tokens=2048 for this model (not 50).
  Expect 3-5x longer inference time.

### Frontier target models (API inference)
  GPT-5.5              OpenAI     "gpt-5.5"
  GPT-5.6 Sol          OpenAI     "gpt-5.6-sol"
  Claude Sonnet 4.6    Anthropic  "claude-sonnet-4-6"
  Claude Opus 4.6      Anthropic  "claude-opus-4-6"
  Gemini 3.1 Pro       Google     "gemini-3.1-pro-preview"

  Temperature=0 for all. max_tokens=50 (field values short).
  
  Gemini may refuse to process GHOST-encoded strings.
  Record refusals separately — do not score as 0.
  A refusal is an extraction failure and a defense success.
  Report refusal_rate alongside FEA for Gemini.

---

## The Evaluation Task

There is ONE primary task: Field Extraction Accuracy (FEA).

Given an obfuscated document and a field name, the model
is asked to report the exact value of that field.

Prompt template:
  "Read the following document carefully and report the
   exact value of {FIELD_NAME}. Return only the value
   with no additional text or explanation.

   Document:
   {DOCUMENT_TEXT}"

Metric: Exact string match after formatting normalisation.
Normalisation: strip dashes and spaces (for BSB/TFN/ABN).
Partial digit recovery scores 0.

Lower FEA = better defense. The paper reports FEA on
clean documents as baseline (should be near 1.0), then
FEA on each encoding condition.

The gradient experiment uses reconstruction similarity
(cosine similarity via all-MiniLM-L6-v2) as a secondary
metric, because exact match is too strict for natural
language content types.

---

## What Results the Paper Needs

### Table 3 — Main Results
Rows: 10 models
Cols: clean | BAE | TextFooler | HOMOCHAR | VS only |
      Bidi only | GHOST
Values: FEA
Expected: GHOST achieves lowest FEA (best defense) across
all models and all conditions.

### Table 4 — Ablation
Rows: 10 models
Cols: VS only | Bidi only | GHOST
Values: FEA per domain (Financial / Medical / Legal / Technical)
Expected: GHOST < VS only AND GHOST < Bidi only,
confirming the combination is stronger than either alone.

### Table 5 — In-Context Defense
Rows: baseline ghost | after 5 shots | after 10 | after 20
Cols: GPT-5.6 Sol | Claude Opus 4.6
Values: FEA on 20 financial documents
Expected: FEA does not improve substantially with more
examples, showing models cannot learn to denoise GHOST
from in-context demonstrations.

### Table 6 — Normalization Attack
Rows: GHOST | strip VS | strip Bidi | NFKC | strip both
Cols: 10 models
Values: FEA
Expected:
  strip VS: FEA rises somewhat (bidi still confuses model)
  strip Bidi: FEA rises somewhat (VS still fragments tokens)
  NFKC: FEA rises (VS gone, bidi remains)
  strip both: FEA rises most (both layers gone, but
    stored sequence is still reversed — model must know
    to invert for full recovery)

### Figure 1 — Prior Strength Gradient
X-axis: random digits | formatted identifiers |
        alphanumeric codes | rare proper nouns |
        natural language
Y-axis: reconstruction similarity
10 lines (one per model)
Expected: similarity increases monotonically across
content types as prior strength increases, confirming
Hypothesis 1.

---

## Frequently Confused Implementation Details

### VS applies to the STORED (reversed) sequence
When encoding GHOST:
  original field: "48271039"
  reversed (stored): "93017284"
  VS injection is on: "93017284"   ← the stored chars
  NOT on: "48271039"              ← NOT the original

### NFKC is NOT strip_both
  NFKC removes VS characters.
  NFKC does NOT remove bidi control characters.
  strip_both removes both VS and bidi.
  These produce different results and must be measured
  separately in Table 6.

### Ground truth never changes for GHOST, VS, Bidi
  The ground truth field value is always the ORIGINAL
  unencoded value (e.g. "48271039").
  Encoding changes what the model sees, not what is correct.
  Exception: BAE/TextFooler/HOMOCHAR may alter the actual
  field value characters. If so, update GT for that
  instance and flag it.

### Qwen2.5-7B FEA is reported separately
  Qwen2.5-7B is both proxy and target.
  Its FEA is labelled "proxy-optimised" in the tables.
  It is NOT included in the mean transferability figures.
  The transferability claim covers the OTHER 9 models.

### DeepSeek generates <think> tags
  Strip ALL content between <think> and </think> tags
  before running exact match evaluation.
  The final answer appears AFTER the closing </think> tag.
  Apply this stripping in EVERY evaluation path that
  includes deepseek_r1_14b — do not handle it as a
  special case only in some scripts.

### Refusals
  If Gemini (or any model) produces a refusal response
  rather than a field value, record:
    exact_match: 0
    refusal: True
  Compute refusal_rate separately from FEA.
  Report refusal_rate for Gemini in a table footnote.
  Do NOT average refusals into the FEA computation.

---

## File that Centralises Everything

unicode_utils.py contains ALL Unicode encoding/decoding
primitives. No other file should reimplement these
functions. Import from unicode_utils everywhere.

Functions you will use across multiple scripts:
  encode_bidi(field_value)        → bidi-encoded string
  encode_ghost(field, vs_seqs)    → fully ghost-encoded
  strip_vs(text)                  → remove VS characters
  strip_bidi(text)                → remove bidi controls
  strip_all(text)                 → remove both
  apply_nfkc(text)                → NFKC normalisation
  strip_think_tags(text)          → DeepSeek output cleaning
  show_codepoints(text)           → debugging only

---

## The Overarching Claim the Code Must Support

GHOST reduces field extraction accuracy on numerical
content in structured documents to near-chance levels
across 10 frontier and open-source models, while
preserving exact human readability. The defense exploits
two independent failure modes — VS tokenizer fragmentation
and bidi rendering gap — that together resist single-layer
stripping attacks and in-context denoising attempts.

The effectiveness is bounded by semantic prior strength:
GHOST works because numerical identifiers carry no
distributional prior that would allow model self-correction.
Natural language text is explicitly out of scope.

Every table and figure in the paper should support this
claim or honestly characterise where it does not hold.
