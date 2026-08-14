# GHOST-Agent: A Self-Improving Unicode Obfuscation Agent for Defeating LLM Text Extraction

*Draft — Methodology and System Design*

---

## 1. Motivation

The original GHOST defense combines two independent Unicode-level failure
modes — bidirectional override reversal and logprob-guided Variation
Selector (VS) injection — into a fixed, calibrated encoding algorithm. A
proxy model's token logprobs determine when enough VS injection has been
applied; a single global threshold, calibrated once on a held-out split,
is then applied uniformly to every field in the corpus.

This works well for short, low-entropy numeric fields (account numbers,
case IDs), where the space of plausible encodings is small and a fixed
recipe generalizes. It is a poor fit for two things the static pipeline
was never designed to handle:

1. **Natural-language content.** A frontier model's prior over fluent
   text is far stronger than its prior over a 10-digit account number.
   The right amount and shape of injection for a sentence is not the
   right amount for a phone number, and a single calibrated threshold
   cannot express that difference.
2. **A moving target.** The logprob threshold is a *proxy* for the thing
   we actually care about — whether a real adversary model can extract
   the protected value — not a direct measurement of it. A configuration
   that clears the proxy's threshold can still fail against a real
   extraction attempt, and vice versa.

The Prior Strength Hypothesis that motivates the static pipeline's
numeric/text framing is not, on inspection, a hard content-type wall. It
describes what a *fixed, low-complexity* injection strategy can defeat.
A sufficiently complex, non-repetitive encoding can defeat reconstruction
on long, natural-language text as well as on numbers — what determines
success is injection complexity relative to the target's prior strength,
not the target's content type. The obstacle is that "sufficiently
complex" is not something a single global threshold, calibrated once, can
express per-instance.

GHOST-Agent replaces the calibrated-threshold pipeline with a search
procedure: an LLM agent iteratively generates an encoding, queries a real
adversary model to test it, reflects on why the attempt succeeded or
failed, and proposes an improved encoding — repeating until extraction
fails or a budget is exhausted. The feedback signal is real extraction
failure, not a logprob surrogate, which removes the proxy-model dependency
from the core defense loop and lets the search adapt its strategy to
whatever content type and adversary it is actually facing.

## 2. Threat Model

GHOST-Agent inherits the base GHOST threat model unchanged. The adversary
scrapes raw Unicode text (BeautifulSoup/Scrapy/CommonCrawl/LangChain-style
loaders) and does **not** apply NFKC normalization before feeding it to an
LLM for field extraction or reconstruction. This assumption is
load-bearing: a normalization-aware adversary partially defeats both the
static pipeline and the agent, and that gap is measured honestly rather
than assumed away.

The hard constraint on the defender's side is unchanged and non-negotiable:
the encoded document must render identically to the original for any
standard Unicode renderer. GHOST-Agent's search space is restricted to
encodings that satisfy this constraint by construction — every candidate
encoding is verified to render back to the original text before it is
ever sent to an adversary model (Section 4).

What changes relative to the static pipeline is the adversary's role at
*encoding time*. The static pipeline needs only a proxy model's logprobs;
GHOST-Agent needs a model that can answer "can you extract this value?"
Two distinct adversary roles follow from this:

- **Search-tier adversary.** A cheap, local, multi-model ensemble used
  during the iterative search loop, so that thousands of candidate
  encodings can be tested without frontier-model API cost. This is a cost
  proxy, analogous in spirit to the static pipeline's Qwen logprob proxy
  — its verdicts are never the paper's reported defense numbers.
- **Verify-tier adversary.** Real frontier models (Claude, GPT, Gemini),
  queried once per converged encoding, whose extraction accuracy against
  the search-tier's "defended" output is the number that actually matters
  for the paper's claims. The gap between what the cheap search proxy
  reports as defended and what a real frontier model can still extract is
  itself a first-class measurement, not an afterthought — an inflated
  search-tier success rate that does not transfer to the verify tier would
  indicate the ensemble is too weak a stand-in for a real adversary, the
  same failure mode a miscalibrated logprob threshold could produce in the
  static pipeline.

The agent role (the model proposing encodings) and the adversary role
(the model attacked by them) are always separate models, in both tiers.
The reasoning agent is never used as its own adversary in a reported
result.

## 3. System Architecture

GHOST-Agent has three moving parts: a fixed set of **encoding tool
primitives** the agent can call, an **agent loop** that wires those tools
to a reasoning model and an adversary, and a **persistent memory** layer
that lets experience accumulate across documents rather than resetting
each time the process restarts.

```
                    ┌─────────────────────────────┐
                    │      Persistent Memory       │
                    │  (strategy_memory.json,      │
                    │   experience_log.jsonl)      │
                    └───────────┬─────────────────┘
                    load          │        save
                    principles    ▼        principle
                    ┌─────────────────────────────┐
   new field  ─────▶│         Agent Loop           │
   value            │  generate → query → reflect  │
                    │        → improve              │
                    └───────────┬─────────────────┘
                                │ tool calls
                                ▼
                    ┌─────────────────────────────┐
                    │   Encoding Tool Primitives    │
                    │  render / get_stored /        │
                    │  hamming / encode_bidi /      │
                    │  encode_vs / combine          │
                    └───────────┬─────────────────┘
                                │ candidate encoding
                                ▼
                    ┌─────────────────────────────┐
                    │   Adversary (two-tier)        │
                    │  search: local ensemble       │
                    │  verify: frontier models      │
                    └─────────────────────────────┘
```

### 3.1 Two-tier adversary, not the single adversary of the original design

The initial design called for a single configurable adversary model
answering the agent's queries directly. In practice this conflates two
different costs: iterating a search loop hundreds of times per document
is only tractable against a cheap, local model, but a cheap local model's
verdict about whether extraction "failed" is not evidence about a real
attacker's capability. The architecture therefore splits into:

- A **search-tier ensemble** of four locally-hosted models chosen for
  genuinely distinct tokenizer families, with a supermajority consensus
  vote standing in for "defended." A clean-floor check excludes any
  ensemble member that cannot even extract from *unobfuscated* text from
  the vote — otherwise a weak model's inherent failure would be
  confounded with the defense actually working, silently inflating the
  apparent success rate.
- A **verify-tier** of real frontier models, queried only once the
  search tier has converged, to measure whether search-tier "success"
  transfers to an actual adversary.

Local models in the search tier are loaded one at a time on shared GPU
hardware and unloaded between models, never held in memory
simultaneously — a hard operational constraint of the shared-hardware
setting, not a design choice free to relax.

### 3.2 Encoding tool primitives

The agent does not emit raw Unicode directly. It composes a candidate
encoding from six pure-function primitives, each with no side effects and
no external dependency beyond the Unicode Bidirectional Algorithm
reference implementation:

| Tool | Role |
|---|---|
| `render(encoding)` | What a human sees — strips control characters, applies the real UAX#9 bidi algorithm. Used to verify the hard rendering constraint before anything is sent to an adversary. |
| `get_stored(encoding)` | What an LLM tokenizer ingests — the raw codepoint sequence in memory order, excluding control and VS characters. |
| `hamming(s1, s2)` | Positional difference between the stored sequence and the true value. A cheap, tool-computable proxy for "how far has the model's actual input drifted from the ground truth," used by the agent to self-assess a candidate before spending an adversary query on it. |
| `encode_bidi(text, config)` | Applies a bidi control configuration (full reversal, split isolates, nested overrides, multi-block permutations) that changes stored order without changing rendered order. |
| `encode_vs(text, payload)` | Injects invisible Variation Selector characters after visible characters, fragmenting familiar tokens without altering the rendered string. |
| `combine(text, bidi_config, vs_payload)` | Applies bidi encoding first, then VS injection to the *already-reversed/permuted* stored sequence — never to the original field value. |

The ordering constraint in `combine` mirrors the static pipeline's
"order of operations" invariant: VS injection must land on the stored
(post-bidi) sequence, or the invisible characters end up adjacent to the
wrong digits in the model's actual token stream, silently weakening the
defense in a way that is invisible from the rendered output alone.

### 3.3 The agent loop

Each field value goes through:

```
GENERATE:  agent proposes an encoding via the tool primitives
VERIFY:    render(encoding) == original text (hard constraint check)
QUERY:     the adversary (search-tier ensemble, or verify-tier model
           once converged) attempts extraction
REFLECT:   agent is shown the adversary's response and the full attempt
           history, and reasons about why it succeeded or failed
IMPROVE:   agent proposes a revised encoding, informed by that reasoning
REPEAT:    until extraction fails (success) or the iteration budget
           is exhausted
```

This follows the Self-Refine/Reflexion generate–feedback–reflect–improve
paradigm, with one departure that matters for the paper's framing: the
feedback signal is real extraction failure against a real adversary
model, not a self-critique from the same model that generated the
attempt, and not a logprob surrogate. The agent's reflection at each step
answers a specific diagnostic question — did the adversary recover the
exact value (bidi alone is being read through), the reversed/permuted
value (VS is not yet blocking the bidi channel), or a partial value
(injection is working but incomplete) — and that diagnosis determines
whether the next iteration strengthens VS injection, changes the bidi
configuration, or both.

## 4. Self-Improvement: Three Required Components

Iterative refinement within a single field's encoding loop (Section 3.3)
is not, by itself, self-improvement — it is output refinement that resets
for every new document, indistinguishable from vanilla Reflexion. GHOST-
Agent's actual contribution is a mechanism for experience to persist and
compound *across* documents and across process restarts. Three components
are necessary and, together, sufficient for that:

**A. Experience capture.** Every attempt, not just successful ones, is
appended immediately to an append-only, per-attempt log
(`experience_log.jsonl` — JSONL rather than a single JSON array
specifically so a crash mid-write cannot corrupt previously-recorded
history). Each record captures the configuration tried, the resulting
Hamming distance, the adversary's actual response, whether extraction
succeeded, and the target's content profile — the raw material principle
distillation and later analysis both depend on.

**B. Principle distillation.** On a successful encoding, the agent is
prompted to generalize the specific attempt into an abstract, reusable
principle: not "splitting `48271039` at position 4 defeated the model,"
but a claim about what content profile the strategy applies to and why
it works mechanistically. The distillation call reuses the same client
the run is already using, and is deliberately constrained to a short
output (avoiding overly narrow principles that only restate one instance)
and to omit the literal target string (forcing the abstraction).

**C. Persistent memory with content-matched retrieval.** Distilled
principles are stored in `strategy_memory.json`, indexed by a lightweight
content profile (content type, character-count range, presence of
numbers, complexity). A new field's profile is used to retrieve the
most relevant principles by content-type match, length-range overlap,
and profile similarity — weighted by each principle's accumulated
supporting evidence — before the encoding loop for that field even
starts. Retrieved principles are injected directly into the agent's
system prompt as a "learned strategies" section, so later documents see
a richer, more specific prompt than the very first document did.

Together these three components close the loop shown in Section 3's
diagram: a new field triggers structure analysis, memory retrieval,
prompt augmentation, the encoding loop itself, and then — on success —
principle distillation and a memory write, before the system is ready
for the next field. Each of the three is individually necessary: without
capture there is nothing to distill from; without distillation the raw
log never generalizes past its own instances; without retrieval-and-
injection a rich memory file on disk would still leave every new
document starting cold.

### 4.1 A taxonomy for what "self-improving" means here

Because "self-improving agent" is used loosely in the literature, the
system is defined against three explicit levels, used throughout this
paper to state precisely what is being claimed:

- **Level 1 — output refinement.** The agent improves within a session
  via iterative feedback, but the agent itself carries nothing forward
  between documents. This is what Reflexion/Self-Refine already describe,
  and is not treated as a contribution in its own right here — it is the
  substrate the following levels build on.
- **Level 2 — in-session accumulation.** Experience carries forward
  between documents within one running process, but resets on restart.
- **Level 3 — cross-session self-improvement.** Experience persists to
  disk, is loaded on process restart, and — critically — the system's
  measured behavior changes as a result: later documents converge in
  fewer iterations than earlier ones, because the agent's very first
  attempt on a new document starts from a richer, more specific prompt
  than it would have without the accumulated memory.

The empirical signature that distinguishes Level 3 from Level 2 is a
downward trend in iterations-to-convergence as document index increases,
measured across process restarts rather than within one continuous run.
This is what `convergence.py`'s convergence log is designed to produce
and what the paper reports as evidence for the claim: cross-session
memory measurably compounds, and an agent encountering its 100th field
converges faster, on its very first attempt, than the agent encountered
its 1st.

## 5. Summary of the Design's Departures from a Naive Agent Wrapper

It is worth stating plainly what distinguishes this from simply putting
an LLM in a loop around the static GHOST primitives:

1. The feedback signal is real adversarial extraction failure, replacing
   a calibrated logprob threshold that is a crude, indirect proxy for the
   same question.
2. The search and verify roles are explicitly separated into a cheap
   local ensemble and real frontier models respectively, so that
   iteration cost and reported defense strength are never conflated —
   the number the paper reports is always a verify-tier number.
3. Self-improvement is treated as a falsifiable empirical claim with a
   specific required signature (decreasing iterations-to-convergence
   across process restarts, not just within one run), not asserted from
   the presence of a memory file on disk.
4. The hard rendering-fidelity constraint from the static pipeline's
   threat model is preserved exactly — every agent-proposed encoding is
   verified against it before ever reaching an adversary, so the search
   procedure cannot trade away human-readability for defense strength.
