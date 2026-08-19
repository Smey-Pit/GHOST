# GHOST

Defense that stops LLMs from correctly extracting sensitive numeric fields
(account numbers, patient IDs, case numbers, etc.) from scraped documents,
while the document still renders identically to a human reader. Two
independent Unicode-level failure modes are combined, not one: bidirectional
override reversal (model tokenizes characters in a different order than a
human sees) and logprob-guided Variation Selector injection (invisible
characters that fragment familiar digit tokens into sequences the model has
no learned representation for). Full background: `plans/GHOST_BACKGROUND_FOR_CLAUDE_CODE.md`.
Full pipeline/results spec: `plans/GHOST_EXPERIMENT_PLAN.md`.

## Repo layout

Git-initialized 2026-08-14 (`master`, root commit `123e375`). Planning
docs live in `plans/` (`GHOST_BACKGROUND_FOR_CLAUDE_CODE.md`,
`GHOST_EXPERIMENT_PLAN.md`, `GHOST_AGENT_TASKS.md`, `GHOST_self_improving.md`,
`TRACK_A_SPEC.md`, `GHOST_AGENT_PAPER_DRAFT.md`), shell entry points in
`bash_script/` (`submit_all.sh`), all Python in `src/` (including
`track_a_prompts.py`, moved from repo root — it's a plain sibling import
of `track_a_generate.py`, no path hack needed). `.gitignore` excludes
`data/raw/`, `data/encoded/`, `data/track_a/`, `results/raw/`,
`results/tables/`, `results/experience_log.jsonl`,
`results/strategy_memory.json`, `logs/`, `.checkpoint.*.json`, `.claude/`,
`venv/` — all generated/runtime state, never commit these.

## Threat model

Adversary scrapes raw Unicode (BeautifulSoup/Scrapy/CommonCrawl/LangChain-style
loaders) and does NOT apply NFKC normalization. This assumption is load-bearing
— a normalization-aware adversary partially defeats GHOST, and the paper
measures that honestly (Table 6). Hard constraint on the defender's side: the
encoded document must render identically to the original for any standard
Unicode renderer. Never trade this off for stronger obfuscation.

## Single source of truth: `src/unicode_utils.py`

All Unicode encode/strip/normalize primitives live here. Never reimplement
`encode_bidi`, `strip_vs`, `strip_bidi`, `strip_all`, `apply_nfkc`,
`strip_think_tags`, or the VS byte mapping elsewhere — import them.

`derive_payload_bytes(seed, salt, *context)` is the keyed-PRNG payload
generator used everywhere VS injection or permutation search needs
randomness. Every character occurrence gets its own pseudorandom byte
stream derived from `(seed, disruption_payload, doc_id, field_name,
char_index)` — never reuse one fixed payload across characters/fields.
A shared payload or a shared permutation seed leaves a corpus-wide,
pattern-matchable signature, which is strictly weaker than the `strip_vs`
ablation the paper already evaluates. This determinism is also required for
checkpointing: same inputs must always reproduce the same bytes across
reruns.

## Order of operations matters

VS injection is always applied to the STORED (already-reversed, or
already-permuted) character sequence — never to the original field value.
Getting this order backwards puts VS characters adjacent to the wrong
digits in the model's actual token stream and silently weakens the defense
without it being obvious from the visible output.

`NFKC != strip_both`. These are measured as separate attack conditions in
Table 6 — do not collapse them. **Correction (2026-08-14):** this section
previously claimed "NFKC strips VS characters but preserves bidi controls."
That's wrong as a statement about standard Unicode normalization, and was
disproved directly (`src/test_normalization_forms.py`, no model/GPU
needed): NFC/NFD/NFKC/NFKD all leave a real composed-GHOST encoding
byte-identical to raw (50/50 VS chars and 2/2 bidi controls survive every
form). Reason: `apply_nfkc` (`unicode_utils.py`) is literally
`unicodedata.normalize('NFKC', text)`, and Variation Selectors
(U+FE00-FE0F, U+E0100-E01EF) have no compatibility decomposition to
nothing in the UCD — real NFKC does not touch them. `strip_all`/strip_both
is the only one of these that actually removes VS chars; it does so via
an explicit codepoint filter (`strip_vs`), not normalization. See
"Normalization-attack verification" under GHOST-Agent below for what this
means for frontier-model extraction, not just the byte-level check.

## Encoding conditions (`src/encode.py`)

`clean | bae | textfooler | homochar | vs_only | bidi_only | bidi_permute |
ghost | ghost_nfkc | ghost_permute | ghost_permute_nfkc`

- `bidi_permute` / `ghost_permute` (`src/bidi_permute.py`) are an enhancement
  beyond the original plan: instead of a flat RTL reversal, each field is
  stored as a maximally-Hamming-displaced *separable permutation* (avoiding
  patterns 2413/3142, the class realizable by nested bidi isolates/overrides),
  verified against the real UAX#9 `python-bidi` implementation.
- `ghost_nfkc` / `ghost_permute_nfkc` are derived conditions — encoded as a
  separate SLURM step that reads the already-encoded `ghost`/`ghost_permute`
  output from disk. Encode the source condition first.
- `vs_only` and `ghost` need the proxy model (`Qwen2.5-7B-Instruct`) loaded
  locally for logprob-guided stopping; the other conditions don't.
- Ground truth (ground-truth field value) never changes for GHOST/VS/Bidi
  conditions. It CAN change for `bae`/`textfooler`/`homochar`, since those
  attacks may alter the field's actual characters — `encode_textattack` flags
  and updates GT for those instances.
- Encoding is checkpointed per-item (`.checkpoint.{dataset}.json`); reruns
  skip already-encoded items.

Before changing `threshold_tau` or `disruption_payload` in `config.yaml`,
re-run `python src/encode.py --calibrate` (uses the disjoint
`data/raw/calibration.json` split, never written into `data/encoded/`).

## Evaluation conventions

- Primary metric: Field Extraction Accuracy (FEA) — exact match after
  stripping dashes/spaces. Lower FEA = better defense.
- `deepseek_r1_14b` emits `<think>...</think>` before its answer — strip with
  `unicode_utils.strip_think_tags` in every evaluation path that touches this
  model, not just some scripts.
- Refusals (e.g. Gemini sometimes refuses to process GHOST-encoded strings)
  count as a defense success but are NOT averaged into FEA — report
  `refusal_rate` separately.
- `Qwen2.5-7B` is both the proxy (encoding-time) and a target model. Its FEA
  is labelled "proxy-optimised" and excluded from the transferability claim
  over the other 9 models.
- The gradient experiment (Phase 6, `src/gradient.py`) uses reconstruction
  cosine similarity (`all-MiniLM-L6-v2`), not exact match — natural language
  content types make exact match too strict.

## Pipeline (SLURM, `slurm/00_setup.sh` → `08_metrics.sh`)

1. `00_setup` — venv + `requirements.txt`
2. `01_dataset.sh` → `src/dataset.py` — generates `data/raw/{documents,calibration,gradient}.json`
3. `02_encode.sh` → `src/encode.py` — all conditions, GPU (H100), sequential
4. `03_eval_local.sh` / `04_eval_api.sh` → `src/evaluate.py` — **stub, not implemented**
5. `05_inctx_defense.sh` → `src/inctx_defense.py` — **stub, not implemented**
6. `06_norm_attack.sh` → `src/norm_attack.py` — **stub, not implemented**
7. `07_gradient.sh` → `src/gradient.py` — **stub, not implemented**
8. `08_metrics.sh` → `src/metrics.py` — **stub, not implemented**, produces
   Tables 3-6 and Figure 1

Local target models are loaded ONE AT A TIME on the shared H100 and unloaded
(CUDA cache cleared) before the next loads — do not try to keep multiple
loaded simultaneously.

## Prior strength is about injection complexity, not a content-type wall

The Prior Strength Hypothesis (weak-prior numerical content ⇒ defense works,
strong-prior natural language ⇒ model self-corrects past it) describes the
FIXED algorithm's behavior at a fixed, low injection complexity — it is not
a hard boundary of the mechanism itself. Sufficiently complex, non-repetitive
injection defeats reconstruction on long/natural-language text too; a weak
or repetitive signal (whether it's the model's prior or a fixed/predictable
payload) is what a stronger prior — or an attacker — can see through. This is
why per-character keyed payload derivation (`derive_payload_bytes`) matters
beyond anti-pattern-matching: it's the mechanism that makes higher-complexity,
non-repetitive injection possible at all, and it's what GHOST-Agent exists to
push further per-instance.

## GHOST-Agent (`GHOST_AGENT_TASKS.md`, Tasks 1-6 built, real single-field runs done)

A separate, self-improving variant: instead of the fixed encode-then-check
pipeline above (one logprob threshold, calibrated once, applied uniformly),
an LLM agent (`claude-sonnet-4-6` by default -- CRITICAL RULE 5 originally
fixed this, now a config-driven ablation axis, see "Configurable agent
backbone" below) iterates generate → query adversary → reflect → improve, using six tool
primitives (`render`, `get_stored`, `hamming`, `encode_bidi`, `encode_vs`,
`combine`) in `ghost_tools.py`. The point is not "extend GHOST to a new
content type" — it's searching for sufficiently complex, per-instance
encoding using real extraction-failure as feedback instead of a logprob
surrogate.

All six modules are implemented: `ghost_tools.py`, `adversary.py`,
`ensemble.py`, `reflection.py`, `ghost_agent_prompt.py`, `ghost_agent.py`,
`convergence.py`. Each has a passing smoke test (`smoke_test_task{1,3,4,5,6}.py`,
`smoke_test_ensemble.py`) that runs without a GPU by mocking model
load/query. Real ensemble inference (GPU + real 4-model ensemble + real
`claude-sonnet-4-6` agent, no mocking) has now been run on 2 fields end to
end — see "Real single-field runs" below. Two more modules
(`strategy_memory.py`, `ghost_tools_structural.py`) were added later for
cross-session persistence — see "Self-improving audit" below;
`convergence.py` now has a real CLI (`--n_samples`/`--max_iterations`/
`--start_index`/`--no_memory`), not just a callable function.

**Deviation from `GHOST_AGENT_TASKS.md`'s literal spec:** the doc describes
a single adversary model. That was replaced with a two-tier design (see
`config.yaml`'s `agent_ensemble` section):
  - **Search tier** — a local 4-model ensemble (`llama31_8b`, `mistral7b`,
    `deepseek_r1_14b`, `deepseek_llm_7b`), chosen for 4 genuinely distinct
    tokenizer families (`deepseek_r1_14b` is a Qwen-tokenizer distill, so
    plain Qwen is deliberately excluded from this roster — it's already
    the encoding-time proxy above). `src/ensemble.py` owns the
    load-one-at-a-time/query/unload lifecycle and a supermajority
    consensus vote (`consensus_threshold`), with a clean-floor check that
    excludes any member that can't even extract from unobfuscated text
    (otherwise a weak model's "failure" is confounded with the defense
    actually working). This tier is a cost proxy — same status as the
    Qwen logprob proxy — and its success rate is NEVER the paper's
    reported number.
  - **Verify tier** — `frontier_verify_models` (real Claude/GPT/Gemini) in
    `config.yaml`, meant to be run on the converged encoding to measure the
    search/verify gap. **A proper runner for this tier still does not
    exist in the repo** — `convergence.py` only exercises the search tier.
    The gap has been measured manually with a scratch script (not
    committed) calling `adversary.query_adversary` directly on 2 converged
    encodings; see "Real single-field runs" below for what it found.
    Writing the real runner into `convergence.py` needs no GPU (pure API
    calls) and should follow that scratch script's shape.

**Known bug found and fixed (Task 5, mocked-test era):** the task doc's
original `extract_proposed_encoding` required the target's first 3
characters to appear literally in the agent's reported-encoding line —
impossible for a genuinely obfuscated encoding, which by design contains
none of the plaintext. Fixed by having the agent wrap its encoding in
`<final_encoding>...</final_encoding>` tags (`ghost_agent_prompt.py`) and
parsing that tag directly (`ghost_agent.py`).

**Environment blockers are cleared:** GPU access confirmed (idle A100
80GB), and `meta-llama/Llama-3.1-8B-Instruct` /
`mistralai/Mistral-7B-Instruct-v0.3` license acceptance confirmed via the
cached HF token — both gated repos are downloadable. Neither blocker
listed in earlier drafts of this file applies anymore.

### Real single-field runs (first real, non-mocked execution)

Ran `run_ghost_agent` end to end (real GPU ensemble, real
`claude-sonnet-4-6` agent, no mocking) on 2 fields: `account_number`
(`5926847003`) and `reference_number` (`REF-8B9PCAP2`). Both converged in
1 iteration, Hamming distance at the maximum possible for their length
(10/10 and 12/12) — both real data points, though 2 is too few to
conclude 1-iteration convergence is typical; the old static pipeline saw
"3-5 iterations typical," so multi-iteration fields could still change
the cost picture below substantially.

**Bug found and fixed (real API, not caught by mocks):** `run_agent_turn`
(`ghost_agent.py`) originally hardcoded exactly one continuation call
after the model's first round of tool use. When the model called tools
again in that continuation, those `tool_use` blocks were appended to
`messages` with no matching `tool_result` — invisible until the *next*
`client.messages.create()` call, which the real Anthropic API rejects
with a 400 (`tool_use ids were found without tool_result blocks`). Fixed
by looping calls + tool execution until a response contains no
`tool_use` blocks, capped by `tool_budget_per_iter`. A mocked fake client
never exercised two consecutive tool-use rounds, so this was invisible
until run against the real API. Regression test added to
`smoke_test_task5.py`.

**Timing (search tier, `run_ensemble_query`'s load-every-iteration cost):**
cold (first-time weight download) 447.9s for 1 field-iteration; warm
(cached weights) 173-193s. At warm speed, load/unload overhead across the
4 ensemble members (~15-20s checkpoint load each) still dominates —
extrapolating to `GHOST_AGENT_TASKS.md`'s 50-document run (~4-5 fields/doc
→ ~200-250 field-runs) gives roughly 10-12 hours *if* most fields converge
in 1 iteration like these 2 did. Time a larger sample (5-10 fields) before
committing to the full run to get a real iteration-count distribution;
if multi-iteration fields are common, revisit the per-model-batching
restructuring option already noted below.

**Search/verify gap — measured, not hypothetical:** on the first 2 fields
(`account_number`, `reference_number`), the search-tier ensemble declared
the encoding DEFENDED (3/3 valid members), and `claude_sonnet` also failed
to extract on the verify tier — but `gpt56_sol` **correctly extracted the
true value both times** once its token-budget bug (below) was fixed. I.e.
the local-ensemble proxy's "success" did not transfer to at least one real
frontier model, on 2/2 fields tested at that point. `gemini_31_pro`
untested at that point (no `GOOGLE_API_KEY`, `google.generativeai` not
installed in this environment). This is exactly the gap `config.yaml`'s
comments and this file already flagged as a risk of using a cheap local
ensemble as a search-time proxy — confirmed with real data, not just
anticipated.

**Update — `gemini_31_pro` now verified against a real converged
encoding, and the gap did NOT reproduce on this run:** re-ran
`run_ghost_agent` on `account_number = 5926847003` fresh (script:
`src/verify_gemini_run.py`, results:
`results/raw/verify_gemini_account_number.json`). Converged in 1
iteration, Hamming 10/10, search tier 3/3 defended — same as before. This
time all three frontier verify models (`claude_sonnet`, `gpt56_sol`,
`gemini_31_pro`) FAILED to extract the true value; `gemini_31_pro`'s
response (`3007486295`) matched `gpt56_sol`'s exactly. This contradicts
the earlier "`gpt56_sol` extracted both times" finding on this same
field/target. Two non-exclusive explanations, neither ruled out yet: (1)
`gpt56_sol` doesn't accept `temperature=0` (see Gotchas) so its output is
not deterministic run-to-run; (2) `derive_payload_bytes`'s per-character
keyed randomness means a fresh agent run can converge on a genuinely
different bidi/VS encoding each time, even for the same target — the
gap may be encoding-dependent, not just model-dependent. Notably, both
`gpt56_sol` and `gemini_31_pro` answered with the *stored/tokenized*
digit sequence (`3007486295`), not the true value — they ARE reading real
digits off the encoding, just in the obfuscated order, so "defended" here
is a correct FEA verdict but not evidence the model is blind to content.
**Do not treat 1 non-reproduction as the gap being closed** — this needs
a real multi-field, multi-run sample (the still-missing `convergence.py`
frontier-verify runner, see blockers below) before concluding anything
about whether the gap is stable per-field or noisy per-run.

**Two more real-API bugs found and fixed, both in `adversary.py`'s
`_call_openai` (never caught by mocks or the local-ensemble-only testing
so far):**
  1. `gpt-5.6-sol` rejects the `max_tokens` param outright (400) — OpenAI
     requires `max_completion_tokens` for this model. Fixed by switching
     the call to `max_completion_tokens`.
  2. `config.yaml`'s `gpt56_sol.max_tokens: 50` was silently wrong for a
     reasoning model: with a 50-token budget, `gpt-5.6-sol` burned the
     *entire* budget on hidden reasoning tokens (`finish_reason: "length"`,
     `reasoning_tokens: 50`, zero visible output) and returned an empty
     string. `check_extraction` correctly read that as "not extracted,"
     which looked like a defended verdict but was actually a false
     negative — with the budget raised to 2000 (config.yaml, now fixed),
     the SAME encoded input was extracted correctly (see the gap above).
     Same failure mode already documented for `deepseek_r1_14b`
     (`max_new_tokens: 2048`) — `gpt56_sol` just never got the analogous
     fix until this was actually tested against the real API. **Any other
     reasoning-style API model added to `api_models` needs its `max_tokens`
     checked against real `usage.completion_tokens_details.reasoning_tokens`
     before trusting a "failed extraction" verdict from it.**

**Blockers before the full 50-document run:**
  - `run_ensemble_query` loads/unloads all 4 members every single
    iteration (required by the one-model-at-a-time GPU-sharing rule
    above) — cost is dominated by model-load time, not generation (see
    timing above). If a larger sample shows multi-iteration convergence
    is common, the fix is restructuring from per-field convergence to
    per-model batching (load each model once, advance all fields one
    iteration, move to the next model) — an architecture change, not a
    config tweak.
  - The frontier verify-tier runner still needs to be written into
    `convergence.py` (see above) before the 50-doc run can report a real
    search/verify gap at scale instead of 2 hand-run data points.
  - `gemini_31_pro`: **now verified against the real API from an
    `sinteractive` session** (the earlier login-node attempt never
    completed — see Gotchas). The `google-genai` SDK migration in
    `_call_google` is correct as-is; `response.text` returning `None` was
    NOT a response-shape change. Root cause: `gemini-3.1-pro-preview` is a
    reasoning model, and `config.yaml`'s old `gemini_31_pro.max_tokens: 50`
    got silently consumed entirely by hidden thinking tokens
    (`thoughts_token_count=47`, `finish_reason=MAX_TOKENS`, empty
    `content` — no parts at all, so both `response.text` and
    `candidates[0].content.parts` are `None`). Identical failure mode to
    the `gpt56_sol` bug above. Fixed the same way: `config.yaml`'s
    `gemini_31_pro.max_tokens` raised 50 → 2000; a raw smoke-test prompt
    ("say hello") then returned real text (`thoughts_token_count=517`,
    `finish_reason=STOP`). Any other reasoning-style API model added to
    `api_models` still needs this same check before trusting a "failed
    extraction" verdict from it.

The agent role and the adversary role are always separate models — never
hardcode the adversary.

### Real reconstruction run — GHOST-Agent applied to natural language

First real (non-mocked) test of GHOST-Agent on a reconstruction target
instead of a numeric field, closing the loop on the injection-complexity
framing in "Prior strength is about injection complexity, not a
content-type wall" above. Target: a 127-character sentence (a real song
lyric line, chosen arbitrarily, not for any copyright-related reason —
see confound check below). Scripts: `src/verify_reconstruction_run.py`
(search tier), `src/verify_reconstruction_frontier_only.py` (re-run just
the frontier tier against an already-converged encoding). Results:
`results/raw/verify_reconstruction_lyric.json`.

**This required extending `adversary.py`/`ensemble.py`, not just calling
existing code**, since both were hardcoded for field-extraction's exact
match:
- Added `RECONSTRUCTION_PROMPT` (repeat back verbatim, no commentary)
  and a `prompt_template` param to `query_adversary`/
  `query_adversary_local` (default `EXTRACTION_PROMPT`, so field
  extraction call sites are unaffected).
- Added `check_reconstruction()` — cosine similarity via
  `all-MiniLM-L6-v2` (same model the still-unbuilt `gradient.py` is
  specced to use), threshold 0.7, same return shape as
  `check_extraction` so it's a drop-in.
- Added a `checker` DI param to `run_ensemble_query` (mirrors the
  existing `loader`/`unloader`/`querier` pattern) so reconstruction mode
  swaps in `check_reconstruction` without duplicating the ensemble
  load/unload lifecycle.

**Three real bugs found and fixed getting this to actually run:**

1. **The entire local ensemble had silently been running on CPU, not
   the H100, for every run in this project so far** (including the
   `account_number`/`reference_number` runs above). `requirements.txt`
   had `torch>=2.3.0` (unpinned); `uv pip install` resolved a
   `torch==2.13.0+cu130` build requiring a newer NVIDIA driver than this
   node has (driver 570.211.01 → CUDA 12.8 max).
   `torch.cuda.is_available()` returned `False` with only a `UserWarning`
   ("driver on your system is too old"), no error — `device_map="auto"`
   silently falls back to CPU with the run otherwise completing normally,
   so nothing about the output signals this happened. Fixed: reinstalled
   `torch==2.7.1+cu128` (`uv pip install torch==2.7.1 --extra-index-url
   https://download.pytorch.org/whl/cu128` — note `--extra-index-url`,
   not `--index-url`, or every other package stops resolving from PyPI),
   confirmed with a real GPU matmul, and pinned the exact version +
   install command in `requirements.txt` so a future unpinned resolve
   can't regress this silently again. This invalidates the GPU timing
   numbers quoted in "Timing (search tier...)" above — they may have
   been measured on CPU. Re-time before trusting them for the 50-doc
   run estimate.
2. **`run_agent_turn`'s hardcoded `tool_budget_per_iter=8` /
   `max_tokens=2000`** were calibrated against short numeric targets
   (10-12 chars) and were both too small for a 125-char sentence: the
   agent got cut off mid-tool-loop (8 rounds exhausted while still
   calling tools, never reaching a finishing text response) and, once
   that was raised, cut off again mid-encoding inside the
   `<final_encoding>` tag (a VS-injected ~125-char target produces
   roughly 10x that many characters of encoding, which doesn't fit in
   2000 output tokens alongside commentary). Fixed: both are now
   parameters (`response_max_tokens` on `run_agent_turn`,
   `agent_response_max_tokens` on `run_ghost_agent`) instead of hardcoded
   — this run used `tool_budget_per_iter=20`,
   `agent_response_max_tokens=8000`. Callers with longer targets should
   raise these; the old defaults are unchanged for short-field callers.
3. **`_call_anthropic` didn't handle Claude's `stop_reason="refusal"`**
   (a distinct API signal, `content=[]`, separate from a normal short
   response) — it blindly indexed `response.content[0]`, threw
   `IndexError`, and `query_adversary`'s broad except flattened that into
   the literal string `"ERROR"`, indistinguishable from a real API
   failure and NOT tagged as a refusal. This broke the "refusals count
   as a defense success, reported separately as `refusal_rate`"
   convention (Evaluation conventions, above) for every refusal
   `_call_anthropic` has ever hit, not just this run. Fixed: checks
   `stop_reason == "refusal"` and returns the `"REFUSAL"` sentinel
   (matching `_call_google`'s existing convention for its empty-content
   case); `check_extraction`/`check_reconstruction` now also recognize
   the literal `"REFUSAL"` sentinel as `is_refusal=True` (previously
   only `REFUSAL_PHRASES` substring matches did, so `_call_google`'s own
   `"REFUSAL"` return was ALSO silently mis-tagged before this fix —
   same bug, different call site, never caught until reconstruction mode
   surfaced it here).

**Result, once all three were fixed:** search tier converged in 1
iteration (Hamming 114/127). All three frontier verify models defended,
by two different mechanisms:
  - `claude_sonnet`: genuine refusal (`stop_reason: refusal`) —
    confirmed NOT a copyright-lyrics confound: the same model reconstructs
    the clean, unencoded sentence perfectly verbatim with
    `stop_reason: end_turn`, so the refusal is triggered by the
    obfuscated Unicode pattern itself, not the lyric content.
  - `gpt56_sol` and `gemini_31_pro`: both echoed back the raw
    stored/tokenized sequence verbatim (identical string from both
    models) rather than the true sentence — reading real characters off
    the encoding, just failing to undo the obfuscation, same
    "correct-verdict-but-not-blind" pattern already seen in the
    `account_number` gemini run above. Cosine similarity 0.084 (well
    under the 0.7 threshold) for both, once given enough token budget
    (`gemini_31_pro` needed ~12000 — `thoughts_token_count` alone hit
    ~3839 on this input at an intermediate budget, eating almost the
    entire allowance before any visible output).

**Not yet done:** only 1 sentence, 1 run — same "N=1, don't generalize"
caveat as the field-extraction runs above. `gradient.py` itself (Phase 6,
the committed pipeline's reconstruction-mode stage) is still a stub;
`check_reconstruction`'s threshold=0.7 is a provisional default, not a
calibrated one.

### Configurable agent backbone (`src/agent_backbone.py`) — deviation from CRITICAL RULE 5

GHOST_AGENT_TASKS.md's CRITICAL RULE 5 originally fixed the agent role
(the model that proposes encodings, as opposed to the adversary
ensemble/verify tiers that attack them) to `claude-sonnet-4-6`. That
hardcode is gone: the agent role is now backbone-configurable via a new
`config.yaml` section, `agent_backbones` (`default: claude_sonnet`,
`options:` keyed by backbone name), resolved through
`agent_backbone.resolve_backbone(name, config)`. Motivation: real API
cost of running claude-sonnet-4-6 as the agent across hundreds of
fields is expensive, and self-improving-agent literature generally uses
a cheap/local model for the acting role, reserving frontier models for
judging — this makes backbone choice itself an ablation axis (Claude
Sonnet/Haiku/Opus vs. a local DeepSeek-R1 distill) rather than a fixed
constant, same in kind as the earlier single-adversary → ensemble-panel
deviation above.

Two backbone implementations behind one interface
(`reset`/`run_turn`/`generate_text`/`close`):
  - `AnthropicBackbone` — thin wrapper around `run_agent_turn` (moved
    from `ghost_agent.py`, unchanged logic, `model_id` now a parameter
    instead of a hardcoded constant). Native `tool_use`/`tool_result`
    blocks, same as before.
  - `LocalReActBackbone` — local HF causal LMs (e.g.
    `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`, added to
    `local_models`/`agent_backbones.options` alongside the existing
    `deepseek_r1_14b`) have no Anthropic-style structured tool use, so
    tool calls go through a text protocol instead: the model emits
    `<tool_call>{"name": ..., "input": {...}}</tool_call>`, parsed by
    regex + `json.loads` (malformed JSON is skipped, not fatal — a
    local model emitting broken JSON is an expected failure mode, not
    a crash). Same `TOOL_FUNCTIONS`/`execute_tool_call` dispatch as the
    Anthropic path — only the calling *protocol* differs, never the
    tools themselves. `strip_think_tags` applied before parsing (a
    stray `<tool_call>`-shaped string inside a hidden `<think>` block
    must not be parsed as real).

Loaded ONCE and reused across an entire ablation run, not reloaded per
field: `LocalReActBackbone.reset()` clears conversation state but keeps
weights resident, mirroring the load-time-dominates-cost lesson already
learned from the search-tier ensemble (see "Timing" under Real
single-field runs above) — reloading a 14B/32B model every field would
reintroduce that same problem for the agent role too. `convergence.py`
now takes `--agent_backbone <name>`, constructs the backbone once
before its document loop, passes the same instance to every
`run_ghost_agent` call via a new `backbone=` param, and closes it in a
`finally` after the loop.

`run_ghost_agent`'s original `client=` param (an injectable
Anthropic-SDK-shaped client, used by tests) is preserved unchanged for
back-compat: when neither `backbone=` nor `agent_backbone_name=` is
given, it builds a default `AnthropicBackbone(model_id="claude-sonnet-4-6",
client=client)` — every pre-existing call site (`smoke_test_task5.py`,
`verify_gemini_run.py`, `verify_reconstruction_run*.py`,
`self_improving_length_test.py`) is unaffected and needed no changes.
`distil_principle` (principle distillation, Component B) now takes a
`backbone` instead of a `client` + hardcoded model, so a local-model
ablation run distils principles using the local model too, not a
hardcoded Anthropic call underneath.

**Known confound, not yet resolved by tooling — must be handled by the
experiment design instead:** `deepseek_r1_14b` is listed in BOTH
`agent_ensemble.members` (search-tier ensemble) and
`agent_backbones.options` (agent role). Using it as the agent backbone
while it's also an ensemble voter is potentially circular — same risk
already flagged for why `qwen25_7b` (the encoding-time proxy) is
excluded from `agent_ensemble.members`. If `deepseek_r1_14b` is ever
used as the agent backbone for a real ablation run, exclude it from
`agent_ensemble.members` for that run, or treat any "defended" verdict
from it with the same skepticism as a proxy grading its own homework.
`deepseek_r1_32b` has no such conflict (not an ensemble member) and is
the more defensible local-model ablation arm for this reason alone,
separate from its stronger reasoning capacity.

**Not yet run:** no real backbone-ablation results exist yet — only a
mocked smoke test (`smoke_test_agent_backbone.py`, no GPU/API calls)
verifying `LocalReActBackbone`'s tool-call round-trip and
`resolve_backbone`'s config dispatch. Real convergence numbers per
backbone (success rate, iterations, cost) still need an actual run.

### Real `deepseek_r1_32b` backbone debugging session (first real, non-mocked exercise)

First attempt to run `deepseek_r1_32b` as the agent backbone on a real
field (`jurisdiction_code = "QLD-17"`, Track A pilot doc `regfiling_0000`,
`src/verify_deepseek_r1_32b_backbone.py`, scratch/uncommitted). Surfaced
three real, unrelated failures in sequence, each fixed before the next
was even visible — not a single bug, a chain of them:

1. **GPU VRAM OOM at iteration 2** — the resident 32B backbone (~64GB
   bf16) and a search-tier ensemble member's own GPU residency briefly
   overlapped mid-run (`torch.OutOfMemoryError`, ~78GB in use of 79.25GB
   capacity). Fixed by adding `release_gpu()`/`reacquire_gpu()` to the
   `Backbone` interface — `LocalReActBackbone` moves its weights to CPU
   immediately before `ensemble_query_fn` and back immediately after;
   no-op for `AnthropicBackbone`. `ghost_agent.py`'s loop calls these
   around every `ensemble_query_fn` call unconditionally.
2. **Host-RAM SIGKILL (exit 137) before any GPU work started** — this
   SLURM interactive allocation only has `--mem=60G`, and staging a 32B
   model's checkpoint shards into host RAM during `from_pretrained`
   already exceeds that on its own, independent of fix #1 above (which
   made it WORSE, since `release_gpu()` also needs ~64GB of host RAM to
   hold the offloaded weights). Fixed by adding 8-bit quantization
   (`bitsandbytes`, `agent_backbone.py`'s `_load_backbone_model`,
   config-gated via `agent_backbones.options.deepseek_r1_32b.quantization:
   "8bit"`) — ~32-34GB VRAM instead of ~64-66GB, fixing both the VRAM
   contention AND the host-RAM staging spike at once. `release_gpu()`/
   `reacquire_gpu()` are deliberately no-ops when quantized: the
   footprint no longer needs relief, AND `bitsandbytes`' `Linear8bitLt`
   layers hold GPU-specific quantization state that isn't guaranteed
   safe to round-trip through an arbitrary `.to("cpu")`/`.to("cuda")`
   move. Introduces quantization noise as an uncharacterised variable in
   the encoding-search results — not yet measured against an unquantized
   run.
3. **Tool-call parser gap, not a model capability gap** — once loading
   was fixed, the backbone kept failing to converge:
   `_parse_tool_calls` only recognised the instructed
   `<tool_call>...</tool_call>` tag, but real `deepseek_r1_32b` output
   used markdown ` ```json ` fences instead (confirmed via
   `diagnose_local_backbone.py`, a lighter backbone-only probe built
   specifically to capture FULL untruncated model text — `ghost_agent.py`'s
   own prints truncate to 300/500 chars, which was hiding this). Fixed:
   `_parse_tool_calls` now accepts both forms, matched in the order they
   appear; a JSON block without a `"name"` key is still correctly
   ignored (so an unrelated ` ```json ` example in the model's prose
   isn't misparsed as a real call). Also found and fixed: `run_turn`
   ignored `self.max_new_tokens` (the backbone's own configured budget)
   in favor of whatever the caller's `response_max_tokens` happened to
   be — both `run_turn` and `generate_text` now take
   `max(caller_value, self.max_new_tokens)` as a floor.
4. **Still unresolved, separate from all three fixes above:** even with
   the parser fixed, the backbone's `<final_encoding>` output doesn't
   reflect the transformation it narrates — both real attempts on this
   field ended with the literal, untransformed `"QLD-17"`. One response
   also hallucinated having already called `combine("QLD-17", "full_rtl",
   "ghost")` in a prior iteration when no tool call had actually
   happened. This reads as a genuine capability gap for this model at
   this ReAct-over-text-protocol task (narrating a plausible plan
   without executing/copying it), not something a parser or prompt
   tweak alone is confirmed to fix — deliberately NOT addressed yet
   (scope explicitly deferred: "fix the parser only" was the chosen
   scope for this session). The verification run that would give a real
   converged/failed final answer for this field was stopped mid-way
   (backbone-loaded, iteration 1 in progress) to prioritize the
   sentence-level obfuscation work below — no real end-to-end result
   exists yet for `deepseek_r1_32b` as backbone on any field.

### Sentence-level obfuscation (Phase 1 + 2 built; VS/GHOST-Agent extension deferred)

Every existing condition (`bidi_only`/`vs_only`/`ghost`/`bidi_permute`/
`ghost_permute`, `src/encode.py`'s `_apply_field_transform`) and
GHOST-Agent both obfuscate ONLY the isolated field value, never the
surrounding sentence (label, expected format, prose) — leaving an
adversary contextual scaffolding it can use to reconstruct a garbled
value. Chosen over whole-document scope: nearly all the real contextual
leakage sits in the immediate sentence, and whole-document-scale bidi
permutation is a much larger, previously-untested surface for the
project's non-negotiable "must render identically" guarantee to quietly
break on. `bidi_permute` (the separable-permutation mechanism) was
extended first, not the combined bidi+VS `ghost_permute` — one new
scaling problem at a time.

**Phase 1 (de-risking, `src/bidi_permute.py`) found and fixed a real bug
before any pipeline wiring, exactly why the gate existed:**
`find_target_permutation` only ever optimised Hamming distance and
NEVER verified round-trip correctness — true by coincidence for the
digit-only fields (2-12 chars) this module was built for, but a real
~10-30% round-trip failure rate on realistic sentence-like text (20-400
chars, letters/spaces/punctuation), confirmed via a new
`_self_test_sentence_scale()` (`--selftest-sentence` CLI flag). Root
cause: neutral/weak bidi-type characters (spaces, punctuation) can land
in a permutation position where UAX#9's real neutral-resolution rules
(W1-W7/N1-N2) diverge from the naive "every character is an
independently reorderable opaque unit" construction this module makes —
NOT simply "any real prose eventually fails" (two direct control tests —
no boundary neutrals; forced boundary neutrals — both passed 0/40,
ruling out the simplest hypothesis). Fixed by making round-trip
correctness a hard filter INSIDE the search loop (`find_target_permutation`
now rejects any candidate that doesn't actually render back via the real
`bidi.get_display` oracle, not just an after-the-fact assumption) —
re-verified 100% round-trip across all tested lengths, embedding depth a
clean max of 25 against UAX#9's real 125-level cap (also unverified at
this scale before now). Cost: search is now real, non-trivial API-oracle
work per trial instead of a cheap integer comparison (field-length
self-test 330/330 went from near-instant to ~8s; sentence-scale 500-trial
budget took ~52s for 50 instances) — `config.yaml`'s
`bidi_permute_search_trials: 500` was calibrated when search was free;
worth re-timing against a realistic per-document budget before a full
run. The two existing production callers (`encode_bidi_permute`,
`encode_ghost_permute`) were hardened with a loud `RuntimeError` instead
of a confusing crash for the (rare, previously-impossible) case where
the filtered search exhausts its trial budget with no valid candidate.

**Phase 2 (`src/sentence_utils.py`, `src/encode.py`) wires this into a
new condition, `bidi_permute_sentence`** (added to `config.yaml`'s
`encoding_conditions`, never collapsed with `bidi_permute` — same
"measured as separate conditions" convention as `NFKC != strip_both`).
`find_field_sentence(text, field_value, char_span=None)` supports BOTH
dataset shapes in one function: `data/raw/documents.json`
(`src/dataset.py`'s `text = " ".join(sentences)`, one phrase-template
sentence per field — no boundary detection needed, `text.find` suffices)
and Track A's narrative prose (real sentence-boundary detection via
`char_span`, since a field value can repeat elsewhere in a longer
document). Boundary regex splits on whitespace immediately following
`[.!?]` — deliberately NOT triggered by the punctuation alone, since
decimal amounts (`"1234.56"`) and ICD-style codes (`"A12.3"`,
`dataset.py`'s `_gen_icd_code`) never have adjacent whitespace at their
internal period, sidestepping digit-adjacency lookarounds entirely.
`_apply_sentence_transform` dedupes fields sharing one sentence — a REAL
case, not hypothetical: Track A's `regfiling_0000` has `jurisdiction_code`
and `registration_date` genuinely in the same sentence, confirmed
directly and used as a smoke-test fixture. `smoke_test_sentence_utils.py`
verifies all of this against real data (both dataset shapes, the
decimal/ICD edge cases, the real shared-sentence case) with NO mocking of
the bidi oracle, including the most important check: **full-document
round-trip** (`bidi_visible(encoded_doc["text"]) == original_doc["text"]`)
— not just the isolated sentence in a vacuum, since that's the scope the
condition actually ships at.

**Explicitly deferred, not silently dropped:** `ghost_permute_sentence`
(VS injection at sentence scope — a second new scaling problem,
injection density over much longer text); real timing/cost validation of
`bidi_permute_search_trials` at sentence scale for a full dataset run;
whole-document scope (already rejected in favour of sentence scope).
GHOST-Agent's sentence-scope extension is NO LONGER deferred — see below.

### GHOST-Agent sentence-scope extension (first real run, Claude Haiku 4.5 backbone)

Extends GHOST-Agent (the generate → query-ensemble → reflect loop, distinct
from the static `bidi_permute_sentence` condition above -- these are two
separate mechanisms, initially conflated in discussion before this work
started) to obfuscate the SENTENCE containing a field, not just the bare
field value, mirroring the static condition's motivation.

**Code change, `src/ghost_agent.py`:** `run_ghost_agent`/
`_run_ghost_agent_loop` gained `field_ground_truth: Optional[str] = None`,
decoupling `target` (what the agent obfuscates -- now a sentence) from
what an extraction must recover to count as success (must stay the bare
field value). `None` (default) reproduces every existing caller's exact
prior behavior (`field_ground_truth` collapses to `target`, `clean_reference_text
= f"{field_name}: {target}"` as before) -- zero change for bare-field
callers. When given: `clean_reference_value = field_ground_truth`,
`clean_reference_text = target` (the sentence reads naturally on its own,
no artificial prefix needed), and `ensemble_query_fn(...,
ground_truth=field_ground_truth, ...)` instead of `target`. Confirmed via
direct re-reading that NOTHING else needed changing --
`ghost_agent_prompt.py`'s system prompt/`build_iteration_prompt`,
`adversary.py`'s prompt templates, `extract_proposed_encoding`, and
`ghost_tools.py`'s six tools were all already string-length-agnostic.

**Config change:** added a `claude_haiku` entry under `config.yaml`'s
`api_models` (did not exist -- `agent_backbones.options.claude_haiku` is a
SEPARATE section for the backbone role; the verify-tier query path reads
`api_models` instead). NOT added to `agent_ensemble.frontier_verify_models`
(that production list is untouched; a one-off scratch script hardcodes its
own 4-model adversary panel instead).

**Real run, `src/verify_sentence_agent_haiku.py`:** pilot example
`regfiling_0003`, field `jurisdiction_code = "VIC-17"`, sentence located
via `sentence_utils.find_field_sentence` (157 chars, single-field sentence
-- chosen over reusing `regfiling_0000` specifically because it has no
co-occurring second field, cleaner to interpret for a first test). Backbone:
Claude Haiku 4.5 (`agent_backbones.options.claude_haiku`).

**Bug hit and fixed on the first attempt, an already-documented lesson not
applied the first time:** ran with the default `tool_budget_per_iter=8`/
`agent_response_max_tokens=2000` (calibrated for short numeric fields) and
failed to converge in all 5 iterations -- every single one showed the
agent genuinely reasoning about real tool calls and strategies (`combine()`,
real `hamming` results) but never reaching the closing `<final_encoding>`
tag before running out of budget. This is the EXACT failure mode already
documented above under "Real reconstruction run" for a 125-char target
(`tool_budget_per_iter=20`/`agent_response_max_tokens=8000` needed there) --
our 157-char sentence is in the same regime, and the lesson just wasn't
applied when the script was first written. Fixed by setting both explicitly
in the script; re-run succeeded immediately.

**Result (1 field, 1 run -- do not generalize past N=1):** search tier
converged in ONE iteration via `full_rtl` + VS injection (`ghost` payload),
Hamming 144/158 (91%), **4/4 valid search-tier ensemble members defeated**
-- a full consensus on the first try, better than every field-scope
`regfiling_0000` test earlier this session (which typically needed 2
iterations). Frontier verify tier, 4 real models:

| model | verdict | mechanism |
|---|---|---|
| `claude_haiku` | DEFENDED | genuine refusal -- but see caveat below |
| `claude_sonnet` | DEFENDED | genuine refusal (`REFUSAL` sentinel) |
| `gpt56_sol` | **EXTRACTED** | correctly recovered `"VIC-17"` |
| `gemini_31_pro` | DEFENDED | not blind -- read a garbled fragment of the stored sequence, didn't recover the field |

3/4 defended, 1/4 extracted. Two real caveats on how to read this, not
hedging for its own sake:
  - **`claude_haiku` circularity**: it is BOTH the agent backbone that
    proposed this encoding AND one of the four verify-tier adversaries
    testing it -- same risk already documented for `deepseek_r1_14b` being
    both an ensemble member and a candidate backbone. That it refused
    cleanly (rather than silently mis-extracting) is a real, not-nothing
    data point, but this column should be weighted less than the other
    three.
  - **`gpt56_sol` non-determinism** (already documented above): rejects
    `temperature=0`, so this exact result is not guaranteed to reproduce
    run-to-run -- its extraction here is consistent with its
    already-observed "least reliably defended" pattern across this whole
    session, not a new finding specific to sentence scope.

Full result (exact sentence, full encoding, search-tier detail, all 4
verify responses) saved to
`results/raw/verify_sentence_agent_haiku_jurisdiction_code.json`.
`use_memory=False` was used deliberately for this run --
`strategy_memory.json`'s principles were all distilled from FIELD-scope
attempts under a schema that doesn't distinguish scope, so this run
neither read nor wrote that shared file.

**Not yet done:** only 1 field, 1 run -- same "don't generalize" caveat as
every other single-instance real run this session. No comparison yet
against the static `bidi_permute_sentence` condition on the SAME field (the
natural next question: does the agent's LLM-searched encoding do better or
worse than the deterministic permutation search on this exact sentence?).
`convergence.py` still has no batch-mode sentence-scope path -- this was a
single hand-run script, not wired into the real pipeline.

### Self-improving audit (`GHOST_self_improving.md`)

That doc defines 3 levels of "improvement" and demands the codebase be
audited against them before claiming any of it:
  - Level 1 (output refinement only, resets every document) —
    Reflexion/Self-Refine-style, not genuine self-improvement.
  - Level 2 (in-session accumulation, resets between process restarts).
  - Level 3 (genuine self-improvement: experience persists to disk,
    loads on restart, and — the doc's own required proof — iterations
    to convergence measurably DECREASES as document index increases).

**Audit finding: the codebase was Level 1**, confirmed by direct
inspection, not inference:
  - `reflection.py`'s `ReflectionMemory` is a fresh in-memory dataclass
    constructed inside every `run_ghost_agent()` call — never written
    to disk, never loaded. `convergence.py` created a new one per
    field with nothing threaded between fields or documents, so not
    even Level 2 held within a single process.
  - No `strategy_memory.py`, no content-profile/`analyse_structure`
    tool, and no `if __name__ == "__main__"` / `argparse` in
    `convergence.py` existed anywhere in `src/` before this audit.
  - `ghost_agent_prompt.py`'s `GHOST_AGENT_SYSTEM_PROMPT` was a bare
    module constant, never concatenated with anything — no
    memory-injection path existed at all.

**Implemented (all net-new, nothing rewritten):**
  - `src/ghost_tools_structural.py` — `tool_analyse_structure()`, a
    content-profile heuristic (content_type/n_chars/n_sentences/
    has_numbers/complexity). Simple by design — it only needs to group
    "similar" targets for retrieval, not classify text in general.
  - `src/strategy_memory.py` — `results/strategy_memory.json` (indexed
    principles, atomic write-then-rename) + `results/experience_log.jsonl`
    (append-only, Component A). Paths are anchored to the repo root via
    this file's own location (`Path(__file__).resolve().parent.parent`),
    not CWD, so behaviour doesn't depend on which directory a script is
    launched from.
  - `src/ghost_agent.py` — `run_ghost_agent()` now: loads memory and
    injects `retrieve_relevant_principles()` into the system prompt
    BEFORE the loop starts (Component C); appends an experience record
    to the JSONL after every iteration (Component A); on success, calls
    a new `distil_principle()` (reuses the SAME `client` the run is
    already using, not a fresh one — see its docstring on why) and
    saves the strengthened/new principle (Component B). Added
    `doc_index`/`use_memory` params — `use_memory=False` reproduces the
    exact pre-memory behaviour and is what the existing
    `smoke_test_task5.py` now passes, since its `FakeClient` is
    pre-scripted for exactly 2 `.messages.create()` calls and a 3rd
    (distillation) would IndexError, plus tests must not touch real
    `results/` files.
  - `src/convergence.py` — real CLI (`--n_samples`, `--max_iterations`,
    `--start_index`, `--no_memory`); `doc_index`/`content_type`/
    `n_principles_available` CSV columns; deterministic non-overlapping
    sampling (seeded shuffle + slice, not `random.sample` — so
    `--start_index` walks a reproducible, genuinely different document
    window rather than an independent re-roll); CSV write switched from
    overwrite to append so a second invocation extends the same
    convergence curve instead of clobbering the first's rows. Also
    fixed a latent path bug: the old `../config.yaml`-style defaults
    only worked if CWD was `src/`, but the doc's own smoke-test
    commands invoke `python src/convergence.py` from the repo root —
    defaults are now anchored to this file's location instead.

**Real smoke test run (`results/tables/agent_convergence.csv`,
`results/strategy_memory.json`, `results/experience_log.jsonl` — all 3
are real artifacts from this run, not illustrative):**
  1. `python src/convergence.py --n_samples 3 --max_iterations 3` (docs
     0-2, 13 fields): 10/13 succeeded, 2 principles distilled
     (`numeric`, `alphanumeric_id`), `n_principles_available` climbed
     0→1→2 WITHIN this single process as successes accumulated —
     Level 2 behaviour confirmed live.
  2. `python src/convergence.py --n_samples 3 --max_iterations 3
     --start_index 3` (docs 3-5, 10 fields, a completely FRESH
     process): every field started with `n_principles_available=2` —
     loaded from disk, not carried over in memory (there was no shared
     memory to carry over — separate process). **This is the direct,
     confirmed evidence of cross-session persistence** — Level 3's
     defining mechanism is real, not just plumbed in but inert.
     `supporting_evidence` on both principles grew 6→9 and 4→9 across
     the two batches (both now `"confidence": "high"`), confirming
     `add_principle`'s strengthen-existing path also fired for real.

**Honest assessment (per the doc's own "do not claim Level 3 without
the curve" instruction) — the MECHANISM is Level 3, the BENEFIT is not
yet proven:** mean iterations to success was 1.4 in both batches,
identical. No decreasing-iterations curve at this N. Why, honestly:
  - Most fields (short numeric/alphanumeric IDs) already converge in 1
    iteration cold — a floor effect leaves no room for memory to show a
    speedup on the easy majority.
  - The fields that failed weren't rescued by memory: `dosage` failed
    in BOTH batches (cold and warm, identically), and `invoice_number`
    failed in batch 2 despite 2 high-confidence principles being
    available. The two distilled principles mostly restate the agent's
    own default first move (full RTL + VS injection) — not novel
    enough to fix genuinely harder cases.
  - N=6 documents / 2 principles is too small regardless; the doc's own
    "document 100, mature memory" framing implies dozens+ documents,
    not 6.
  - Do not cite this as "Level 3 achieved" without re-running at a much
    larger `--n_samples` and actually plotting iterations vs.
    `doc_index` from the (now real, growing) CSV.

**Known imprecision, not yet fixed:** `strategy_memory.json`'s
`n_documents_processed` (matches the spec's literal field name) is
actually incremented once per successful FIELD-level distillation, not
once per document — a mismatch inherited from the spec's simpler
one-field-per-document assumption against this codebase's per-field
`run_ghost_agent()` granularity. Harmless functionally (nothing reads
this field for a real calculation yet), but misleading if it's ever
reported as a document count.

### Agentic-ness critique and fixes (2026-08-16)

User raised two concerns about GHOST-Agent's real behavior (from the two
real single-field runs and the sentence-scope run documented above), not
hypothetical design worries:

1. **The agent converges in 1 iteration and likely never reaches a
   maximally complex encoding.** Confirmed by direct code inspection, not
   speculation: `_run_ghost_agent_loop`'s stopping rule
   (`ghost_agent.py`) is a single binary check —
   `ensemble_result["defended"]`, i.e. `consensus_threshold` (3 of 4)
   *search-tier* local models fail to extract. There is no margin
   condition, no reward for continuing past first success, and the
   search tier is deliberately the WEAK proxy (chosen for tokenizer
   diversity, not strength) — real frontier models are never consulted
   during search at all (`frontier_verify_every_n_iterations: 0`). The
   agent is optimizing for "cheapest thing that clears a low fixed bar,"
   not "most complex injection" — this is the same mechanism already
   documented as the search/verify gap (search tier says defended,
   `gpt56_sol`/`gemini_31_pro` still extract).
2. **What the agent actually sees/learns is coarser than it looks.**
   Per-iteration reflection (`reflection.py`'s `format_for_reflection`)
   only gave the agent a categorical per-member verdict
   (excluded/refused/extracted/defeated) — never the model's actual
   guessed value, a logprob, or which characters it stumbled on.
   **Worse, a real bug**: `Attempt.bidi_config`/`vs_payload` were
   hardcoded to the literal string `"from_agent"` (`ghost_agent.py`),
   not the real parameters — because the agent emits one already-
   combined `<final_encoding>` string via tool calls rather than
   structured fields, nothing had ever captured what tool calls it
   actually made. The agent effectively could not see its own past
   configuration in structured form, only vague free-text reasoning
   snippets (truncated to 500 chars).

**Fix 1 — real config/payload now flows into reflection.** Both
`AnthropicBackbone` and `LocalReActBackbone` (`agent_backbone.py`) now
track `self.last_tool_calls` (a `(name, input_dict)` list, reset every
turn) as each tool call executes. New
`agent_backbone.extract_config_from_tool_calls()` recovers the real
`bidi_config`/`vs_payload` from that log (prefers the last `combine`
call, falls back to standalone `encode_bidi`/`encode_vs` calls).
`ghost_agent.py`'s loop now passes these into `Attempt` instead of the
placeholder — `strategy_memory`'s persisted principles inherit this fix
for free, since they already read the same `Attempt` fields. Verified
against both existing smoke tests (`smoke_test_task5.py`,
`smoke_test_agent_backbone.py`), which still pass; not yet verified
against a real run that this materially changes agent behavior (only
that the plumbing is correct).

**Fix 2 — a calibration-only mechanism for learning a better stopping
heuristic, without corrupting transferability.** The obvious next fix
for problem 1 — consult a real frontier model during search and require
it to also fail — would optimize the search directly against that
model, which could then no longer be reported as a held-out
transferability result for it. This is the exact same risk this repo
already flags for `qwen25_7b` (encoding-time proxy, excluded from the
transferability claim) and `deepseek_r1_14b` (ensemble-member-vs-
agent-backbone conflict) — user independently identified this
themselves ("that's technically optimising ON the frontier and not
transferability") before any code was written, and separately reported
manually confirming a real converged encoding extracts correctly
against `gpt56_sol` via the ChatGPT UI (anecdotal, N=1, undocumented
which encoding — not a logged result).

Resolution mirrors `encode.py`'s own `threshold_tau` calibration
pattern: reuse `data/raw/calibration.json` (the 20-doc split already
disjoint from the main eval set) as the ONLY place a frontier model is
allowed to influence search:
- `ensemble.run_ensemble_query` gained an optional `frontier_check_fn`
  param (`None` by default — every existing call site is unaffected).
  When given, it's called ONLY on attempts the cheap local ensemble
  already calls "defended" (cost control — no point checking before
  that), and if the frontier model DOES extract, the verdict flips back
  to `defended=False` (a clearly-labeled `"... [CALIBRATION-ONLY
  frontier gate]"` row is appended to `per_member` so it shows up in
  reflection text like any other member).
- `run_ghost_agent`/`_run_ghost_agent_loop` (`ghost_agent.py`) thread
  `frontier_check_fn` straight through to `ensemble_query_fn`, but only
  include the kwarg when actually given, so a custom
  `ensemble_query_fn` (test fakes, reconstruction-mode swap-ins) that
  doesn't accept it keeps working unchanged.
- `adversary.make_frontier_check_fn(model_key, config)` builds the
  actual callable from a `config.yaml` `api_models` entry (e.g.
  `"gpt56_sol"`, matching the user's own manual check) via
  `query_adversary` + `check_extraction`.
- New `src/calibrate_agent_search.py`: runs each calibration field
  TWICE — baseline (no gate) vs. gated (frontier must also fail) — and
  reports the iteration/Hamming delta, plus how many fields the gate
  never resolved within `max_iterations` (i.e. the local ensemble alone
  would have falsely declared success). `use_memory=False` for both
  runs, so calibration never pollutes `strategy_memory.json`'s
  principles with frontier-gated search data.
- Docstrings in all three files spell out the constraint explicitly:
  **`frontier_check_fn` must never be passed on real eval/Track A
  data** — only against the disjoint calibration split, to derive a
  stopping heuristic (e.g. "require 4/4 not 3/4," "iterate N more
  rounds past first pass") that then gets applied with
  `frontier_check_fn=None` again on the real per-document search.

**Verified (mocked, no GPU/API):** `tests/smoke_test_search_calibration.py`
covers the frontier gate's cost-control behavior (only fires when local
tier already says defended), the overturn logic, `make_frontier_check_fn`'s
config resolution, and `calibrate_agent_search.run_calibration`'s
comparison/CSV logic against a fake calibration file and fake
`run_ghost_agent_fn`.

**Real calibration run (2026-08-16), N=2 fields, first real numbers from
this mechanism:** `python src/calibrate_agent_search.py --n_fields 2
--frontier_model gpt56_sol --agent_backbone_name claude_haiku`, against
`cal_fin_001` (financial domain) from `data/raw/calibration.json`.
Results (`results/tables/agent_search_calibration.csv`):

| field | baseline (local-ensemble-only) | gated (also requires gpt56_sol to fail) |
|---|---|---|
| `transaction_date` (`2022-02-15`) | success=True, 1 iter, Hamming 8 | success=**False**, 4 iters, Hamming 9 |
| `reference_number` (`REF-7W8R4NEA`) | success=True, 1 iter, Hamming 10 | success=**False**, 3 iters, Hamming 11 |

**2/2 fields**: the local-ensemble-only stopping rule declares victory
after exactly 1 iteration, but `gpt56_sol` would still have extracted
the true value from that "converged" encoding — a real, quantified
instance of the search/verify gap already documented above, not just
theoretical. Notably, forcing the agent to keep trying under the
stricter gate did NOT find an escaping encoding within the `max_iterations`
budget (5) either — Hamming distance only inched up by 1 in both cases
(8→9, 10→11) despite 2-3 extra attempts. This suggests the fix isn't
simply "iterate more" — something about *what* the agent tries when it
iterates needs to change too (the `bidi_config`/`vs_payload` reflection
fix earlier this session was aimed at exactly this, but hasn't yet been
checked with `verbose=True` against a real run to confirm the agent is
varying strategy meaningfully across these extra iterations, vs. mostly
repeating similar attempts).

**One iteration-counting subtlety worth knowing before reading any
`n_iterations` number from this mechanism:** `n_iterations` in
`run_ghost_agent`'s return dict is `memory.n_attempts` — a count of
iterations that produced a PARSEABLE `<final_encoding>` tag, not a count
of raw loop passes. `_run_ghost_agent_loop` silently `continue`s (no
`Attempt` recorded, no ensemble/frontier query) whenever
`extract_proposed_encoding` returns `None` for that turn. Confirmed for
real on the `reference_number` gated run above: counting the local
ensemble's own "Loading checkpoint shards" log groups (4 members loaded
per ensemble query) showed only 3 ensemble queries fired even though the
5-iteration budget should have produced up to 5 — one turn's `run_turn`
call never yielded a scoreable encoding at all, most likely a
`tool_budget_per_iter`/`agent_response_max_tokens` (defaults 8/2000,
calibrated for short numeric fields) budget squeeze on Haiku while it
was doing extra reasoning to try to escalate under the stricter gate —
the same failure mode already documented for the 125-char sentence run
needing `tool_budget_per_iter=20`/`agent_response_max_tokens=8000`. This
run used `verbose=False` (`calibrate_agent_search.py`'s calls don't
thread it through), so this is inferred from the ensemble-load log
pattern, not confirmed via the `"Could not extract encoding from agent
text"` print that would have shown directly. **Next calibration run
should pass `verbose=True`** to confirm this directly and decide whether
to raise the token/tool budget so a genuine escalation attempt isn't
silently discarded as a parsing failure.

**Not yet done:** only 1 document's worth of fields (N=2) — same "don't
generalize" caveat as everywhere else in this file. No `--n_fields`
sweep across the other 3 domains (medical/legal/technical) in the
calibration split yet, and no `verbose=True` re-run to confirm the
iteration-counting hypothesis above.

### Static bidi-permutation baseline check on real Track A data (2026-08-16)

Direct follow-up question during the agentic-ness discussion: does the
STATIC `bidi_permute` mechanism (Phase 1/2, `src/bidi_permute.py`) do an
exhaustive all-combinations search? Confirmed by reading
`find_target_permutation` directly: no — it's a SEEDED RANDOM search
over up to `trials` (2000 is this module's own internal default; the
real pipeline's `config.yaml` uses `bidi_permute_search_trials: 500`)
sampled separable permutations, keeping the best Hamming-distance
candidate that also survives a round-trip-correctness check against the
real bidi oracle. This is a best-of-sampled approximation, not a proof
of reaching the true theoretical maximum — though on the one field
tested below, it happened to find the actual maximum anyway.

**First real end-to-end test against a real Track A field**
(`src/verify_bidi_permute_track_a.py`, scratch/uncommitted, same status
as `verify_gemini_run.py` etc.): field `regfiling_0000` /
`jurisdiction_code` = `'VIC-23'`. Ran the search at 2000 trials (seed=0),
spliced the encoded field back into its real Track A document at its
real `char_span`, and verified the FULL document (not just the isolated
field) still renders byte-identical to the original via the real bidi
oracle — the hard rendering constraint held.

**Result:** search found Hamming distance **6/6 — the maximum possible**
for this 6-character field (stored sequence `'C-VI32'`). Queried
`gpt-5.6-sol` (real API call) on the full spliced document with the
field's real extraction prompt: **response was `'VIC-23'` — extracted
correctly, no refusal. Defended = False.**

So even at the maximum achievable Hamming distance for a PURE
bidi-permutation encoding (no VS injection at all), `gpt56_sol` read
straight through it on this field. Consistent with every other result
against this model this session (search/verify gap, normalization
attack, sentence-scope run, the calibration numbers immediately above)
— `gpt56_sol` has been the least reliably defended adversary tested
throughout, across static AND agentic mechanisms alike. This is bidi-only,
not the combined `ghost_permute` (bidi + VS) — a natural next check is
the same field through `ghost_permute` for a fairer comparison against
the agent's VS-inclusive results, not yet done. N=1 field — don't
generalize past this single data point.

Full result saved to
`results/raw/verify_bidi_permute_track_a_jurisdiction_code.json`.

### Sentence-scope permutation + real logprob-guided VS injection (2026-08-16)

Direct follow-up requested after the field-only static-bidi check above:
repeat it at SENTENCE scope (the whole sentence containing the field,
not just the field's own characters) AND with real VS injection layered
on top — the exact `ghost_permute_sentence` combination CLAUDE.md's
"Sentence-level obfuscation" section already flagged as explicitly
deferred (bidi-scope-only `bidi_permute_sentence` existed; the VS+
sentence combination didn't).

**New code**: `encode_ghost_permute_sentence` added to `src/encode.py`,
directly beside the existing `encode_bidi_permute`/
`encode_bidi_permute_sentence`/`encode_ghost_permute` family — combines
`encode_bidi_permute_sentence`'s sentence-scope permutation search with
`encode_ghost_permute`'s logprob-guided per-character VS injection
(via the real `Qwen2.5-7B-Instruct` proxy, `encode_char_with_vs_logprob`),
applied in STORED (permuted) order over the whole sentence. Not yet
added to `config.yaml`'s `encoding_conditions` or run at corpus scale —
first exercised as a single real end-to-end check
(`src/verify_ghost_permute_sentence_track_a.py`, scratch/uncommitted,
same status as the other `verify_*.py` scripts).

**Why this needs the GPU, unlike the field-only bidi-only check**: real
"VS injection" in this pipeline is logprob-GUIDED, not fixed-depth —
`encode_char_with_vs_logprob` appends VS bytes one at a time and queries
the real proxy model's logprob of the true character after each one,
stopping once it drops below `threshold_tau` (-8.0) or hits a 64-iteration
safety cap. This is the actual calibrated production mechanism (what
`ghost`/`ghost_permute` measure), distinct from GHOST-Agent's own
`ghost_tools.tool_encode_vs`, which uses a fixed-length keyed payload
with no model query at all.

**Result, same field/sentence as the bidi-only check**
(`regfiling_0000`/`jurisdiction_code`), sentence: *"The company operates
within the jurisdiction of VIC-23, effective as of 15 October 2023."*
(88 chars):
- Permutation search (2000 trials) + logprob-guided VS injection:
  Hamming distance **86/88** (near-total displacement), 1-60 VS
  iterations per character (mean 9.9).
- Full-document round-trip verified: renders byte-identical to the
  original after splicing the encoded sentence back in.
- Stored/tokenized sequence: `'.-23,ICV5 Obtocer3202 1f  efe afetivco s
  fo itconitin heir ujsdhtwi payn poretsea ehTcom'`
- **`gpt-5.6-sol` query: empty response, extraction failed. Defended =
  True.**

**This is the opposite outcome from the field-only bidi-only check**,
where `gpt56_sol` extracted `'VIC-23'` correctly even at the maximum
possible Hamming distance for the bare field. Sentence-scope permutation
combined with real logprob-guided VS injection defeated this model
outright on the same underlying field, where bidi-only field-scope
obfuscation didn't. This is the first real evidence (beyond the
GHOST-Agent sentence-scope run's own separate finding) that BOTH scope
(sentence vs. field) AND mechanism (real VS injection vs. bidi alone)
matter substantially against this specific model — consistent with the
"Prior strength is about injection complexity, not a content-type wall"
framing already in this file. Still N=1 — don't generalize past this
single data point; the empty (rather than refused) response from
`gpt56_sol` is also itself worth checking against a few more sentences
before treating "empty response = defended" as reliable for this model
specifically, since an empty completion has looked like a token-budget
issue for reasoning models elsewhere in this project (see the earlier
`gpt56_sol`/`gemini_31_pro` `max_tokens` bugs) rather than a genuine
extraction failure — not yet ruled out here.

Full result saved to
`results/raw/verify_ghost_permute_sentence_track_a_jurisdiction_code.json`.

### GHOST-Agent given the static pipeline's real tools (2026-08-16)

Direct follow-up question after the sentence+VS static result above:
the agent SHOULD be able to reach this level and learn why it wins —
what's stopping it? Answer, from direct inspection, was a real
STRUCTURAL capability gap, not just the stopping-rule issue already
diagnosed earlier: `ghost_tools.py`'s `tool_encode_bidi` only exposed 6
manually-parameterized config families (no searched permutation), and
`tool_encode_vs` only exposed a fixed-length keyed payload (no
logprob-guided adaptive depth) — the agent could never have proposed
the static test's encoding because the mechanisms that produced it
were never available as tools. Two smaller compounding factors also
apply: the agent doesn't choose field-vs-sentence SCOPE itself (that's
the caller's `target` argument, decided before the loop starts), and
its reflection has no ceiling reference (`tool_hamming` returns a raw
integer, never "X% of theoretical max"), so even a strong result
wouldn't be recognizable to the agent AS strong.

**Fix — three new tools added to `ghost_tools.py`, giving the agent the
SAME primitives the static pipeline used:**
- `tool_bidi_permute(text, trials=2000)` — wraps
  `bidi_permute.find_target_permutation` + `decompose` + `encode`. No
  proxy model needed (pure search + a real bidi-render check), always
  available.
- `tool_encode_vs_logprob(text, payload, threshold_tau)` — wraps
  `encode.encode_char_with_vs_logprob` per character, keyed the same
  way `tool_encode_vs` already is (`derive_payload_bytes(payload,
  salt, char_index, n_bytes=64)`) but depth is now ADAPTIVE (driven by
  a real proxy logprob check) instead of fixed by payload length.
- `tool_combine_permute(text, trials, payload, threshold_tau)` — both
  of the above in one call, mirroring `tool_combine`'s role as the
  strongest single tool. Same bidi-then-VS ordering rule as `combine`.

Both `encode_vs_logprob`/`combine_permute` need a live proxy model,
which is caller-owned (never loaded by `ghost_tools.py` itself) via new
`set_proxy_model()`/`clear_proxy_model()` module-level functions — a
call without one first raises a clear `RuntimeError`, caught by
`execute_tool_call`'s existing try/except and surfaced as a normal
`"ERROR: ..."` tool result, so a run without a proxy degrades
gracefully (falls back to the original 6 tools) instead of crashing.
**GPU-sharing between the proxy and the search-tier ensemble/agent
backbone is NOT yet wired up** (no `release_gpu()`-style coordination
for the proxy specifically) — whoever loads it is responsible for not
colliding with either, same as `verify_ghost_permute_sentence_track_a.py`'s
own explicit load/del/`empty_cache` lifecycle.

**Wiring**: `agent_backbone.py`'s `TOOL_DEFINITIONS` gained matching
schema entries for all three (auto-propagates into
`LocalReActBackbone`'s text-protocol description too, since
`_build_tool_protocol_text` builds itself from `TOOL_DEFINITIONS`).
`extract_config_from_tool_calls` now recognizes `bidi_permute`/
`combine_permute` (recorded as `f"bidi_permute(trials={trials})"` since
a searched permutation has no short config string worth printing) and
`encode_vs_logprob`. `run_ghost_agent` gained a `proxy=` param — when
given, calls `set_proxy_model()`/`clear_proxy_model()` around the loop
automatically, so a caller just loads the proxy and passes it in,
without needing to know `ghost_tools`' internals.

**Prompt updated** (`ghost_agent_prompt.py`): the agent is now told
about all three new tools, instructed to try them first and silently
fall back to the original 6 on an `ERROR` result (no proxy loaded this
run), and pointed at the concrete evidence for why
`combine_permute`/`bidi_permute` are worth preferring — the sentence+VS
static result immediately above (86/88 Hamming, defeated `gpt56_sol`
where a manually-configured encoding at the SAME maximum Hamming
distance did not).

**Verified (mocked, no GPU/API):** new
`tests/smoke_test_ghost_tools_permute.py` covers `tool_bidi_permute`'s
round-trip correctness with no proxy; both new proxy-dependent tools
correctly erroring with no proxy set, working correctly against a fake
proxy (deterministic `query_logprob` stand-in), and correctly
re-erroring after `clear_proxy_model()`; `TOOL_DEFINITIONS`/
`TOOL_FUNCTIONS` registration; and `extract_config_from_tool_calls`'s
handling of all three new tool names. All 5 existing smoke test suites
(`smoke_test_task5`, `smoke_test_agent_backbone`, `smoke_test_ensemble`,
`smoke_test_search_calibration`, plus this new one) still pass.

**Clarifying the Qwen proxy's role before the real run** (user question,
worth recording since it surfaces a residual risk not previously
flagged for the agent specifically): the agent never directly "sees"
Qwen2.5-7B — it's invoked only INSIDE `encode_vs_logprob`/
`combine_permute`'s tool implementation to decide per-character VS
depth; the agent just gets back the finished encoded string, same as
any other tool call. Qwen is NOT one of the four search-tier ensemble
judges (`agent_ensemble.members` already excludes `qwen25_7b`/
`qwen25_3b` specifically to avoid proxy-as-judge circularity — this was
already true before these tools existed). **Residual risk, not fully
resolved**: `deepseek_r1_14b` (an actual judge) is a Qwen-tokenizer-
family distill, so VS depth calibrated to fool Qwen's logprob could
transfer unusually well to `deepseek_r1_14b` specifically for shared-
tokenizer reasons, inflating that judge's "defended" vote independent
of genuine cross-tokenizer robustness — same underlying category of
risk as the `qwen25_7b`-excluded-as-judge rule, just via construction
bias instead of judge circularity. Not new to the agent — the static
pipeline has used exactly this Qwen-as-sole-constructor setup the whole
project (Qwen's own target-model result is already labeled
"proxy-optimised" and excluded from the transferability claim) — the
agent's new tools just expose the same pre-existing tension in a new
place.

**First real run (2026-08-16), Claude Haiku backbone, live Qwen2.5-7B
proxy loaded via `proxy=`** (`src/verify_agent_with_permute_tools.py`):
same field/sentence as the two static checks above
(`regfiling_0000`/`jurisdiction_code`, sentence-scope, real Track A
data — so, correctly, `frontier_check_fn` was NOT used during search,
only as a post-hoc verify-tier check, per the calibration-only
constraint). **Result: the agent used the OLD `combine()` tool
(`full_rtl` + `"ghost"` payload) — it never called `combine_permute`,
`bidi_permute`, or `encode_vs_logprob` at all**, despite all three being
available. Hamming 84/88 (respectable, but below the static
`ghost_permute_sentence` check's 86/88). Cleared the local ensemble 4/4
in 1 iteration, stopped immediately — the same weak-stopping-rule
pattern as every other real run this session. `gpt-5.6-sol` (verify
tier, post-hoc) then extracted `'VIC-23'` correctly — same failure mode
already seen repeatedly against this model.

**Root cause found and fixed**: `build_iteration_prompt`'s iteration-1
message (`ghost_agent_prompt.py`) HARDCODED "Begin by trying combine()
with 'full_rtl' config" — a concrete per-turn instruction that silently
overrode the system prompt's own new guidance to prefer
`combine_permute`. A specific per-turn instruction beats a general
system-prompt suggestion, which is exactly why the agent defaulted to
the old tool on the very first (and only) iteration it ran. Fixed:
iteration-1 message now tries `combine_permute` first (trials=2000,
payload="ghost", threshold_tau=-8.0), falling back to `combine()` only
on an `ERROR` result (no proxy loaded). Verified `smoke_test_task5.py`
still passes after this change (its `FakeClient` doesn't assert on the
literal prompt text, only on scripted tool_use responses).

**What this run clarifies about "what's agentic here" (direct user
question)**: now that the agent has access to the SAME underlying
mechanisms the static pipeline uses, the theoretical difference is no
longer "what it can build" but "whether it builds ADAPTIVELY, using
real feedback, instead of a fixed blind recipe" --- specifically: (1)
the static pipeline's objective at encoding time is blind (maximize
Hamming + one proxy's logprob threshold, uniformly, never checking if
it actually defeats extraction until later evaluation), while the
agent observes a REAL per-attempt extraction verdict before deciding
what to try next; (2) the static pipeline always runs the identical
fixed sequence for every field, while the agent can in principle branch
strategy per instance and reason about which specific judge keeps
succeeding; (3) `strategy_memory.json` gives the agent real cross-
document persistence (confirmed working, Level 3 mechanism) that the
static pipeline has zero equivalent of; (4) `tool_analyse_structure`
gives the agent content-type adaptivity the static pipeline lacks.
**This run shows those advantages are currently theoretical, not
demonstrated** — the agent didn't exercise ANY of them (no strategy
branching was needed, no memory was used since `use_memory=False`, no
adaptive tool choice happened due to the prompt bug just fixed) because
it cleared the weak bar trivially on iteration 1, exactly as static
processing would. A real test of the fixed prompt, ideally on a field
where the local ensemble does NOT cave on iteration 1, is what would
actually demonstrate agentic value over static — not yet done.

**Re-run after the prompt fix (2026-08-16, same day), confirms the fix
worked — and surfaces a new, real finding about the tool itself:** same
field/sentence, same backbone/proxy. `extract_config_from_tool_calls`
now correctly reports `bidi_config="bidi_permute(trials=2000)"`,
`vs_payload="ghost"` — the agent DID call `combine_permute` on
iteration 1 this time, exactly as intended. But: **Hamming came back
81/88 — LOWER than both the manually-configured `combine()`+`full_rtl`
attempt from the first run (84/88) and the static
`ghost_permute_sentence` check earlier (86/88).** Local ensemble still
4/4 defended, 1 iteration, and `gpt-5.6-sol` (verify tier, post-hoc)
still extracted `'VIC-23'` correctly — same failure as every other real
run against this model on this field, and this time with a WORSE
Hamming distance than the run that didn't even use the new tool.

**Root cause, found by checking the implementation**:
`tool_bidi_permute`/`tool_combine_permute` (`ghost_tools.py`) call
`find_target_permutation(text, trials=trials)` with NO seed argument —
always the function's own hardcoded default (`seed=0`). The static
pipeline's `encode_ghost_permute_sentence`, by contrast, derives a
per-field/sentence KEYED seed
(`_derive_int_seed(seed, salt, "ghost_permute_sentence", item["id"],
sentence_text)`) before searching — a genuinely different starting
point for the same 2000-trial random search. Since `find_target_permutation`
only samples 2000 of an enormous permutation space, WHICH seed you
start from matters a lot for a fixed trial budget — this instance's
default seed=0 draw happened to land on something worse than even a
naive plain-reversal baseline (a full reversal of an 88-character
non-palindromic sentence already achieves Hamming ~84-88 "for free," no
search needed). **This is a real, actionable tool limitation, not a
one-off fluke**: the agent has no way to escape a bad default-seed draw
— `payload` lets it diversify VS injection, but nothing lets it
diversify the permutation search itself. Natural fix: add an optional
`seed` parameter to `tool_bidi_permute`/`tool_combine_permute` (default
0 for back-compat) so the agent — or a calling harness — can retry with
a different seed when a given trial budget's result isn't good enough,
mirroring the flexibility `payload` already provides for VS. **Not yet
implemented** — this session's scope stopped at diagnosing it.

**Updated read on "what's agentic here"**: the prompt fix successfully
got the agent to invoke the new tool, so the earlier finding (theoretical
advantages, not yet exercised) is partially addressed — tool *selection*
is now working as intended. But this run adds a new, orthogonal
limitation: even when the agent uses the objectively "correct" strongest
tool, a hardcoded implementation detail (fixed seed) can make it perform
WORSE than either a naive fallback or the static pipeline's own use of
the identical underlying mechanism. This reinforces the running theme
across this whole investigation: "agentic" is not automatically better —
every claimed advantage (tool access, adaptivity, memory) needs the
underlying plumbing to actually support it, and this session has found a
real, fixable gap at nearly every layer checked so far (reflection
placeholder, stopping rule, missing tools, prompt override, now a fixed
search seed).

**Not yet done**: add the `seed` parameter and re-test; a field/sentence
hard enough that the agent needs 2+ iterations (everything tested so
far converges in 1, a floor effect that gives no room to observe
adaptive strategy change); a real comparison of `strategy_memory`-driven
principle reuse against a cold run on similar content.

Full results saved to
`results/raw/verify_agent_with_permute_tools_jurisdiction_code.json`
(overwritten by the re-run — the first attempt's `combine()`-based
result is only preserved in this file's own history above, not on
disk).

### Normalization-attack verification (scratch scripts, real API calls)

First real test of the normalization-aware-adversary threat model
(Threat model section above, and the still-unbuilt `06_norm_attack.sh` →
`src/norm_attack.py` stub) against an actual converged composed-GHOST
encoding rather than a hypothetical. Two scripts, both scratch/uncommitted
status (same as `verify_gemini_run.py`):
`src/test_normalization_forms.py` (pure `unicodedata`, no model/GPU, byte-
level only) and `src/verify_normalization_attack.py` (CPU-only, no local
model — 15 real frontier API calls: 5 normalization forms × 3
`frontier_verify_models`). Both run against the existing composed-GHOST
example, `results/raw/verify_gemini_account_number.json`
(`account_number = 5926847003`, search tier converged 1 iteration,
Hamming 10/10). Full results: `results/raw/norm_attack_account_number.json`.

**Byte-level result (`test_normalization_forms.py`):** NFC/NFD/NFKC/NFKD
all leave the encoding byte-identical to raw — this is what motivated the
"NFKC != strip_both" correction above.

**Frontier-model result (`verify_normalization_attack.py`), 1 field, 1
run — do not generalize past N=1:**

| form | claude_sonnet | gpt56_sol | gemini_31_pro |
|------|--------------|-----------|----------------|
| raw  | DEFENDED | EXTRACTED | EXTRACTED |
| NFC  | DEFENDED | EXTRACTED | EXTRACTED |
| NFD  | DEFENDED | EXTRACTED | **DEFENDED** |
| NFKC | DEFENDED | EXTRACTED | EXTRACTED |
| NFKD | DEFENDED | EXTRACTED | EXTRACTED |

- `gpt56_sol` extracted the true value (`5926847003`) in all 5
  conditions this run — contradicts the earlier "gap did not reproduce"
  finding above (same field, same target) where `gpt56_sol` failed to
  extract. Consistent with the already-documented `gpt56_sol`
  non-determinism gotcha (rejects `temperature=0`) — not evidence of a
  new bug, but a reminder this model's column is noise, not signal, run
  to run.
- `claude_sonnet` defended in all 5, but this time by misreading the
  digits (e.g. `3007485629`), not by a clean refusal
  (`stop_reason: refusal`) like the reconstruction-run case above — a
  second, distinct defended-but-not-blind failure mode for this model.
- `gemini_31_pro` extracted in 4/5 forms but was DEFENDED under NFD
  specifically, despite the encoding being byte-identical across all 5
  forms (confirmed above) — the divergence is coming from the model,
  not the input. Plausibly run-to-run noise given `gemini_31_pro`'s
  known reasoning-token-budget sensitivity; one data point can't
  distinguish that from a genuine NFD-specific effect. Needs repeat runs
  before drawing any conclusion.

**Next direction:** generate one more composed-GHOST example via
GHOST-Agent (`run_ghost_agent`) to get past N=1 on this test. User will
then supply 5 more examples to run the same normalization-attack check
against, to see whether the `gemini_31_pro`/NFD anomaly and the
`gpt56_sol` non-reproduction are per-field noise or something more
systematic. This has not been started yet.

## Gotchas

- `requirements.txt` pins `torch==2.7.1+cu128` — do NOT loosen this back
  to an unpinned `>=` range. This node's driver (570.211.01) tops out at
  CUDA 12.8; an unpinned install previously resolved a cu130 build that
  requires a newer driver, and `torch.cuda.is_available()` silently
  returns `False` with only a `UserWarning` (no exception) — every local
  HF model call then runs on CPU via `device_map="auto"`'s silent
  fallback, with no signal in the output that this happened. Confirmed
  this actually occurred for real runs, not just a theoretical risk (see
  GHOST-Agent's "Real reconstruction run" above). Install with
  `--extra-index-url https://download.pytorch.org/whl/cu128` (not
  `--index-url`, which would stop every other package resolving from
  PyPI). If `torch.cuda.is_available()` ever needs rechecking after a
  dependency change, do it explicitly — don't infer GPU vs CPU from
  whether a run "looked fine", since CPU fallback produces normal-looking
  output, just silently on the wrong device.
- No venv existed in the repo until this session — created with
  `uv venv --python 3.11 venv` (not the plain `python -m venv` +
  `module load python/3.11` that `slurm/00_setup.sh` uses; user prefers
  uv). `uv pip install -r requirements.txt` installed clean, `numpy`
  correctly resolved to `1.26.4` under the `<2` pin. Activate with
  `source venv/bin/activate` before running anything project-related.
- `google-generativeai` (the package `_call_google` originally used) is
  fully EOL per Google — no more updates/fixes, hard deprecation warning
  on import. Replaced with the new `google-genai` package
  (`requirements.txt` updated: `google-genai>=1.0.0`) and
  `src/adversary.py`'s `_call_google` rewritten to the new client API
  (`genai.Client(api_key=...).models.generate_content(model=..., contents=...,
  config={...})` instead of `genai.configure()` +
  `GenerativeModel(...).generate_content(...)`). **This migration is not
  yet verified against a real API call** — see the `gemini_31_pro`
  blocker note above.
- Spartan (this HPC)'s login node explicitly forbids running any code —
  not just GPU jobs (MOTD: "Do not run programs or code on the login
  node. Submit them to the queue with 'sbatch' or 'sinteractive'."). This
  applies even to trivial CPU-only API-call smoke tests. Use
  `sinteractive` for a quick interactive check, `sbatch` for anything
  longer, never a bare `python -c ...` on the login shell.
- `GOOGLE_API_KEY` is set in `~/.bash_profile` (and `GEMINI_API_KEY` is
  also set — `_call_google` / the `google-genai` client will pick
  `GOOGLE_API_KEY` since that's what's passed explicitly). Neither var is
  exported into non-interactive/tool shells automatically — source
  `~/.bash_profile` (or `~/.bashrc`) first if a script needs it and isn't
  running in your interactive login shell.
- `textattack`'s BAE/TextFooler recipes pull in TensorFlow (Universal
  Sentence Encoder constraint) in the same process that may have Qwen loaded
  on GPU via PyTorch — `_make_textattack_wrapper` forces TF onto CPU
  explicitly (`tf.config.set_visible_devices([], "GPU")`), not via
  `CUDA_VISIBLE_DEVICES`, which would break the PyTorch proxy model too.
- `requirements.txt` pins `numpy<2` — `tensorflow_hub` otherwise pulls numpy
  2.x, which breaks pandas' ABI and violates scipy's own pin.
- No hardcoded API keys anywhere — always `os.environ`. Temperature=0 for
  every model call (determinism), with one confirmed exception: `gpt-5.6-sol`
  (`config.yaml`'s `gpt56_sol`) rejects `temperature=0` outright (400
  `unsupported_value`) and only accepts its default (1). `adversary.py`'s
  `_call_openai` catches that specific error and retries without the
  `temperature` param — so `gpt56_sol` verify-tier results are NOT
  deterministic run-to-run like every other model in this repo. Also note:
  `_call_openai` sends `max_completion_tokens`, not `max_tokens` —
  `gpt-5.6-sol` rejects the latter too.

## Track A dataset generation (`plans/TRACK_A_SPEC.md`) — GENERATION COMPLETE, flags triaged (2026-08-16)

`src/track_a_generate.py` generates synthetic documents (DeepSeek-R1-Distill-
Qwen-14B, local GPU inference) for Phase 2 full (250 docs/domain, 1000 total)
into `data/track_a/full/` (`documents.jsonl`, `fields.jsonl`,
`generation_log.jsonl`, `.checkpoint.full.json`). Checkpointed per-instance
(atomic write-then-rename), so a kill only ever loses the one in-flight
instance, never more.

**Bug found and fixed (2026-08-14):** `process_instance` trusted the model's
self-reported `field_type` over the schema. For `company_type`
(`field_type: "category_code"`, description *"e.g. 'PTY LTD' or 'PUB CO'"*),
the model echoed the example VALUE `'PTY LTD'` back as the `field_type`
string itself, and `validate_field`'s strict dict lookup raised
`KeyError: 'PTY LTD'` — deterministically, on every resubmit, since
generation is seeded per `doc_id`. This is what silently ate the
`slurm/track_a_full.sh` auto-resubmit chain's `MAX_RESUBMITS=12` budget on
2026-08-13 with zero progress past instance 32 (`regfiling_0031`) — the
chain's own final log line correctly pointed at a stuck instance, not a
transient timeout. Fixed (commit `6ed54b8`): `field_type` is now always
`DOMAIN_FIELDS[domain][field_name]["field_type"]` — `field_name` is already
validated against `DOMAIN_FIELDS` at that point, so the schema is
authoritative and the model's echo is never trusted. Verified fixed live:
resumed from checkpoint (31 done), instance 32 that previously crashed now
clears normally.

**Running status:** after the fix, generation was restarted interactively
against a free A100 (not via `sbatch`, to start immediately) and ran to
158/1000 before being handed off — the interactive allocation (job
`29188267`) had only a 4h limit with no auto-resubmit logic, unlike
`slurm/track_a_full.sh`. At ~14:11 on 2026-08-14, the interactive process
(PID 210101) was stopped cleanly (SIGTERM, checkpoint intact at 158) and
`sbatch slurm/track_a_full.sh` was submitted as job `29192502` to take over
— that script's own SIGUSR1-trap + resubmit chain (`MAX_RESUBMITS=12`) means
it should now carry itself across future time-limit boundaries without
manual intervention, unless it hits a genuine bug again.

**Observed rate:** ~1.7 min/instance in practice (slower than the ~1 min/
instance pilot estimate) — full 1000-instance run is closer to ~28h than the
originally estimated 16-17h.

**To check status in a fresh session:** `squeue -u $USER` for the current
job in the `track_a_full` chain; `python3 -c "import json; print(len(json.load(open('data/track_a/full/.checkpoint.full.json'))['completed_doc_ids']))"`
for instances completed; `tail logs/track_a_full_<latest_jobid>.log` for
recent activity; grep `generation_log.jsonl` for `format_invalid`,
`field_name_mismatch`, `literal_example_copy_unresolved`,
`collision_review` flags — all triaged now, see below.

**Generation completed 2026-08-15** (`.generation_complete.full` marker
present): 1000/1000 instances, 0 `generation_failed`, `sbatch`'s
resubmit chain only needed 2 of its 12-resubmit budget.

**Flag triage (2026-08-16), all 1000 docs' `generation_log.jsonl` flags
inspected directly, not just counted:**

- **`char_span_unresolved` (73 field-instances / 67 docs) — real bug,
  now partially fixed.** This flag means the model's reported ground
  truth doesn't appear verbatim in the generated document text.
  `verify_char_span` (`src/track_a_generate.py`) only ever did an
  exact-substring check, so it missed two recoverable classes: (1)
  case/whitespace/dash differences between the reported `value` and how
  it's actually written in the prose (e.g. reported `PVT LTD`, text says
  `Pvt Ltd`) — **10/73**; (2) the same digits regrouped differently
  (e.g. reported `123 456 789 012`, text has `123 45 678 9012`) —
  **25/73**. Fixed: `verify_char_span` now retries with a
  case/whitespace/dash-insensitive flexible-regex pass, then a
  digits-only pass, before giving up. The remaining **38/73 (≈3.8% of
  all 1000 docs) are genuine paraphrase** — e.g. ground truth `QLD-10`
  but the text only says "Queensland"; `PUB CO` vs. "public company" —
  and are NOT recoverable by any string-matching fix; these fields are
  not actually verbatim-extractable from their document and stay
  `unresolved`. **Decision on excluding these 38 field rows from the
  eval-ready field set is still open** — deferred, not yet decided.
- **`field_name_mismatch` (5) / `field_missing_in_output` (3) — all
  ABN-specific, one real metadata bug found.** The model sometimes
  reports the ABN field under a different key (`abn_format`, lowercase
  `abn`) or adds unrequested extra fields. For 3 docs
  (`regfiling_0031/0054/0175`) this silently dropped the ABN field
  record entirely, but `documents.jsonl`'s `field_count` still counted
  the *requested* field list (4) instead of what actually made it into
  `fields.jsonl` (3) — a real, verified mismatch. Fixed: `field_count`
  is now derived from `len(field_records)` after the field loop, not
  `len(field_names)` before it. **Not fixed:** the model's ABN
  field-naming inconsistency itself — still a live generation quirk if
  Track A is ever regenerated or extended.
- **`format_invalid` (367, 362 of them ABN) — not a bug, a model
  capability limit.** `validate_abn_format` (`src/track_a_validators.py`)
  correctly implements the real ATO modulus-89 ABN checksum (tested
  against a real published ABN). The model was asked to generate a
  checksum-valid ABN and failed ~36% of the time. Doesn't affect FEA
  (ground truth is whatever string is in the text, checksum-valid or
  not) — just means these ABNs aren't real by checksum; worth a doc
  note if Track A framing ever implies they are.
- **`collision_review` (601) — not a quality signal.** `check_collision`
  unconditionally flags every instance containing an `ABN`/`doi_suffix`/
  `grant_id` field for "needs manual external-registry spot-check,"
  regardless of the actual value — no automated check was ever
  implemented beyond the flag. This is a **publish-policy question**
  (could a synthetic value coincidentally collide with a real
  registered identifier?), not a generation defect. Still unresolved:
  decide before HF publish whether a real spot-check is needed for
  these ~601 instances or the risk is accepted with a documented
  caveat.
- **`char_span_realigned` (3229) — benign**, the realignment mechanism
  (span drifted during generation, value still found exactly elsewhere
  in text) working as designed.

**Post-hoc data patch applied** (not a full regeneration — both fixes
are pure re-derivations from already-stored `carrier_text` +
`ground_truth`, no model call needed): backed up pre-patch files to
`data/track_a/full/_pre_triage_backup/`, then re-ran the fixed
`verify_char_span` over every `unresolved` field row (35 recovered →
`char_span_status: "realigned_posthoc"`, 38 confirmed genuine
paraphrase, left `unresolved`) and recomputed `field_count` for all
1000 docs (3 corrected) directly against the live `data/track_a/full/`
files. `generation_log.jsonl`'s own flags were updated to match
(`char_span_unresolved:X` → `char_span_realigned_posthoc:X` for the 35
recovered), so the log doesn't silently disagree with `fields.jsonl`
anymore.

**Do not publish to HuggingFace yet** — two open decisions remain: (1)
whether to exclude the 38 genuinely-unresolvable field rows from the
eval-ready field set (or hand-fix/regenerate just those), and (2)
whether the 601 `collision_review`-flagged ABN/DOI/grant-ID instances
need a real external-registry spot-check before release, or the risk is
accepted with a documented caveat. Also still worth deciding at that
point whether to bundle the release with the encoded GHOST conditions
(`src/encode.py` output) rather than shipping raw/clean documents alone,
since the paper's actual experiments run on the encoded versions, not
the raw generation.

## Seed fix for `bidi_permute`/`combine_permute`, and `convergence.py` proxy wiring (2026-08-17)

Direct follow-up to the "Agentic-ness critique and calibration" session
above: the diagnosed-but-unfixed root cause of the agent's `combine_permute`
attempt scoring WORSE (81/88) than even the older `combine()` tool (84/88)
on `regfiling_0000`'s sentence was `tool_bidi_permute`/`tool_combine_permute`
(`src/ghost_tools.py`) always calling `find_target_permutation` with the
implicit default `seed=0` — no way for the agent to escape a bad draw
within its `trials` budget.

**Fixed:** both tools now take an optional `seed: int = 0` param, threaded
straight to `bidi_permute.find_target_permutation(seed=seed)`.
`agent_backbone.py`'s `TOOL_DEFINITIONS` schemas for `bidi_permute` and
`combine_permute` gained a matching `seed` property (auto-propagates into
`LocalReActBackbone`'s text-protocol tool description too, since that's
built programmatically from `TOOL_DEFINITIONS`). `extract_config_from_tool_calls`
now records the seed actually used (e.g. `"bidi_permute(trials=2000,seed=0)"`,
was previously just `"bidi_permute(trials=2000)"`) so reflection/strategy_memory
can see it. `ghost_agent_prompt.py`'s system prompt now explicitly tells the
agent: if a `combine_permute`/`bidi_permute` result isn't near-maximal Hamming
distance, retry with a DIFFERENT seed before accepting the result as final —
this was previously not something the agent could even do, let alone was
told to try. `tests/smoke_test_ghost_tools_permute.py`'s two assertions on
the recorded config string were updated to match the new format; all smoke
tests (`smoke_test_ghost_tools_permute`, `smoke_test_search_calibration`,
`smoke_test_task5`, `smoke_test_agent_backbone`, `smoke_test_ensemble`,
`smoke_test_task6`) still pass. **Not yet re-verified against a real run**
(no GPU was live at the time this landed) — see the plan below for what
that real run should look like.

**`convergence.py` gained a `use_proxy`/`--use_proxy` param.** Previously,
`convergence.py` never passed a `proxy=` into `run_ghost_agent`, so in every
batch run so far `combine_permute`/`encode_vs_logprob` silently errored out
(caught by `execute_tool_call`'s existing try/except, surfaced as a normal
`ERROR: ...` tool result) and the agent fell back to `combine()`/`encode_bidi()`
— only `bidi_permute`'s search (no proxy needed) was ever actually reachable
in a batch context. This means **every prior `agent_convergence.csv` row
was produced without access to the real logprob-guided VS mechanism at
all** — worth remembering when reading old rows in that CSV, they're not
directly comparable to rows produced with `--use_proxy`. Fixed: `--use_proxy`
loads Qwen2.5-7B-Instruct once before the document loop (mirrors the
already-existing load-backbone-once pattern for `--agent_backbone`), passes
the SAME `proxy` object into every field's `run_ghost_agent` call, and
unloads it (`del` + `torch.cuda.empty_cache()`) in the loop's `finally`.
Verified via `smoke_test_task6.py` (mocked `run_ghost_agent_fn`, whose
`**kwargs` already absorbed the new `proxy` kwarg harmlessly) — not yet
verified against a real proxy load.

**Plan for the next real run, to be picked up directly from an active GPU
allocation (user has an A100 interactive job live as of 2026-08-17):**
1. Pilot first, NOT straight to 100-200 docs — this session's own repeated
   lesson (see the agentic-ness critique above) is that every "obvious"
   fix so far needed a real run to confirm it actually helped, and cost/time
   at this agent-loop's per-iteration ensemble load/unload rate is still
   only estimated, not measured post-seed-fix. Recommended: 10-20 docs
   first (`python src/convergence.py --n_samples 15 --max_iterations 5
   --agent_backbone claude_haiku --use_proxy`), from the repo root with
   the venv activated (`source venv/bin/activate`).
2. Backbone: `claude_haiku` (user's choice, cost reasons) — pass
   `--agent_backbone claude_haiku`.
3. Check after the pilot: (a) does the agent actually call `bidi_permute`/
   `combine_permute` with a non-zero seed on a retry when Hamming is
   low (the system-prompt instruction is new and unverified against a
   real model); (b) does `agent_hamming` in `results/tables/agent_convergence.csv`
   improve versus pre-seed-fix rows on similar field lengths; (c) real
   per-field wall-clock time with `--use_proxy` (Qwen adds a load at the
   start, but shouldn't add per-iteration cost since it's loaded once,
   unlike the search-tier ensemble which still reloads every iteration
   per the existing GPU-sharing constraint).
4. If the pilot looks reasonable (iteration counts and wall-clock in a
   plausible range, no crashes, real Hamming improvement vs. the
   `agent_convergence.csv` rows from before this fix), scale up toward the
   100-200 doc range the original question asked about, ideally in
   batches via `--start_index` so a `sinteractive` time-limit boundary
   doesn't lose progress (the CSV write is append-only, per-doc-index
   deterministic sampling already supports this).
5. This IS the direct test of the still-unproven Level 3 question from
   the self-improving audit above: does `agent_iterations`/`agent_hamming`
   in the CSV actually trend better as `doc_index` increases (memory
   accumulating real principles) now that the agent also has a working
   seed-diversification escape hatch? Plot `agent_iterations` and
   `agent_hamming` against `doc_index` once there's enough data — this
   was explicitly flagged as not yet done at any N large enough to see a
   trend.
6. Remember this batch still runs against `data/raw/documents.json`
   (field-scope, the older synthetic dataset), NOT Track A and NOT
   sentence-scope — those remain separate, not-yet-batch-wired axes (see
   "Sentence-level obfuscation" and Track A sections above). Don't
   conflate a result from this run with a Track A or sentence-scope
   claim.

## Margin-aware stopping + agent-initiated scope widening (2026-08-18)

Direct response to a readiness critique raised mid-session: GHOST-Agent
could search adaptively within a fixed problem (a given target string,
a fixed toolset), but had no way to decide the problem itself was too
narrow — scope (field vs. sentence) was always fixed externally by
whoever called `run_ghost_agent`, and the stopping rule
(`ensemble_result["defended"]`, i.e. `consensus_threshold` local models
failing) was a single binary check with no margin, which is exactly why
every real run this whole session converged in 1 iteration. An agent
that can't push past a bar it clears trivially isn't self-improving in
any meaningful sense — it's a smarter static pipeline with memory
bolted on. Two fixes, both opt-in and fully back-compat (verified: all
6 pre-existing smoke test suites pass unchanged with both left at their
defaults).

**Fix 1 — `require_unanimous` (`ghost_agent.py`).** New
`run_ghost_agent`/`_run_ghost_agent_loop` param, default `False`. When
`True`, the loop only stops at FULL ensemble consensus (every valid
member defeated), not just `consensus_threshold` (e.g. 3-of-4) — and
when it clears the weaker bar but not the full one, the per-iteration
result message now explicitly tells the agent its margin is thin and to
retry a seed/payload or call `widen_scope()`, instead of silently
accepting a narrow win. The returned dict's `"success"` key now reflects
this stricter `should_stop` bar via a new `final_success` local
variable — `memory.succeeded` (the old source of that field) is `True`
the instant ANY attempt clears the bare `consensus_threshold`, which
under `require_unanimous=True` can be an iteration this loop correctly
judged insufficient and kept going past.

**Real bug found and fixed getting this to interact correctly with the
existing calibration-only frontier gate**
(`ensemble.run_ensemble_query`'s `frontier_check_fn`, see "Agentic-ness
critique and calibration" above): the first version of `should_stop`
computed `is_full_consensus` purely from the LOCAL ensemble's
`n_failed`/`n_valid`, with no awareness that a frontier gate can
overturn `ensemble_result["defended"]` back to `False` even when the
local tier is unanimous. This meant `require_unanimous=True` could stop
the loop on "full LOCAL consensus" one line after a real frontier model
had just proven it could still extract the value — confirmed for real,
not hypothetical (see the calibration run below, first attempt).
Fixed: `should_stop = ensemble_result["defended"] and (is_full_consensus
if require_unanimous else True)` — ANDing with the (possibly
gate-overturned) `defended` flag means an overturn is never ignored,
and when `require_unanimous=False` this reduces to exactly the
pre-existing `ensemble_result["defended"]` check (unchanged behavior).
Regression test added (`tests/smoke_test_agentic_scope.py`, "Test 3")
reproducing this exact shape: local unanimous-but-frontier-overturned
on iteration 1, frontier finally also failing on iteration 2 — asserts
the loop does NOT stop after iteration 1.

**Fix 2 — `widen_scope()` tool (`ghost_tools.py`, `agent_backbone.py`,
`ghost_agent.py`, `ghost_agent_prompt.py`).** A tenth tool the agent can
call itself, mid-run, to re-target its encoding from the original
`target` to a caller-supplied `wider_context` (e.g. the sentence
containing a field) — previously this was fixed for the whole run by
the caller, with no lever for the agent to decide the current target
isn't defensible on its own. `run_ghost_agent` gained a
`wider_context: Optional[str] = None` param, registered via
`ghost_tools.set_wider_context()`/cleared via `clear_wider_context()`
(same lifecycle pattern as the existing `proxy` param). When `None`
(default), `widen_scope()` returns a harmless `ERROR` string and the
agent must keep working with `target` — zero change for every existing
caller. When given, and the agent calls `widen_scope()` mid-turn: the
loop detects this via `backbone.last_tool_calls`, flips internal
`effective_target`/`effective_clean_reference_text` to `wider_context`
(one-way, never back), resets `best_hamming`/`best_encoding` since the
scale changed, and recomputes `content_profile` from the wider text for
principle distillation. Critically, `field_ground_truth`/
`clean_reference_value` (what counts as a successful extraction) are
**never** touched by a widen — obfuscating more surrounding text must
never change what "defended" means for the underlying field. Result
dict gained `scope_widened`/`final_target` keys so this is visible
after the fact. System prompt (`ghost_agent_prompt.py`) updated to
document both fixes and instruct the agent: retry a seed/payload before
widening, and only widen after that's been tried — widening isn't a
free first move (larger encoded payload, more conspicuous).

### Real run 1 — floor effect at the OPPOSITE end (`src/verify_agentic_scope_track_a.py`)

Same Track A field as every other real check this session
(`regfiling_0000`/`jurisdiction_code = 'VIC-23'`), started at FIELD
scope (not sentence) with `wider_context` set to the field's sentence
and `require_unanimous=True`, live Qwen proxy loaded (so
`combine_permute`/seed retry are available), Claude Haiku backbone, no
`frontier_check_fn` (real eval data, not calibration — the firewall
rule holds).

**Neither new mechanism fired.** `combine_permute(seed=0, trials=2000)`
hit Hamming **6/6 — the true maximum possible** for this 6-character
field, and the search-tier ensemble came back 4/4 (full unanimity) on
iteration 1. With the bar already perfectly cleared, there was nothing
to retry or escalate past. This is a DIFFERENT floor effect from the
one `require_unanimous` was built to fix: a short field's permutation
space is small enough that a 2000-trial search at the default seed
already finds the true optimum, so the fix had nothing to bite on here
— not a flaw in the fix, just the wrong test case for exercising it.

Post-hoc frontier verify (3 real models): `claude_sonnet` DEFENDED
(garbled misread), `gpt56_sol` **EXTRACTED** `'VIC-23'` correctly
(consistent with every other result against this model on this exact
field all session — static bidi-only max-Hamming, static
`ghost_permute_sentence`, and now this), `gemini_31_pro` DEFENDED
(echoed the raw stored sequence back, not blind, just didn't solve it).
Full result:
`results/raw/verify_agentic_scope_track_a_jurisdiction_code.json`.

### Real run 2 — retry logic fires for real, on the calibration split (`src/verify_agentic_scope_calibration.py`)

To actually force a non-trivial first attempt, this run combined the
new fixes with the existing calibration-only frontier gate
(`adversary.make_frontier_check_fn`, disjoint `data/raw/calibration.json`
split, never real eval data — see "Agentic-ness critique" above for why
this firewall exists) on `cal_fin_001`/`reference_number =
'REF-7W8R4NEA'`, a field `calibrate_agent_search.py` had already shown
needs multiple iterations under this exact gate. `require_unanimous=True`,
`wider_context` set to the field's sentence, live Qwen proxy, Claude
Haiku backbone. The FIRST attempt at this run (before the should_stop
bugfix above) silently reported `success=True` after 1 iteration despite
`gpt56_sol` genuinely extracting the value — exactly the bug described
above, caught by inspecting this run's own output, then fixed and
re-run.

**After the fix, real 3-iteration trace:**

| iter | seed | trials | VS payload | threshold_tau | Hamming | local ensemble | `gpt56_sol` |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2000 | `ghost` | -8.0 | 11/12 | unanimous (3/3) | extracted — gate overturns, continues |
| 2 | 1 | 3000 | `aaaaaa` | -8.0 | **12/12 (max)** | unanimous | still extracted — gate overturns again |
| 3 | 3 | 3000 | `ghostghostghost` | **-10.0** | 12/12 (max) | unanimous | failed (`"NAE4R7-8WERF"`, garbled) — gate finally passes |

The agent genuinely varied strategy across all 3 iterations (new seed,
increased trial budget, longer/stronger VS payload, and on iteration 3
a self-initiated `threshold_tau` tightening beyond the -8.0 default
mentioned anywhere in its prompt) using real per-iteration frontier
feedback — this is the first real evidence of the retry mechanism doing
something, not just being reachable. `widen_scope()` was never called;
the agent found a win within field scope by pushing VS strength harder,
so scope escalation wasn't needed on this field.

**The clean finding: iterations 2 and 3 hit the IDENTICAL Hamming
distance (12/12, the theoretical maximum for this field) but only
iteration 3 defeated `gpt56_sol`.** The only difference was VS payload
strength/threshold. This is a controlled same-model, same-field,
same-Hamming comparison — not an anecdote — directly confirming this
session's repeated finding that Hamming distance saturating at its max
says nothing about defense against a specific model; VS injection
depth is doing real, independent work past that point.

**Caveats, not silently proceeded past:** N=1 field. This result is
calibration-split data queried directly against `gpt56_sol` during
search — by design NOT a transferability claim for that model (that's
the entire point of the firewall; a search that succeeds here should be
read as "the mechanism can be pushed to defeat this specific model when
given real feedback," not as a held-out result). `gpt56_sol`'s
already-documented non-determinism (rejects `temperature=0`) means this
exact 3-iteration trace might not replicate on a re-run. `widen_scope()`
still has ZERO real (non-mocked) exercise — both real runs so far found
success without needing it. Full result:
`results/raw/verify_agentic_scope_calibration_reference_number.json`.

**Not yet done:** a real run where `widen_scope()` actually fires (needs
a field/sentence combination where seed/payload retries alone don't
close the gap); wiring `wider_context`/`require_unanimous` into
`convergence.py`'s batch loop (both are currently only exercised via
one-off scratch scripts, same status as every other `verify_*.py` file);
distilling "VS strength matters independently of Hamming" into an
actual `strategy_memory` principle instead of leaving it as a narrative
finding in this file.
