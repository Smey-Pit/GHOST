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
injection density over much longer text); GHOST-Agent tool/loop changes
to operate at sentence scope (the six tools are already content-agnostic,
so this is mostly a caller-side change in `convergence.py` plus a
check-extraction variant that pulls the field back out of a
sentence-scale response); real timing/cost validation of
`bidi_permute_search_trials` at sentence scale for a full dataset run;
whole-document scope (already rejected in favour of sentence scope).

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

## Track A dataset generation (`plans/TRACK_A_SPEC.md`) — IN PROGRESS as of 2026-08-14

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
`collision_review` flags — none of these have been triaged yet.

**Do not publish to HuggingFace yet** — wait until the full 1000-instance
run completes AND the validator-flagged instances above are reviewed. Also
worth deciding at that point whether to bundle the release with the encoded
GHOST conditions (`src/encode.py` output) rather than shipping raw/clean
documents alone, since the paper's actual experiments run on the encoded
versions, not the raw generation.
