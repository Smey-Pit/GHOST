# GHOST

Defense that stops LLMs from correctly extracting sensitive numeric fields
(account numbers, patient IDs, case numbers, etc.) from scraped documents,
while the document still renders identically to a human reader. Two
independent Unicode-level failure modes are combined, not one: bidirectional
override reversal (model tokenizes characters in a different order than a
human sees) and logprob-guided Variation Selector injection (invisible
characters that fragment familiar digit tokens into sequences the model has
no learned representation for). Full background: `GHOST_BACKGROUND_FOR_CLAUDE_CODE.md`.
Full pipeline/results spec: `GHOST_EXPERIMENT_PLAN.md`.

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

`NFKC != strip_both`. NFKC strips VS characters but preserves bidi controls
(model still reads the wrong digit order). `strip_all`/strip_both removes
both. These are measured as separate attack conditions in Table 6 — do not
collapse them.

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
an LLM agent (`claude-sonnet-4-6`, fixed per the task doc's CRITICAL RULE 5)
iterates generate → query adversary → reflect → improve, using six tool
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
