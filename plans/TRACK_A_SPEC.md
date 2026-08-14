# Track A Dataset — Project Brief and Build Spec

## Context: what this dataset is for

This dataset supports a research paper on GHOST-Agent, a defense against
LLM-based extraction of sensitive content from published documents.
GHOST-Agent obfuscates numerical/identifier fields in text using invisible
Unicode encodings (Variation Selector injection, bidirectional override
encoding) so that a human reader sees the original document unchanged,
but an LLM attempting to extract a specific field value fails to recover
it correctly.

To measure whether the defense works, we need documents with fields whose
values are impossible to guess from context — only readable by actually
processing the text. This dataset ("Track A") provides that: synthetic
documents in domains where identifiers are legitimately, routinely made
public (so the threat model is realistic), each containing several
fields with known ground-truth values.

This is NOT a dataset of real people's or companies' data. Every entity,
company, case, and identifier must be fictional. The dataset simulates
the *shape* of real public documents without using real ones.

## Why domain choice matters here

Earlier versions of this project considered medical records and bank
statements as example domains. These were rejected: nobody voluntarily
publishes real medical or banking records, so a defense scenario built
around "protecting a publicly posted bank statement" doesn't reflect a
real threat model. Track A instead uses domains where the identifiers
genuinely are meant to be public record:

1. **public_business_filing** — business registration numbers (ABNs),
   filing dates, jurisdiction codes. These are legally required to be
   public.
2. **technical_documentation** — serial numbers, IP addresses, firmware
   versions, in product manuals/changelogs. Routinely published.
3. **public_legal_record** — case/docket numbers, hearing dates, statute
   references. Public court record by design.
4. **academic_identifier** — grant IDs, DOIs, ORCID fragments, accession
   numbers. Designed to be public and citable.

## Dataset shape

- **1000 documents total, 250 per domain** (4 domains above)
- Each document contains **2-5 fields** (randomly chosen per document)
- Each field has: a name, a type (format category), a ground-truth
  value, its character position in the document, and a natural-language
  extraction prompt
- **90/10 split**: 90% "eval" (used for reported results), 10%
  "calibration" (used to tune defense hyperparameters — must never be
  touched when reporting final numbers). Split stratified per domain.
- Field values must NOT be inferable from surrounding context — a
  reader must be able to locate them in the text, not guess them.
  Ground truth are things like account-style numbers, codes, and dates
  with no semantic content a model could reconstruct without reading it.

## Output schema

Two linked JSONL files:

**documents.jsonl** — one line per document:
```json
{
  "doc_id": "string, e.g. regfiling_0142",
  "domain": "one of the 4 domain names above",
  "split": "eval | calibration",
  "field_count": "int, 2-5",
  "carrier_text": "the full document text, fields embedded naturally",
  "style": "which style variant was used (see track_a_prompts.py)"
}
```

**fields.jsonl** — one line per field, foreign-keyed to documents:
```json
{
  "field_id": "string, e.g. regfiling_0142_f01",
  "doc_id": "matches a documents.jsonl entry",
  "field_name": "human-readable label, e.g. 'Australian Business Number'",
  "field_type": "format category, e.g. abn_format, formatted_date, ip_format",
  "ground_truth": "the exact value to be extracted",
  "char_span": [start_int, end_int],
  "extraction_prompt": "rendered prompt, see template below"
}
```

Field name → prompt template: `"Read the following document and report the exact value of {field_name}. Return only the value, with no additional text."`

## Model backbone

Use **DeepSeek-R1-Distill-Qwen-14B**, local inference. This model emits
`<think>...</think>` reasoning blocks before its final answer — strip
these before parsing any output as JSON.

## Field specifications per domain

Full field specifications (names, field_types, and format descriptions)
are defined in `track_a_prompts.py`'s `DOMAIN_FIELDS` dict. Do not
duplicate them here — refer to that module as the source of truth.

## Build phases

### PHASE 1 — Pilot (5 documents per domain = 20 total). STOP after this phase for manual review. Do not proceed to Phase 2 without explicit approval.

#### Step 1.1 — Style and field sampling
Handled by `track_a_prompts.py` — see Step 1.2 below. Do not implement
separately.

#### Step 1.2 — Generation prompt construction
Do NOT write prompt templates from scratch. Use the prompt-construction
module `track_a_prompts.py` (included alongside this spec) — it already
implements:
- `DOMAIN_FIELDS`: the field list and format description per domain
- `STYLE_INSTRUCTIONS`: the 5 style variants
- `sample_style`, `sample_fields`: random sampling given a seeded
  `random.Random` instance
- `build_generation_prompt(domain, style, field_names)`: builds the
  full prompt string, including the fictional-entities constraint,
  the structured JSON output format, and the char_span accuracy
  instruction
- `build_pilot_instance_prompt(domain, rng, field_count=None)`:
  convenience wrapper that samples style + fields + field_count (2-5
  if not given) and returns `{domain, style, field_names, prompt}` in
  one call — use this directly in the generation loop
- `build_memorization_check_prompt(document_text, prefix_fraction=0.3)`:
  for Phase 2's memorization check (not needed until Phase 2)

Generation loop per instance should call `build_pilot_instance_prompt`,
send the returned `prompt` to the model, and log `domain`, `style`,
and `field_names` from the returned dict alongside the raw model
response — these are the generation-metadata fields required in
`generation_log.jsonl`.

The model's self-reported `char_span` in its JSON output is a starting
point only, not ground truth — see Step 1.4 for mandatory verification.

#### Step 1.2b — Validation module (must be written; does not yet exist)
Before running Step 1.4, implement a small validation module
(e.g. `track_a_validators.py`) with one function per `field_type`
appearing in `track_a_prompts.DOMAIN_FIELDS`, each returning a boolean
validity check:
- `abn_format`: real ABN checksum algorithm (11-digit weighted
  modulus-89 check — look up the published ATO algorithm)
- `ip_format`: valid IPv4 syntax AND not in a reserved/private range
  (use Python's `ipaddress` module: reject `is_private`, `is_reserved`,
  `is_loopback`, `is_link_local`)
- `formatted_date`: parseable as a real calendar date
- `doi_format`, `orcid_format`, `formatted_id`, `alphanumeric_code`,
  `category_code`, `statute_code`, `port_number`, `version_string`:
  structural/regex checks consistent with the example patterns given
  in each field's `description` string in `track_a_prompts.py`
  (loosen or tighten based on what Phase 1 actually produces — the
  descriptions are guidance for the model, not a strict grammar)

This module doesn't exist yet and must be written before Step 1.4 can
run. Base each check's tolerance on what Phase 1's actual outputs look
like rather than over-fitting to the example values in the prompt
descriptions.

#### Step 1.3 — Generation execution
- Query DeepSeek-R1-Distill-Qwen-14B per instance using the prompt from
  `build_pilot_instance_prompt`.
- Strip `<think>...</think>` reasoning blocks before parsing final JSON
  output.
- Parse JSON; if parsing fails, retry up to 3 times with a "return
  valid JSON only" reinforcement appended before logging a failure.

#### Step 1.4 — Automated validity checks (per instance, per field)
- **char_span verification**: substring at char_span in document_text
  must exactly equal the reported value. If mismatch, attempt
  programmatic re-alignment (search for value in document_text); if
  still unresolved, flag instance for manual review rather than
  silently correcting.
- **Format validity by field_type**: run the checks implemented in
  Step 1.2b. Flag failures for review, do not auto-discard.
- **Real-world collision check**: for ABN, DOI suffix, and grant IDs
  specifically, flag (do not block) any value that looks like it could
  collide with a real identifier, for later manual spot-check. Live
  external lookups are not required at this stage.

#### Step 1.5 — Memorization check
Not yet meaningful at N=5/domain (too small for a similarity
distribution), but implement the check now so it runs automatically in
Phase 2. Use `build_memorization_check_prompt` from `track_a_prompts.py`:
generate each document, then separately query the model to complete it
from the truncated prefix, compute string similarity between the
completion and the true continuation, flag any instance above 0.6
similarity for review.

#### Step 1.6 — HALT
Output the 20 pilot instances (`pilot_documents.jsonl` +
`pilot_fields.jsonl`) to a review directory. Print a summary table: per
domain, per field_type — generation success rate, validity-check pass
rate, any flagged instances with reasons. Do not proceed to Phase 2
without explicit go-ahead.

### PHASE 2 — Full-Scale Generation (pending Phase 1 review)
- Repeat Step 1.2–1.5 to reach 250 instances/domain (1000 total)
- Batch generation with checkpointing every 50 instances (resumable if
  interrupted — do not regenerate from scratch on failure)
- After full generation: assign 90/10 eval/calibration split,
  stratified per domain, via fixed seeded shuffle (seed=168)
- Run the memorization check across the full set; report the
  proportion below the 0.6 similarity threshold per domain (target:
  matching or exceeding the ~96-100% range reported by prior work using
  this method)
- Final outputs: `documents.jsonl`, `fields.jsonl`, `generation_log.jsonl`
  (style draws, retry counts, flagged instances), `validity_report.md`
  (summary statistics for the paper's dataset-construction section)

## Known limitation to log, not solve

DeepSeek-R1-Distill-Qwen-14B is also used elsewhere in this project as
one of four models in a "search-tier surrogate ensemble" that the
obfuscation defense is tested against during development. Using the
same model to generate this dataset means its stylistic patterns
pervade the corpus. This is a known, accepted limitation — note it in
`validity_report.md` but do not attempt to resolve it by switching
models mid-build.
