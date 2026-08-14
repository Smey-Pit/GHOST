"""
src/track_a_generate.py

Track A dataset generation (TRACK_A_SPEC.md). Supports both Phase 1
(--phase pilot, 5/domain, default) and Phase 2 (--phase full,
250/domain) via the same generation loop.

Model backbone: DeepSeek-R1-Distill-Qwen-14B, local inference (GPU).
Emits <think>...</think> before its final JSON -- stripped via
unicode_utils.strip_think_tags before parsing, per CLAUDE.md's
evaluation convention for this model.

Checkpointing / resume (Phase 2's ~1 min/instance x 1000 instances is
~16-17 hours -- expected to span multiple sessions/SLURM allocations,
not one sitting): every instance is appended to the output JSONL files
and recorded in a `.checkpoint.{phase}.json` file (atomic write-then-
rename) IMMEDIATELY on completion, not batched in memory. Re-running the
exact same command (same --phase, --output_dir, --config seed) skips
any doc_id already in the checkpoint and continues from there --
nothing needs to be passed by hand to resume. Each instance's style/
field-count/field-set draw is a pure function of (seed, domain,
doc_index) via a per-instance derived RNG (derive_generation_seed),
NOT a single RNG threaded sequentially through the loop -- so skipping
already-completed instances on resume reproduces bit-for-bit what an
uninterrupted run would have produced for every remaining instance,
with no need to "replay" the skipped draws.

Outputs (to --output_dir):
  {documents,fields}.jsonl, generation_log.jsonl, .checkpoint.{phase}.json
and a printed per-domain / per-field_type summary table (read back from
the full on-disk files at the end, so it reflects the whole cumulative
run across however many sessions it took, not just this invocation).
"""

import argparse
import difflib
import json
import os
import random
import re
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from track_a_prompts import (  # noqa: E402
    DOMAIN_FIELDS,
    build_pilot_instance_prompt,
    build_memorization_check_prompt,
)
from track_a_validators import validate_field  # noqa: E402
from unicode_utils import strip_think_tags, derive_payload_bytes  # noqa: E402
from ensemble import load_local_model, unload_local_model  # noqa: E402

DOMAIN_PREFIX = {
    "public_business_filing": "regfiling",
    "technical_documentation": "techdoc",
    "public_legal_record": "legalrec",
    "academic_identifier": "acadid",
}

# Field names, per spec, whose values should be flagged (not blocked) for a
# later manual real-world-collision spot check.
COLLISION_CHECK_FIELDS = {"ABN", "doi_suffix", "grant_id"}

JSON_REINFORCEMENT = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. Respond with "
    "ONLY a single valid JSON object as specified above -- no markdown code "
    "fences, no commentary before or after."
)

MEMORIZATION_SIMILARITY_THRESHOLD = 0.6

# Field name -> lowercased literal "e.g. '...'" example strings pulled from
# each field's DOMAIN_FIELDS description. Phase 1 pilot found the model
# reproduces these verbatim often enough (even with sampling + an explicit
# "don't copy the example" instruction in the prompt) that a soft nudge
# isn't enough -- generate_with_retry treats an exact match as a forced
# regeneration, not just a logged flag.
_EXAMPLE_PATTERN = re.compile(r"e\.g\.,?\s*'([^']+)'")


def _literal_examples_for(domain: str, field_name: str) -> list:
    description = DOMAIN_FIELDS[domain][field_name]["description"]
    return [m.lower() for m in _EXAMPLE_PATTERN.findall(description)]


# ---------------------------------------------------------------------------
# Model query
# ---------------------------------------------------------------------------

def derive_generation_seed(base_seed, *context) -> int:
    """
    Deterministic per-instance torch seed, derived the same keyed way as
    unicode_utils.derive_payload_bytes (context should include doc_id and
    a call-purpose tag, e.g. ("gen", attempt) or ("mem",)) -- a single
    shared sampling seed across the corpus would make every document's
    "random" draw identical whenever the prompt happens to line up the
    same way, defeating the point of turning sampling on. Reruns with the
    same base_seed reproduce the same sequence of documents.
    """
    payload = derive_payload_bytes(base_seed, "track_a_generation_seed", *context, n_bytes=8)
    return int.from_bytes(payload, "big")


def query_model_raw(tokenizer, model, prompt: str, max_new_tokens: int,
                     do_sample: bool = False, temperature: float = 1.0,
                     top_p: float = 1.0, seed: int = None) -> str:
    """Single-turn chat-template query against an already-loaded HF model."""
    if seed is not None:
        import torch
        torch.manual_seed(seed)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **generate_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _strip_markdown_fences(text: str) -> str:
    """Defensive: the prompt forbids fences, but strip them if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _find_literal_copies(domain: str, parsed: dict) -> list:
    """
    Return the field_names in a parsed response whose reported value is
    an exact (case-insensitive) match to one of its own field's literal
    "e.g." example strings in DOMAIN_FIELDS.
    """
    offenders = []
    for entry in parsed.get("fields", []):
        field_name = entry.get("field_name")
        value = entry.get("value")
        if not field_name or not isinstance(value, str):
            continue
        if field_name not in DOMAIN_FIELDS.get(domain, {}):
            continue
        if value.strip().lower() in _literal_examples_for(domain, field_name):
            offenders.append(field_name)
    return offenders


def generate_with_retry(tokenizer, model, prompt: str, max_new_tokens: int,
                         base_seed: int, doc_id: str, domain: str,
                         temperature: float = 0.8, top_p: float = 0.9,
                         max_retries: int = 3):
    """
    Query the model and parse its response as JSON, retrying up to
    `max_retries` additional times with a reinforcement suffix appended
    on parse failure (TRACK_A_SPEC.md Step 1.3), OR on a detected literal
    copy of a field's "e.g." example value.

    Sampling (do_sample=True) with a per-(doc_id, attempt) derived seed --
    not greedy decoding -- so the model isn't anchored onto the single
    literal "e.g." example value present in every field's prompt
    description (Phase 1 pilot run 2: greedy decoding reproduced that
    exact example verbatim in 20% of fields across just 20 documents).
    Each retry gets its own derived seed so a retry is a genuinely
    different draw, not a repeat of the same failing sample.

    Sampling + an explicit "don't copy the example" prompt instruction
    (both added after pilot run 2) reduced but did NOT eliminate this --
    pilot run 3 still saw literal-example copies in 2/60 fields even
    with both in place, because the literal example is high-probability
    enough under the model's distribution that occasionally landing on
    it survives resampling. So this also checks the parsed response
    for an exact match against DOMAIN_FIELDS' quoted examples and forces
    a retry with a targeted reinforcement naming the offending field/
    example, rather than accepting it silently.

    Returns (parsed_json_or_None, raw_response_last_attempt, n_attempts,
    unresolved_literal_copy_fields). The last element is non-empty only
    if attempts were exhausted while a literal copy was still present --
    Step 1.4's "flag, don't block" convention applies to that case.
    """
    current_prompt = prompt
    raw_response = None
    last_parsed = None
    last_offenders = []
    for attempt in range(1, max_retries + 2):  # first try + max_retries retries
        seed = derive_generation_seed(base_seed, doc_id, "gen", attempt)
        raw_response = query_model_raw(
            tokenizer, model, current_prompt, max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p, seed=seed,
        )
        cleaned = strip_think_tags(raw_response)
        cleaned = _strip_markdown_fences(cleaned)
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            current_prompt = prompt + JSON_REINFORCEMENT
            continue

        offenders = _find_literal_copies(domain, parsed)
        if not offenders:
            return parsed, raw_response, attempt, []

        last_parsed, last_offenders = parsed, offenders
        examples_used = ", ".join(
            f"{name}='{next(e['value'] for e in parsed['fields'] if e.get('field_name') == name)}'"
            for name in offenders
        )
        current_prompt = prompt + (
            f"\n\nIMPORTANT: Your previous response used the literal example "
            f"value given above for: {examples_used}. Regenerate the document "
            f"using DIFFERENT, distinct fictional values for these fields -- "
            f"do not reuse the example."
        )
    return last_parsed, raw_response, max_retries + 1, last_offenders


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def verify_char_span(document_text: str, value: str, char_span):
    """
    Verify (or re-align) a field's char_span against document_text.

    Returns dict with keys: status ("ok" | "realigned" | "unresolved"),
    char_span (corrected span, or the original if unresolved).
    """
    if (
        isinstance(char_span, (list, tuple)) and len(char_span) == 2
        and 0 <= char_span[0] <= char_span[1] <= len(document_text)
        and document_text[char_span[0]:char_span[1]] == value
    ):
        return {"status": "ok", "char_span": [char_span[0], char_span[1]]}

    occurrences = [m.start() for m in re.finditer(re.escape(value), document_text)]
    if len(occurrences) >= 1:
        start = occurrences[0]
        return {"status": "realigned", "char_span": [start, start + len(value)]}

    return {"status": "unresolved", "char_span": list(char_span) if char_span else None}


def check_collision(field_name: str, value: str) -> dict:
    """
    Real-world collision flag (Step 1.4). Live external lookups are not
    required at this stage per spec -- this only marks the field as
    NEEDING a manual spot-check, it never performs one.
    """
    if field_name in COLLISION_CHECK_FIELDS:
        return {"flagged_for_review": True, "reason": "requires manual external-registry spot-check (not performed at pilot stage)"}
    return {"flagged_for_review": False, "reason": None}


def process_instance(tokenizer, model, domain: str, doc_index: int,
                      generation_max_new_tokens: int, memorization_max_new_tokens: int,
                      base_seed: int, temperature: float = 0.8, top_p: float = 0.9):
    """
    Generate, validate, and score one Track A document instance.
    Returns (document_record, field_records, log_record).

    The style/field-count/field-set draw uses an RNG seeded purely from
    (base_seed, domain, doc_index) -- a pure function of this instance's
    identity, not a shared RNG advanced sequentially across the whole
    loop. This is what makes checkpointed resume safe: skipping
    already-completed (domain, doc_index) pairs on a resumed run doesn't
    change what any remaining instance draws, since nothing about it
    depends on how many earlier instances ran in this process.
    """
    instance_rng = random.Random(
        derive_generation_seed(base_seed, domain, doc_index, "instance_sampling")
    )
    instance = build_pilot_instance_prompt(domain, instance_rng)
    style = instance["style"]
    field_names = instance["field_names"]
    prompt = instance["prompt"]

    doc_id = f"{DOMAIN_PREFIX[domain]}_{doc_index:04d}"

    parsed, raw_response, n_attempts, unresolved_literal_copies = generate_with_retry(
        tokenizer, model, prompt, generation_max_new_tokens,
        base_seed, doc_id, domain, temperature=temperature, top_p=top_p,
    )

    log_record = {
        "doc_id": doc_id,
        "domain": domain,
        "style": style,
        "field_names": field_names,
        "n_generation_attempts": n_attempts,
        "generation_failed": parsed is None,
        "flags": [],
    }
    if unresolved_literal_copies:
        log_record["flags"].append(
            f"literal_example_copy_unresolved:{unresolved_literal_copies}"
        )

    if parsed is None:
        log_record["flags"].append("json_parse_failed_after_retries")
        return None, [], log_record

    document_text = parsed.get("document_text")
    fields_reported = parsed.get("fields")

    if not isinstance(document_text, str) or not isinstance(fields_reported, list):
        log_record["flags"].append("malformed_json_shape")
        log_record["generation_failed"] = True
        return None, [], log_record

    reported_names = [f.get("field_name") for f in fields_reported]
    if set(reported_names) != set(field_names):
        log_record["flags"].append(
            f"field_name_mismatch: requested={field_names} reported={reported_names}"
        )

    document_record = {
        "doc_id": doc_id,
        "domain": domain,
        "split": None,  # assigned later (90/10 split is a Phase 2 step)
        "field_count": len(field_names),
        "carrier_text": document_text,
        "style": style,
    }

    field_records = []
    for i, field_name in enumerate(field_names, start=1):
        matches = [f for f in fields_reported if f.get("field_name") == field_name]
        if not matches:
            log_record["flags"].append(f"field_missing_in_output:{field_name}")
            continue
        reported = matches[0]
        value = reported.get("value")
        # field_name is already validated against DOMAIN_FIELDS at this point,
        # so the schema's field_type is authoritative -- the model's own
        # self-reported "field_type" is unreliable (it has echoed back the
        # description's example VALUE instead of the type name, e.g. 'PTY LTD'
        # instead of 'category_code', which crashes validate_field's lookup).
        field_type = DOMAIN_FIELDS[domain][field_name]["field_type"]
        char_span = reported.get("char_span")

        if not isinstance(value, str) or not value:
            log_record["flags"].append(f"field_value_missing:{field_name}")
            continue

        span_result = verify_char_span(document_text, value, char_span)
        if span_result["status"] == "unresolved":
            log_record["flags"].append(f"char_span_unresolved:{field_name}")
        elif span_result["status"] == "realigned":
            log_record["flags"].append(f"char_span_realigned:{field_name}")

        format_valid = validate_field(field_type, value)
        if not format_valid:
            log_record["flags"].append(f"format_invalid:{field_name}({field_type})={value!r}")

        collision = check_collision(field_name, value)
        if collision["flagged_for_review"]:
            log_record["flags"].append(f"collision_review:{field_name}")

        field_id = f"{doc_id}_f{i:02d}"
        field_records.append({
            "field_id": field_id,
            "doc_id": doc_id,
            "field_name": field_name,
            "field_type": field_type,
            "ground_truth": value,
            "char_span": span_result["char_span"],
            "extraction_prompt": (
                f"Read the following document and report the exact value of "
                f"{field_name}. Return only the value, with no additional text."
            ),
            "char_span_status": span_result["status"],
            "format_valid": format_valid,
            "collision_flagged": collision["flagged_for_review"],
        })

    # Memorization check (Step 1.5) -- implemented now so it runs
    # automatically in Phase 2; not meaningful at N=5/domain.
    try:
        mem_prompt, true_continuation = build_memorization_check_prompt(document_text)
        completion = query_model_raw(tokenizer, model, mem_prompt, memorization_max_new_tokens)
        completion = strip_think_tags(completion)
        similarity = difflib.SequenceMatcher(None, completion, true_continuation).ratio()
        log_record["memorization_similarity"] = similarity
        if similarity > MEMORIZATION_SIMILARITY_THRESHOLD:
            log_record["flags"].append(f"memorization_flag:similarity={similarity:.3f}")
    except Exception as exc:  # pragma: no cover -- defensive, log and move on
        log_record["flags"].append(f"memorization_check_error:{exc}")
        log_record["memorization_similarity"] = None

    return document_record, field_records, log_record


# ---------------------------------------------------------------------------
# Checkpointing (Step 2: resumable across sessions, not just within one)
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> set:
    """Return the set of doc_ids already completed, or empty if no checkpoint yet."""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path) as fh:
        data = json.load(fh)
    return set(data.get("completed_doc_ids", []))


def save_checkpoint(checkpoint_path: str, completed_doc_ids: set) -> None:
    """
    Atomic write-then-rename, same convention as strategy_memory.json
    elsewhere in this repo -- a crash mid-write must never leave a
    truncated/corrupt checkpoint that a resumed run would fail to parse.
    """
    directory = os.path.dirname(checkpoint_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".checkpoint_tmp_")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"completed_doc_ids": sorted(completed_doc_ids)}, fh)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def append_jsonl(path: str, record: dict) -> None:
    """Append one record and flush immediately -- never buffer across instances."""
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def reconcile_outputs_with_checkpoint(documents_path: str, fields_path: str,
                                       log_path: str, completed: set) -> None:
    """
    Instances are appended to the output files BEFORE their doc_id is
    added to the checkpoint (see main()) -- so a crash between "append
    finished" and "checkpoint saved" can leave one extra, otherwise-
    complete row in the outputs for a doc_id the checkpoint doesn't
    know about. On resume, drop any row whose doc_id isn't in
    `completed` (that instance will simply be regenerated) and dedupe
    on doc_id/field_id (keeping the last occurrence) in case the same
    row was appended twice across a crash+resume. This keeps the
    on-disk files always exactly consistent with the checkpoint at the
    start of every run, however many sessions it took to get there.
    """
    logs = [r for r in read_jsonl(log_path) if r["doc_id"] in completed]
    logs = list({r["doc_id"]: r for r in logs}.values())
    documents = [r for r in read_jsonl(documents_path) if r["doc_id"] in completed]
    documents = list({r["doc_id"]: r for r in documents}.values())
    fields = [r for r in read_jsonl(fields_path) if r["doc_id"] in completed]
    fields = list({r["field_id"]: r for r in fields}.values())
    write_jsonl(log_path, logs)
    write_jsonl(documents_path, documents)
    write_jsonl(fields_path, fields)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(log_records: list, all_field_records: list, label: str = "PILOT"):
    print("\n" + "=" * 78)
    print(f"TRACK A {label} -- GENERATION SUMMARY")
    print("=" * 78)

    domains = sorted(set(r["domain"] for r in log_records))
    for domain in domains:
        domain_logs = [r for r in log_records if r["domain"] == domain]
        n_total = len(domain_logs)
        n_success = sum(1 for r in domain_logs if not r["generation_failed"])
        print(f"\n[{domain}] generation success: {n_success}/{n_total}")

        domain_field_ids = {f["doc_id"] for r in domain_logs for f in [r]}
        domain_fields = [f for f in all_field_records if f["doc_id"].startswith(DOMAIN_PREFIX[domain])]
        by_type = {}
        for f in domain_fields:
            by_type.setdefault(f["field_type"], []).append(f)
        for field_type, records in sorted(by_type.items()):
            n = len(records)
            n_valid = sum(1 for r in records if r["format_valid"])
            print(f"    {field_type:20s} format-valid: {n_valid}/{n}")

        flagged = [(r["doc_id"], r["flags"]) for r in domain_logs if r["flags"]]
        if flagged:
            print("    Flagged instances:")
            for doc_id, flags in flagged:
                for flag in flags:
                    print(f"      {doc_id}: {flag}")

    print("\n" + "=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track A generation (resumable)")
    parser.add_argument("--phase", choices=["pilot", "full"], default="pilot",
                         help="pilot=5/domain (Phase 1, default), full=250/domain (Phase 2). "
                              "Sets n_per_domain/output_dir/filename defaults; --n_per_domain "
                              "and --output_dir override the preset if given explicitly.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser.add_argument("--n_per_domain", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--hf_id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--generation_max_new_tokens", type=int, default=2048)
    parser.add_argument("--memorization_max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="Sampled decoding (not greedy) so the model isn't anchored to the "
                              "literal 'e.g.' example in each field's prompt description -- Phase 1 "
                              "pilot found greedy decoding reproduced that example verbatim in 20%% "
                              "of fields across 20 documents.")
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    if args.phase == "pilot":
        n_per_domain = args.n_per_domain if args.n_per_domain is not None else 5
        output_dir = args.output_dir or os.path.join(repo_root, "data", "track_a", "pilot")
        documents_filename, fields_filename = "pilot_documents.jsonl", "pilot_fields.jsonl"
    else:
        n_per_domain = args.n_per_domain if args.n_per_domain is not None else 250
        output_dir = args.output_dir or os.path.join(repo_root, "data", "track_a", "full")
        documents_filename, fields_filename = "documents.jsonl", "fields.jsonl"

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    seed = config.get("seed", 168)

    os.makedirs(output_dir, exist_ok=True)
    documents_path = os.path.join(output_dir, documents_filename)
    fields_path = os.path.join(output_dir, fields_filename)
    log_path = os.path.join(output_dir, "generation_log.jsonl")
    checkpoint_path = os.path.join(output_dir, f".checkpoint.{args.phase}.json")

    completed = load_checkpoint(checkpoint_path)
    reconcile_outputs_with_checkpoint(documents_path, fields_path, log_path, completed)

    total_instances = n_per_domain * len(DOMAIN_FIELDS)
    print(f"Resuming: {len(completed)}/{total_instances} instances already completed "
          f"(checkpoint: {checkpoint_path})", flush=True)

    if len(completed) >= total_instances:
        print("All instances already completed -- nothing to generate. "
              "Skipping model load.", flush=True)
    else:
        print(f"Loading {args.hf_id} ...", flush=True)
        tokenizer, model = load_local_model(args.hf_id, args.dtype)

        try:
            for domain in DOMAIN_FIELDS:
                for doc_index in range(n_per_domain):
                    doc_id = f"{DOMAIN_PREFIX[domain]}_{doc_index:04d}"
                    if doc_id in completed:
                        continue
                    print(f"[{domain}] generating instance {doc_index + 1}/{n_per_domain} "
                          f"({doc_id}) ... [{len(completed)}/{total_instances} done overall]",
                          flush=True)
                    doc_record, field_records, log_record = process_instance(
                        tokenizer, model, domain, doc_index,
                        args.generation_max_new_tokens, args.memorization_max_new_tokens,
                        base_seed=seed, temperature=args.temperature, top_p=args.top_p,
                    )
                    # Append outputs BEFORE marking the checkpoint complete -- see
                    # reconcile_outputs_with_checkpoint's docstring for why this order matters.
                    append_jsonl(log_path, log_record)
                    if doc_record is not None:
                        append_jsonl(documents_path, doc_record)
                        for fr in field_records:
                            append_jsonl(fields_path, fr)
                    completed.add(doc_id)
                    save_checkpoint(checkpoint_path, completed)
        finally:
            unload_local_model(tokenizer, model)

    documents = read_jsonl(documents_path)
    all_fields = read_jsonl(fields_path)
    logs = read_jsonl(log_path)

    print(f"\nTotal so far: {len(documents)} documents -> {documents_path}")
    print(f"Total so far: {len(all_fields)} fields -> {fields_path}")
    print(f"Total so far: {len(logs)} generation log entries -> {log_path}")

    print_summary(logs, all_fields, label=args.phase.upper())

    done_sentinel = os.path.join(output_dir, f".generation_complete.{args.phase}")
    if len(completed) >= total_instances:
        with open(done_sentinel, "w") as fh:
            fh.write(f"{len(completed)}/{total_instances} instances complete\n")
        print(f"\nAll {total_instances} instances complete -- wrote {done_sentinel}", flush=True)
    else:
        print(f"\n{len(completed)}/{total_instances} instances done. Re-run this exact "
              f"command (same --phase/--output_dir/--config) to resume -- the checkpoint "
              f"at {checkpoint_path} picks up where this session left off.", flush=True)


if __name__ == "__main__":
    main()
