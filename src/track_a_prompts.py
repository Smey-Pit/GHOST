"""
Track A Generation Prompts
===========================

Prompt-construction module for synthetic Track A document generation.
Used by the generation pipeline described in TRACK_A_SPEC.md.

This module does not perform generation itself — it only builds the
prompt strings sent to the model backbone (DeepSeek-R1-Distill-Qwen-14B).
Import `build_generation_prompt` and `build_memorization_check_prompt`
from the generation script.
"""

import random


# ---------------------------------------------------------------------------
# Domain field specifications
# ---------------------------------------------------------------------------

DOMAIN_FIELDS = {
    "public_business_filing": {
        "ABN": {
            "field_type": "abn_format",
            "description": (
                "an 11-digit Australian Business Number, formatted as "
                "either '## ### ### ###' or '###########', that satisfies "
                "the real ABN checksum validation algorithm"
            ),
        },
        "registration_date": {
            "field_type": "formatted_date",
            "description": "a calendar date in a natural written format, e.g. '14 March 2025'",
        },
        "jurisdiction_code": {
            "field_type": "alphanumeric_code",
            "description": (
                "a short code combining a state/territory abbreviation and a "
                "two-digit number, e.g. 'NSW-04' or 'VIC-11'"
            ),
        },
        "company_type": {
            "field_type": "category_code",
            "description": "a short registered entity type code, e.g. 'PTY LTD' or 'PUB CO'",
        },
    },
    "technical_documentation": {
        "serial_number": {
            "field_type": "alphanumeric_code",
            "description": (
                "a device serial number combining letters and digits, e.g. "
                "'7TR-40281-X'"
            ),
        },
        "ip_address": {
            "field_type": "ip_format",
            "description": (
                "a syntactically valid IPv4 address that is NOT in a reserved "
                "or private range (avoid 10.x, 172.16-31.x, 192.168.x, 127.x, "
                "0.x, 169.254.x) — use a plausible public-range address"
            ),
        },
        "port": {
            "field_type": "port_number",
            "description": "a TCP/UDP port number between 1 and 65535",
        },
        "firmware_version": {
            "field_type": "version_string",
            "description": "a semantic version string, e.g. 'v3.14.2'",
        },
        "model_number": {
            "field_type": "alphanumeric_code",
            "description": "a product model number combining letters and digits",
        },
    },
    "public_legal_record": {
        "case_filing_number": {
            "field_type": "formatted_id",
            "description": (
                "a court case filing number, e.g. 'VID-2025-04831' "
                "(jurisdiction code, year, sequence number)"
            ),
        },
        "docket_number": {
            "field_type": "formatted_id",
            "description": "a docket number distinct in format from the case filing number",
        },
        "hearing_date": {
            "field_type": "formatted_date",
            "description": "a calendar date in a natural written format",
        },
        "statute_reference": {
            "field_type": "statute_code",
            "description": (
                "a statute or section reference, e.g. 's 51(2) Corporations Act 2001'"
            ),
        },
    },
    "academic_identifier": {
        "grant_id": {
            "field_type": "formatted_id",
            "description": "a research grant identifier, e.g. 'DP210103847'",
        },
        "doi_suffix": {
            "field_type": "doi_format",
            "description": (
                "the suffix portion of a DOI following '10.xxxx/', e.g. "
                "'jmb.2025.03.014' — do not include the '10.xxxx/' prefix itself"
            ),
        },
        "orcid_fragment": {
            "field_type": "orcid_format",
            "description": (
                "an ORCID-style identifier fragment in the form "
                "'XXXX-XXXX-XXXX-XXXX' using digits (last group may end in X)"
            ),
        },
        "accession_number": {
            "field_type": "formatted_id",
            "description": "a dataset or sample accession number, e.g. 'GSE184032'",
        },
    },
}


# ---------------------------------------------------------------------------
# Style variants
# ---------------------------------------------------------------------------

STYLE_INSTRUCTIONS = {
    "terse_tabular": (
        "Write in a terse, tabular register — short declarative lines, "
        "minimal connecting prose, similar to a form or a summary printout. "
        "Fields may appear as short labeled lines rather than full sentences."
    ),
    "formal_prose": (
        "Write in formal, complete prose sentences, similar to an official "
        "notice or certificate. Fields should be embedded naturally within "
        "grammatically complete sentences."
    ),
    "cover_letter": (
        "Write in the style of a cover letter or transmittal note — a brief "
        "introductory sentence, then the substantive content, then a closing "
        "line. Slightly more personal in tone than a formal notice."
    ),
    "bulleted_summary": (
        "Write as a bulleted or numbered summary, with each major piece of "
        "information on its own line introduced by a short label or bullet."
    ),
    "verbose_narrative": (
        "Write as a longer narrative paragraph that explains context and "
        "background before and after presenting the required information, "
        "similar to a descriptive report rather than a form."
    ),
}

STYLE_NAMES = list(STYLE_INSTRUCTIONS.keys())


# ---------------------------------------------------------------------------
# Domain framing (what kind of document this is, for realism)
# ---------------------------------------------------------------------------

DOMAIN_FRAMING = {
    "public_business_filing": (
        "a public business registration notice or regulatory filing summary, "
        "of the kind published by a companies register or business regulator"
    ),
    "technical_documentation": (
        "a technical document such as a device configuration summary, "
        "changelog entry, or product support note"
    ),
    "public_legal_record": (
        "a public court record summary or case listing notice, of the kind "
        "published as part of an open court record"
    ),
    "academic_identifier": (
        "an academic or research administrative document, such as a grant "
        "acknowledgment, dataset citation notice, or publication metadata "
        "summary"
    ),
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def sample_style(rng: random.Random) -> str:
    """Sample one style variant uniformly at random."""
    return rng.choice(STYLE_NAMES)


def sample_fields(domain: str, field_count: int, rng: random.Random) -> list[str]:
    """Sample `field_count` distinct field names from the given domain."""
    available = list(DOMAIN_FIELDS[domain].keys())
    field_count = min(field_count, len(available))
    return rng.sample(available, field_count)


def build_generation_prompt(domain: str, style: str, field_names: list[str]) -> str:
    """
    Build the full generation prompt for one document.

    Parameters
    ----------
    domain : one of DOMAIN_FIELDS keys
    style : one of STYLE_INSTRUCTIONS keys
    field_names : list of field names (keys into DOMAIN_FIELDS[domain])
        to include in this document, length 2-5

    Returns
    -------
    str : the full prompt to send to the model
    """
    if domain not in DOMAIN_FIELDS:
        raise ValueError(f"Unknown domain: {domain}")
    if style not in STYLE_INSTRUCTIONS:
        raise ValueError(f"Unknown style: {style}")

    field_specs = DOMAIN_FIELDS[domain]
    field_desc_lines = []
    for name in field_names:
        if name not in field_specs:
            raise ValueError(f"Unknown field '{name}' for domain '{domain}'")
        spec = field_specs[name]
        field_desc_lines.append(
            f"- {name} ({spec['field_type']}): {spec['description']}"
        )
    field_desc_block = "\n".join(field_desc_lines)

    prompt = f"""You are generating a single synthetic document for a research \
dataset. The document must be entirely fictional: invent a fictional company, \
person, case, institution, or device as needed. Do not use any real company \
names, real people's names, real case numbers, real grant numbers, or any \
other real-world identifiers. If you are unsure whether something might \
resemble a real entity, invent something clearly fictional instead.

DOCUMENT TYPE: Write {DOMAIN_FRAMING[domain]}.

STYLE: {STYLE_INSTRUCTIONS[style]}

REQUIRED FIELDS: The document must naturally include the following \
{len(field_names)} pieces of information, each occurring exactly once, \
each embedded naturally in the text (not as a bare unlabeled value):

{field_desc_block}

CONSTRAINTS:
- Each field value must appear verbatim, exactly once, somewhere in the \
document text.
- Field values must not be inferable from other content in the document — \
do not include any other text that would let a reader guess a field value \
without reading it directly (e.g. do not repeat a date in two different \
phrasings, do not spell out a code elsewhere).
- The document should read as a plausible real-world document of this type \
and length (roughly 60-180 words), not a bare list of the fields.
- Do not add extra fields beyond the ones requested.
- The "e.g." examples given above for each field's format are illustrations \
of the FORMAT only. Do not reuse any example value verbatim — invent your \
own distinct value that follows the same format/pattern.

OUTPUT FORMAT: Respond with a single JSON object and nothing else (no \
markdown code fences, no commentary before or after), in exactly this shape:

{{
  "document_text": "<the full document text as one string, with \\n for line breaks if needed>",
  "fields": [
    {{
      "field_name": "<one of the required field names above, exactly as given>",
      "field_type": "<the field_type given above for this field>",
      "value": "<the exact value you used, exactly as it appears in document_text>",
      "char_span": [<start_index>, <end_index>]
    }}
  ]
}}

char_span must give the zero-indexed character offsets of `value` within \
`document_text`, such that document_text[start_index:end_index] == value \
exactly. Double-check this before responding.
"""
    return prompt


def build_memorization_check_prompt(document_text: str, prefix_fraction: float = 0.3) -> tuple[str, str]:
    """
    Build the zero-shot completion prompt used for the memorization check
    (Phase 2). Returns (prompt, true_continuation) where true_continuation
    is the ground-truth remainder to compare the model's completion against.

    Parameters
    ----------
    document_text : the full generated document
    prefix_fraction : fraction of the document (by character count) to
        reveal as the prompt; the remainder is the held-out continuation

    Returns
    -------
    (prompt, true_continuation)
    """
    split_point = int(len(document_text) * prefix_fraction)
    prefix = document_text[:split_point]
    true_continuation = document_text[split_point:]

    prompt = f"""Continue the following document naturally, completing it as \
you would expect it to continue. Respond with only the continuation text, \
no commentary.

{prefix}"""
    return prompt, true_continuation


# ---------------------------------------------------------------------------
# Convenience: one call to build a fully-specified pilot instance's prompt
# ---------------------------------------------------------------------------

def build_pilot_instance_prompt(domain: str, rng: random.Random, field_count: int | None = None):
    """
    Convenience wrapper for Phase 1 pilot generation: samples style and
    fields, builds the prompt, and returns everything needed to log the
    instance's generation metadata alongside the model's response.

    Returns
    -------
    dict with keys: domain, style, field_names, prompt
    """
    if field_count is None:
        field_count = rng.randint(2, 5)
    style = sample_style(rng)
    field_names = sample_fields(domain, field_count, rng)
    prompt = build_generation_prompt(domain, style, field_names)
    return {
        "domain": domain,
        "style": style,
        "field_names": field_names,
        "prompt": prompt,
    }


if __name__ == "__main__":
    # Quick smoke test — prints one example prompt per domain.
    rng = random.Random(168)
    for domain in DOMAIN_FIELDS:
        instance = build_pilot_instance_prompt(domain, rng)
        print(f"\n{'=' * 80}\nDOMAIN: {domain} | STYLE: {instance['style']} | FIELDS: {instance['field_names']}\n{'=' * 80}")
        print(instance["prompt"])
