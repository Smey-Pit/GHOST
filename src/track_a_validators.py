"""
src/track_a_validators.py

Per-field_type structural/format validity checks for Track A dataset
generation (TRACK_A_SPEC.md Step 1.2b). Each `validate_<field_type>`
function returns a plain bool -- Step 1.4 flags failures for manual
review, it never auto-discards on a validator's say alone.

The example values in track_a_prompts.DOMAIN_FIELDS's `description`
strings are guidance given to the generating model, not a grammar the
model is guaranteed to follow -- these checks are intentionally loose
structural checks, not strict parsers.
"""

import re
from datetime import datetime
import ipaddress

from dateutil import parser as dateutil_parser


# ---------------------------------------------------------------------------
# abn_format -- real ATO 11-digit weighted modulus-89 checksum
# ---------------------------------------------------------------------------

_ABN_WEIGHTS = [10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


def validate_abn_format(value: str) -> bool:
    digits = re.sub(r"\s", "", value)
    if not re.fullmatch(r"\d{11}", digits):
        return False
    digit_list = [int(c) for c in digits]
    digit_list[0] -= 1
    total = sum(d * w for d, w in zip(digit_list, _ABN_WEIGHTS))
    return total % 89 == 0


# ---------------------------------------------------------------------------
# ip_format -- syntactically valid IPv4, not private/reserved/loopback/link-local
# ---------------------------------------------------------------------------

def validate_ip_format(value: str) -> bool:
    try:
        addr = ipaddress.IPv4Address(value.strip())
    except ValueError:
        return False
    if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local:
        return False
    return True


# ---------------------------------------------------------------------------
# formatted_date -- parseable as a real calendar date
# ---------------------------------------------------------------------------

def validate_formatted_date(value: str) -> bool:
    # Phase 1 pilot produced natural phrasings like "12th day of October,
    # 2023" that dateutil's strict mode rejects outright but fuzzy=True
    # parses correctly (still correctly rejects non-dates like "not a
    # date") -- loosened per TRACK_A_SPEC.md Step 1.2b guidance.
    try:
        dt = dateutil_parser.parse(value, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return False
    return isinstance(dt, datetime)


# ---------------------------------------------------------------------------
# doi_format -- suffix portion following '10.xxxx/' (prefix excluded per spec)
# ---------------------------------------------------------------------------

def validate_doi_format(value: str) -> bool:
    value = value.strip()
    if not value or value.startswith("10.") or "/" not in value and "." not in value:
        # allow either "jmb.2025.03.014"-style or "abc123/def" style suffixes;
        # just require it isn't accidentally the full DOI including prefix
        pass
    if value.lower().startswith("10.") and "/" in value:
        return False  # looks like the full DOI, not just the suffix
    return bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9\.\-_/]*[A-Za-z0-9])?", value)) and len(value) >= 3


# ---------------------------------------------------------------------------
# orcid_format -- 'XXXX-XXXX-XXXX-XXXX' digits, last group may end in X
# ---------------------------------------------------------------------------

def validate_orcid_format(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[0-9X]", value.strip()))


# ---------------------------------------------------------------------------
# formatted_id -- structured id: jurisdiction/prefix + digits, hyphen-joined
# ---------------------------------------------------------------------------

def validate_formatted_id(value: str) -> bool:
    value = value.strip()
    if not (3 <= len(value) <= 40):
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]", value):
        return False
    return any(c.isdigit() for c in value)


# ---------------------------------------------------------------------------
# alphanumeric_code -- letters+digits, optional hyphens, e.g. serials/models
# ---------------------------------------------------------------------------

def validate_alphanumeric_code(value: str) -> bool:
    value = value.strip()
    if not (2 <= len(value) <= 30):
        return False
    # Phase 1 pilot showed model_number values like "DX-2023 Pro" -- a
    # trailing word-suffix is common for product names, so allow spaces
    # as a separator alongside hyphens, not just hyphens.
    if not re.fullmatch(r"[A-Za-z0-9]+(?:[ \-][A-Za-z0-9]+)*", value):
        return False
    has_letter = any(c.isalpha() for c in value)
    has_digit = any(c.isdigit() for c in value)
    return has_letter and has_digit


# ---------------------------------------------------------------------------
# category_code -- short entity-type code, e.g. 'PTY LTD'
# ---------------------------------------------------------------------------

def validate_category_code(value: str) -> bool:
    value = value.strip()
    # Phase 1 pilot showed models naturally write this in prose case
    # ("Pty Ltd") as well as the all-caps registry style ("PTY LTD") --
    # case-insensitive structural check.
    return bool(re.fullmatch(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+)*", value)) and 2 <= len(value) <= 20


# ---------------------------------------------------------------------------
# statute_code -- statute/section reference, loose structural check
# ---------------------------------------------------------------------------

def validate_statute_code(value: str) -> bool:
    value = value.strip()
    if not (3 <= len(value) <= 120):
        return False
    return any(c.isdigit() for c in value)


# ---------------------------------------------------------------------------
# port_number -- integer in [1, 65535]
# ---------------------------------------------------------------------------

def validate_port_number(value: str) -> bool:
    value = str(value).strip()
    if not re.fullmatch(r"\d+", value):
        return False
    port = int(value)
    return 1 <= port <= 65535


# ---------------------------------------------------------------------------
# version_string -- semantic-ish version, e.g. 'v3.14.2'
# ---------------------------------------------------------------------------

def validate_version_string(value: str) -> bool:
    return bool(re.fullmatch(r"v?\d+\.\d+(\.\d+)?", value.strip()))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

FIELD_TYPE_VALIDATORS = {
    "abn_format": validate_abn_format,
    "ip_format": validate_ip_format,
    "formatted_date": validate_formatted_date,
    "doi_format": validate_doi_format,
    "orcid_format": validate_orcid_format,
    "formatted_id": validate_formatted_id,
    "alphanumeric_code": validate_alphanumeric_code,
    "category_code": validate_category_code,
    "statute_code": validate_statute_code,
    "port_number": validate_port_number,
    "version_string": validate_version_string,
}


def validate_field(field_type: str, value: str) -> bool:
    """
    Run the structural/format validity check for `field_type` against
    `value`. Raises KeyError if `field_type` has no registered validator
    (should never happen for field_types drawn from
    track_a_prompts.DOMAIN_FIELDS -- surface this loudly rather than
    silently treating an unknown type as valid).
    """
    validator = FIELD_TYPE_VALIDATORS[field_type]
    try:
        return bool(validator(value))
    except Exception:
        return False


if __name__ == "__main__":
    # Quick smoke test against the example values in track_a_prompts.py's
    # descriptions (a real ABN checksum example, a public IP, etc).
    examples = [
        ("abn_format", "51 824 753 556", True),   # real published ABN example
        ("abn_format", "12 345 678 901", False),
        ("ip_format", "104.26.10.5", True),
        ("ip_format", "192.168.1.1", False),
        ("ip_format", "10.0.0.1", False),
        ("formatted_date", "14 March 2025", True),
        ("formatted_date", "not a date", False),
        ("doi_format", "jmb.2025.03.014", True),
        ("orcid_format", "0000-0001-2345-6789", True),
        ("orcid_format", "0000-0001-2345-678X", True),
        ("formatted_id", "VID-2025-04831", True),
        ("alphanumeric_code", "NSW-04", True),
        ("category_code", "PTY LTD", True),
        ("statute_code", "s 51(2) Corporations Act 2001", True),
        ("port_number", "8080", True),
        ("port_number", "70000", False),
        ("version_string", "v3.14.2", True),
    ]
    for field_type, value, expected in examples:
        got = validate_field(field_type, value)
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] {field_type}({value!r}) = {got} (expected {expected})")
