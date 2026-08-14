"""
src/dataset.py

Phase 1: dataset generation.

INPUTS:  config.yaml
OUTPUTS: data/raw/documents.json     (main eval set, scored in Tables 3-6)
         data/raw/calibration.json   (disjoint from documents.json; used
                                      ONLY to calibrate threshold_tau in
                                      Phase 2, never scored in results)
         data/raw/gradient.json      (Phase 6 prior-strength subset)

Run with --smoke to generate a tiny dataset (5 docs/domain, no network
calls) for fast local verification before submitting to Spartan.
"""

import argparse
import json
import os
import random
import string

import yaml

try:
    from faker import Faker
except ImportError:
    Faker = None


# ── Field generators, one dict of (generator, phrase templates) per domain ──
# Each phrase template must contain exactly one "{value}" placeholder and
# render as a short, standalone sentence.

def _rand_digits(rng, n):
    return ''.join(rng.choice(string.digits) for _ in range(n))


def _rand_alnum(rng, n, upper=True):
    pool = string.ascii_uppercase + string.digits if upper else string.ascii_letters + string.digits
    return ''.join(rng.choice(pool) for _ in range(n))


def _gen_invoice_number(rng, faker):
    return f"INV-{_rand_digits(rng, 5)}"


def _gen_account_number(rng, faker):
    return _rand_digits(rng, rng.randint(8, 12))


def _gen_bsb(rng, faker):
    d = _rand_digits(rng, 6)
    return f"{d[:3]}-{d[3:]}"


def _gen_amount(rng, faker):
    value = rng.uniform(100.0, 99999.99)
    return f"{value:.2f}"


def _gen_transaction_date_iso(rng, faker):
    return faker.date_between(start_date="-6y", end_date="today").isoformat()


def _gen_reference_number(rng, faker):
    return f"REF-{_rand_alnum(rng, 8)}"


def _gen_patient_id(rng, faker):
    return _rand_digits(rng, rng.randint(7, 10))


def _gen_icd_code(rng, faker):
    letter = rng.choice(string.ascii_uppercase)
    major = _rand_digits(rng, 2)
    minor_len = rng.randint(1, 2)
    minor = _rand_alnum(rng, minor_len, upper=False).upper() if rng.random() < 0.2 else _rand_digits(rng, minor_len)
    return f"{letter}{major}.{minor}"


def _gen_dosage(rng, faker):
    unit = rng.choice(["mg", "ml", "mcg", "units"])
    return f"{rng.randint(50, 1000)}{unit}"


def _gen_prescription_ref(rng, faker):
    return f"RX-{_rand_alnum(rng, 6)}"


def _gen_appointment_date_natural(rng, faker):
    d = faker.date_between(start_date="-2y", end_date="+1y")
    return d.strftime("%-d %B %Y") if os.name != "nt" else d.strftime("%d %B %Y").lstrip("0")


def _gen_case_number(rng, faker):
    year = rng.randint(2020, 2025)
    return f"VID-{year}-{_rand_digits(rng, 5)}"


def _gen_tfn(rng, faker):
    d = _rand_digits(rng, 9)
    return f"{d[:3]}-{d[3:6]}-{d[6:]}"


def _gen_abn(rng, faker):
    d = _rand_digits(rng, 11)
    return f"{d[:2]} {d[2:5]} {d[5:8]} {d[8:]}"


def _gen_hearing_date_natural(rng, faker):
    return _gen_appointment_date_natural(rng, faker)


def _gen_filing_date_iso(rng, faker):
    return faker.date_between(start_date="-3y", end_date="today").isoformat()


_RESERVED_FIRST_OCTETS = {0, 10, 127, 169, 224, 225, 226, 227, 228, 229, 230,
                          231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
                          241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
                          251, 252, 253, 254, 255, 172, 192}


def _gen_ip_address(rng, faker):
    while True:
        o1 = rng.randint(1, 223)
        if o1 in _RESERVED_FIRST_OCTETS:
            continue
        o2, o3, o4 = rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254)
        return f"{o1}.{o2}.{o3}.{o4}"


def _gen_port(rng, faker):
    return str(rng.randint(1024, 65535))


def _gen_serial_number(rng, faker):
    return f"{_rand_alnum(rng, 5)}-{_rand_alnum(rng, 5)}-{_rand_alnum(rng, 5)}"


def _gen_firmware_version(rng, faker):
    return f"{rng.randint(0, 9)}.{rng.randint(0, 99):02d}.{rng.randint(0, 9)}"


def _gen_api_key_prefix(rng, faker):
    return _rand_alnum(rng, 8)


DOMAIN_SPEC = {
    "financial": {
        "lead_sentences": [
            "Please review the following payment details.",
            "This notice confirms the transaction below.",
            "Attached are the details for the pending transfer.",
        ],
        "fields": {
            "invoice_number": (_gen_invoice_number, ["Invoice #{value} is now due.",
                                                       "Please reference invoice {value} in all correspondence."]),
            "account_number": (_gen_account_number, ["Please transfer the funds to account {value}.",
                                                       "The destination account number is {value}."]),
            "bsb": (_gen_bsb, ["The BSB for this account is {value}.",
                                 "Use BSB {value} when processing the transfer."]),
            "amount": (_gen_amount, ["The amount payable is AUD {value}.",
                                       "A total of AUD {value} will be transferred."]),
            "transaction_date": (_gen_transaction_date_iso, ["The transaction is dated {value}.",
                                                                "Processing date: {value}."]),
            "reference_number": (_gen_reference_number, ["Reference: {value}.",
                                                            "Please quote reference {value}."]),
        },
    },
    "medical": {
        "lead_sentences": [
            "The following clinical record has been updated.",
            "Please note the details of this prescription.",
            "This entry summarises the patient's current treatment.",
        ],
        "fields": {
            "patient_id": (_gen_patient_id, ["Patient ID {value}.",
                                               "This record is filed under patient ID {value}."]),
            "icd_code": (_gen_icd_code, ["Procedure code {value} was recorded.",
                                           "The diagnosis is coded as {value}."]),
            "dosage": (_gen_dosage, ["Prescribed {value} per dose.",
                                       "The dosage is set at {value}."]),
            "prescription_ref": (_gen_prescription_ref, ["Prescription reference: {value}.",
                                                            "Ref: {value}."]),
            "appointment_date": (_gen_appointment_date_natural, ["Follow-up appointment: {value}.",
                                                                    "The next review is scheduled for {value}."]),
        },
    },
    "legal": {
        "lead_sentences": [
            "The following matter has been listed for hearing.",
            "This notice relates to the matter detailed below.",
            "Please find the case details outlined below.",
        ],
        "fields": {
            "case_number": (_gen_case_number, ["Matter no. {value}.",
                                                  "This matter is filed under case number {value}."]),
            "tfn": (_gen_tfn, ["TFN {value} is recorded on file.",
                                 "The associated TFN is {value}."]),
            "abn": (_gen_abn, ["ABN {value}.",
                                 "The registered ABN is {value}."]),
            "hearing_date": (_gen_hearing_date_natural, ["Hearing listed for {value}.",
                                                            "The matter will be heard on {value}."]),
            "filing_date": (_gen_filing_date_iso, ["Filed {value}.",
                                                      "Filing date: {value}."]),
        },
    },
    "technical": {
        "lead_sentences": [
            "The following device telemetry was captured.",
            "This log entry records a device connection event.",
            "Device status has been updated as follows.",
        ],
        "fields": {
            "ip_address": (_gen_ip_address, ["Device connected from {value}.",
                                                "Source IP address: {value}."]),
            "port": (_gen_port, ["Connection observed on port {value}.",
                                    "Listening port: {value}."]),
            "serial_number": (_gen_serial_number, ["Device SN: {value}.",
                                                      "Serial number {value} was registered."]),
            "firmware_version": (_gen_firmware_version, ["Firmware v{value}.",
                                                            "Running firmware version {value}."]),
            "api_key_prefix": (_gen_api_key_prefix, ["API key prefix: {value}.",
                                                        "Key prefix {value} was issued."]),
        },
    },
}

RARE_NOUNS = [
    "Ouagadougou", "Djibouti", "Ulaanbaatar", "Antananarivo",
    "Yamoussoukro", "Ngerulmud", "Funafuti", "Vaduz",
    "San Marino", "Naypyidaw", "Asmara", "Malabo",
    "Moroni", "Palikir", "Tarawa", "Honiara",
    "Port Vila", "Nuku'alofa", "Apia", "Suva",
]

# Used only if the SST2 download is unavailable (common on HPC compute nodes
# with restricted internet access) — small fixed fallback so the pipeline
# still runs end-to-end offline.
NATURAL_LANGUAGE_FALLBACK = [
    "The film was a genuinely uplifting experience from start to finish.",
    "A disappointing sequel that fails to capture the original's charm.",
    "Solid performances carry an otherwise predictable plot.",
    "The pacing drags in the middle third but the ending redeems it.",
    "An ambitious debut that doesn't quite stick the landing.",
    "Visually stunning, though the story feels secondary to the spectacle.",
    "A tightly written script with sharp, memorable dialogue.",
    "The lead actor's performance elevates otherwise thin material.",
    "Overlong and self-indulgent, but never boring.",
    "A crowd-pleaser that leans heavily on nostalgia.",
    "The direction is confident even when the writing wavers.",
    "A quiet, understated film that rewards patience.",
    "The humor lands more often than it misses.",
    "A technically impressive but emotionally hollow experience.",
    "The chemistry between the leads carries the weaker scenes.",
    "A messy but earnest attempt at something new.",
    "The score does a lot of heavy lifting in the emotional beats.",
    "A competent thriller that never quite becomes gripping.",
    "The ensemble cast is the film's clear strength.",
    "A flawed but memorable entry in the genre.",
]


def _make_field_subset_doc(rng, faker, domain, doc_id):
    spec = DOMAIN_SPEC[domain]
    field_names = list(spec["fields"].keys())
    k = rng.randint(3, min(5, len(field_names)))
    chosen = rng.sample(field_names, k)

    lead = rng.choice(spec["lead_sentences"])
    sentences = [lead]
    fields = {}
    for field_name in chosen:
        generator, phrase_templates = spec["fields"][field_name]
        value = generator(rng, faker)
        phrase = rng.choice(phrase_templates).format(value=value)
        sentences.append(phrase)
        fields[field_name] = value

    text = " ".join(sentences)
    return {"id": doc_id, "domain": domain, "text": text, "fields": fields}


def _generate_domain_set(rng, faker, domain, count, id_prefix, seen_texts):
    docs = []
    attempts = 0
    while len(docs) < count:
        attempts += 1
        if attempts > count * 50:
            raise RuntimeError(f"Too many collisions generating {domain} documents")
        doc_id = f"{id_prefix}_{domain[:3]}_{len(docs) + 1:03d}"
        doc = _make_field_subset_doc(rng, faker, domain, doc_id)
        if doc["text"] in seen_texts:
            continue
        seen_texts.add(doc["text"])
        docs.append(doc)
    return docs


def generate_main_and_calibration(config, n_per_domain, n_calibration_per_domain):
    seed = config["seed"]
    domains = config["domains"]

    main_rng = random.Random(seed)
    cal_rng = random.Random(seed + 1)  # independent stream, still reproducible

    main_faker = Faker()
    main_faker.seed_instance(seed)
    cal_faker = Faker()
    cal_faker.seed_instance(seed + 1)

    seen_texts = set()

    documents = []
    for domain in domains:
        documents.extend(
            _generate_domain_set(main_rng, main_faker, domain, n_per_domain, "main", seen_texts)
        )

    calibration = []
    for domain in domains:
        calibration.extend(
            _generate_domain_set(cal_rng, cal_faker, domain, n_calibration_per_domain, "cal", seen_texts)
        )

    return documents, calibration


def _load_sst2_sentences(n, seed):
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        sentences = [ds[i]["sentence"].strip() for i in range(n)]
        return sentences
    except Exception as e:
        print(f"WARNING: could not load SST2 ({e}); using offline fallback sentences.")
        return NATURAL_LANGUAGE_FALLBACK[:n]


def generate_gradient(config):
    seed = config["seed"]
    gradient_types = config["gradient_types"]
    n_per_type = config["gradient_n_per_type"]
    rng = random.Random(seed + 2)  # independent stream from main/calibration

    instances = []

    if "random_digits" in gradient_types:
        for i in range(n_per_type):
            digits = _rand_digits(rng, rng.randint(8, 12))
            instances.append({
                "id": f"grad_random_digits_{i + 1:03d}",
                "type": "random_digits",
                "text": f"The value is {digits}.",
                "target": digits,
            })

    if "formatted_identifiers" in gradient_types:
        for i in range(n_per_type):
            kind = rng.choice(["date", "phone", "isbn"])
            if kind == "date":
                y = rng.randint(2018, 2026)
                m = rng.randint(1, 12)
                d = rng.randint(1, 28)
                ident = f"{y:04d}-{m:02d}-{d:02d}"
            elif kind == "phone":
                ident = f"+61-4{rng.randint(0,9)}{rng.randint(0,9)}-{_rand_digits(rng,3)}-{_rand_digits(rng,3)}"
            else:
                ident = f"978-{rng.randint(0,9)}-{_rand_digits(rng,2)}-{_rand_digits(rng,6)}-{rng.randint(0,9)}"
            instances.append({
                "id": f"grad_formatted_identifiers_{i + 1:03d}",
                "type": "formatted_identifiers",
                "text": f"Reference: {ident}.",
                "target": ident,
            })

    if "alphanumeric_codes" in gradient_types:
        for i in range(n_per_type):
            kind = rng.choice(["plate", "product", "tracking"])
            if kind == "plate":
                code = f"{_rand_alnum(rng, 3)}-{_rand_alnum(rng, 3)}"
            elif kind == "product":
                code = f"{_rand_alnum(rng, 2)}-{_rand_alnum(rng, 5)}-{_rand_alnum(rng, 2)}"
            else:
                code = _rand_alnum(rng, 12)
            instances.append({
                "id": f"grad_alphanumeric_codes_{i + 1:03d}",
                "type": "alphanumeric_codes",
                "text": f"Code: {code}.",
                "target": code,
            })

    if "rare_proper_nouns" in gradient_types:
        pool = RARE_NOUNS[:]
        rng.shuffle(pool)
        for i in range(n_per_type):
            noun = pool[i % len(pool)]
            instances.append({
                "id": f"grad_rare_proper_nouns_{i + 1:03d}",
                "type": "rare_proper_nouns",
                "text": f"Located in {noun}.",
                "target": noun,
            })

    if "natural_language" in gradient_types:
        sentences = _load_sst2_sentences(n_per_type, config["seed"])
        for i, sentence in enumerate(sentences):
            instances.append({
                "id": f"grad_natural_language_{i + 1:03d}",
                "type": "natural_language",
                "text": sentence,
                "target": sentence,
            })

    return instances


def validate(documents, calibration, gradient, config):
    n_per_domain = len(documents) // len(config["domains"])
    assert len(documents) == n_per_domain * len(config["domains"])
    assert all(len(doc["fields"]) >= 2 for doc in documents)

    n_cal_per_domain = len(calibration) // len(config["domains"])
    assert len(calibration) == n_cal_per_domain * len(config["domains"])
    assert all(len(doc["fields"]) >= 2 for doc in calibration)

    doc_texts = [d["text"] for d in documents]
    assert len(set(doc_texts)) == len(doc_texts), "Duplicate texts in documents.json"

    cal_texts = [d["text"] for d in calibration]
    assert len(set(cal_texts)) == len(cal_texts), "Duplicate texts in calibration.json"

    assert not (set(doc_texts) & set(cal_texts)), \
        "Calibration split is not disjoint from the main document set"

    for doc in documents + calibration:
        for field_name, field_value in doc["fields"].items():
            raw_value = field_value.replace("-", "").replace(" ", "")
            raw_text = doc["text"].replace("-", "").replace(" ", "")
            assert raw_value in raw_text, \
                f"Field {field_name}={field_value} not found in doc {doc['id']}"

    n_per_type = config["gradient_n_per_type"]
    for t in config["gradient_types"]:
        count = sum(1 for g in gradient if g["type"] == t)
        assert count == n_per_type, f"Expected {n_per_type} gradient instances of type {t}, got {count}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke", action="store_true",
                         help="Generate a tiny dataset (5 docs/domain) for fast local verification")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_per_domain = 5 if args.smoke else config["n_per_domain"]
    n_calibration_per_domain = min(config["n_calibration_per_domain"], n_per_domain) if args.smoke else config["n_calibration_per_domain"]

    documents, calibration = generate_main_and_calibration(config, n_per_domain, n_calibration_per_domain)
    gradient = generate_gradient(config) if not args.smoke else generate_gradient(
        {**config, "gradient_n_per_type": 3}
    )

    validate(documents, calibration, gradient, {**config, "gradient_n_per_type": (3 if args.smoke else config["gradient_n_per_type"])})

    os.makedirs(config["data_raw_dir"], exist_ok=True)
    with open(os.path.join(config["data_raw_dir"], "documents.json"), "w") as f:
        json.dump(documents, f, indent=2)
    with open(os.path.join(config["data_raw_dir"], "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    with open(os.path.join(config["data_raw_dir"], "gradient.json"), "w") as f:
        json.dump(gradient, f, indent=2)

    print(f"Wrote {len(documents)} documents, {len(calibration)} calibration docs, "
          f"{len(gradient)} gradient instances to {config['data_raw_dir']}/")


if __name__ == "__main__":
    main()
