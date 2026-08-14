# GHOST Experimental Plan
# Full instructions for Claude Code execution on Spartan HPC

## Overview

This document is a complete, self-contained plan for Claude Code to
implement and run all experiments required for the GHOST paper.
Execute phases in order. Do not skip phases. Each phase produces
outputs required by subsequent phases.

---

## Repository Structure

Create the following directory structure before writing any code:

```
ghost/
├── GHOST_EXPERIMENT_PLAN.md     # this file
├── config.yaml                  # all parameters, single source of truth
├── requirements.txt
├── setup.py
├── slurm/
│   ├── 00_setup.sh
│   ├── 01_dataset.sh
│   ├── 02_encode.sh
│   ├── 03_eval_local.sh
│   ├── 04_eval_api.sh
│   ├── 05_inctx_defense.sh
│   ├── 06_norm_attack.sh
│   ├── 07_gradient.sh
│   └── 08_metrics.sh
├── src/
│   ├── unicode_utils.py         # all Unicode encoding/decoding primitives
│   ├── dataset.py               # Phase 1: dataset generation
│   ├── encode.py                # Phase 2: document encoding
│   ├── evaluate.py              # Phase 3: model evaluation
│   ├── inctx_defense.py         # Phase 4: in-context defense
│   ├── norm_attack.py           # Phase 5: normalization attack
│   ├── gradient.py              # Phase 6: prior strength gradient
│   └── metrics.py               # Phase 7: results aggregation
├── data/
│   ├── raw/                     # Phase 1 outputs
│   └── encoded/                 # Phase 2 outputs, one dir per condition
│       ├── clean/
│       ├── bae/
│       ├── textfooler/
│       ├── homochar/
│       ├── vs_only/
│       ├── bidi_only/
│       ├── ghost/
│       └── ghost_nfkc/
├── results/
│   ├── raw/                     # Phase 3-6 outputs
│   │   ├── {model}/
│   │   │   ├── {condition}/
│   │   │   │   └── results.json
│   │   │   ├── inctx_defense/
│   │   │   │   └── results.json
│   │   │   └── gradient/
│   │   │       └── results.json
│   └── tables/                  # Phase 7 outputs
│       ├── table3_main.csv
│       ├── table4_ablation.csv
│       ├── table5_inctx.csv
│       ├── table6_norm.csv
│       └── figure1_gradient.csv
└── logs/
    └── {job_name}_{timestamp}.log
```

---

## config.yaml

```yaml
# ── Reproducibility ──────────────────────────────────────────
seed: 168

# ── Dataset ──────────────────────────────────────────────────
n_per_domain: 100          # 100 per domain = 400 main dataset
n_gradient: 100            # 20 per content type = 100 gradient subset
domains:
  - financial
  - medical
  - legal
  - technical

# ── Encoding ─────────────────────────────────────────────────
threshold_tau: -20.0       # logprob threshold for VS injection stopping
                           # UPDATE this after pilot experiment
disruption_payload: "ghost_obfuscation_payload_v1"

encoding_conditions:
  - clean
  - bae
  - textfooler
  - homochar
  - vs_only
  - bidi_only
  - ghost
  - ghost_nfkc

# ── Models ───────────────────────────────────────────────────
proxy_model: "Qwen/Qwen2.5-7B-Instruct"

local_models:
  llama32_3b:
    hf_id: "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens: 50
    dtype: "bfloat16"
  llama31_8b:
    hf_id: "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: 50
    dtype: "bfloat16"
  qwen25_3b:
    hf_id: "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: 50
    dtype: "bfloat16"
  qwen25_7b:
    hf_id: "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: 50
    dtype: "bfloat16"
    is_proxy: true         # dual role: proxy + target
  deepseek_r1_14b:
    hf_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    max_new_tokens: 2048   # reasoning model needs more tokens
    dtype: "bfloat16"
    strip_think_tags: true # strip <think>...</think> before eval

api_models:
  gpt55:
    provider: "openai"
    model_id: "gpt-5.5"
    max_tokens: 50
    temperature: 0
  gpt56_sol:
    provider: "openai"
    model_id: "gpt-5.6-sol"
    max_tokens: 50
    temperature: 0
  claude_sonnet:
    provider: "anthropic"
    model_id: "claude-sonnet-4-6"
    max_tokens: 50
    temperature: 0
  claude_opus:
    provider: "anthropic"
    model_id: "claude-opus-4-6"
    max_tokens: 50
    temperature: 0
  gemini_31_pro:
    provider: "google"
    model_id: "gemini-3.1-pro-preview"
    max_tokens: 50
    temperature: 0

# ── In-context defense ────────────────────────────────────────
inctx_models:
  - gpt56_sol
  - claude_opus
inctx_shots: [5, 10, 20]
inctx_n_docs: 20           # evaluate on first 20 financial docs

# ── Gradient subset ───────────────────────────────────────────
gradient_types:
  - random_digits
  - formatted_identifiers
  - alphanumeric_codes
  - rare_proper_nouns
  - natural_language
gradient_n_per_type: 20

# ── API rate limiting ─────────────────────────────────────────
api_max_retries: 5
api_retry_base_delay: 2.0  # seconds, exponential backoff
api_requests_per_minute: 30

# ── Paths ─────────────────────────────────────────────────────
data_raw_dir: "data/raw"
data_encoded_dir: "data/encoded"
results_raw_dir: "results/raw"
results_tables_dir: "results/tables"
logs_dir: "logs"
```

---

## requirements.txt

```
# Core
torch>=2.3.0
transformers>=4.47.0
accelerate>=0.33.0
sentencepiece
protobuf

# Dataset generation
faker==24.0.0

# Encoding — baselines
textattack==0.3.10

# Evaluation
sentence-transformers==3.0.0

# API clients
openai>=1.35.0
anthropic>=0.34.0
google-generativeai>=0.7.0

# Utilities
pyyaml
tqdm
numpy
pandas
scipy

# Reproducibility
datasets>=2.20.0
```

---

## Phase 0 — Environment Setup

### Script: slurm/00_setup.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/setup_%j.log

module load python/3.11
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pre-download all HuggingFace models to avoid timeout during GPU jobs
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

models_to_download = [
    cfg["proxy_model"]
] + [m["hf_id"] for m in cfg["local_models"].values()]

for model_id in models_to_download:
    print(f"Downloading {model_id}...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    print(f"Done: {model_id}")
EOF

echo "Setup complete"
```

---

## Phase 1 — Dataset Generation

### Script: src/dataset.py

Generate 400 main documents + 100 gradient subset.
Write everything to data/raw/.

#### Implementation requirements

```python
"""
src/dataset.py

INPUTS:  config.yaml
OUTPUTS: data/raw/documents.json
         data/raw/gradient.json

documents.json: list of dicts, one per document
{
  "id": "fin_001",
  "domain": "financial",
  "text": "Invoice #INV-20481 dated 2025-03-14. Please transfer 
           AUD 4,872.50 to account 48271039 (BSB 063-000).",
  "fields": {
    "invoice_number": "INV-20481",
    "date": "2025-03-14",
    "account_number": "48271039",
    "amount": "4872.50",
    "bsb": "063000"
  }
}

gradient.json: list of dicts, one per instance
{
  "id": "grad_random_001",
  "type": "random_digits",
  "text": "The value is 48271039.",
  "target": "48271039"
}
"""
```

#### Document templates per domain

**Financial (100 documents):**
Fields to generate per document (pick 3-5 randomly):
- `invoice_number`: format INV-XXXXX, 5 random digits
- `account_number`: 8-12 random digits
- `bsb`: 6 random digits formatted XXX-XXX
- `amount`: random float 100.00-99999.99, 2 decimal places
- `transaction_date`: random date 2020-01-01 to 2025-12-31, ISO 8601
- `reference_number`: format REF-XXXXXXXX, 8 random alphanumeric

Template (vary wording across documents, do not use identical template for all):
```
"Invoice #{invoice_number} dated {transaction_date}. 
 Please transfer AUD {amount} to account {account_number} 
 (BSB {bsb}). Reference: {reference_number}."
```
Generate at least 5 distinct template variants and sample randomly.

**Medical (100 documents):**
Fields:
- `patient_id`: 7-10 random digits
- `icd_code`: valid ICD-10-AM format — one letter + 2 digits + dot + 1-4 alphanumeric (e.g. M54.5, J18.9, Z23.0)
- `dosage`: random integer 50-1000 + unit randomly chosen from [mg, ml, mcg, units]
- `prescription_ref`: format RX-XXXXXX, 6 random alphanumeric
- `appointment_date`: random date, natural language format (e.g. "14 March 2025")

Template variants:
```
"Patient ID {patient_id}. Prescribed {dosage} 
 (Ref: {prescription_ref}). Procedure code {icd_code}. 
 Follow-up: {appointment_date}."
```

**Legal (100 documents):**
Fields:
- `case_number`: format VID-YYYY-XXXXX where YYYY is year 2020-2025
- `tfn`: 9 random digits formatted XXX-XXX-XXX
- `abn`: 11 random digits formatted XX XXX XXX XXX
- `hearing_date`: random date, natural language format
- `filing_date`: random date, ISO 8601

Template variants:
```
"Matter no. {case_number}. ABN {abn}. 
 TFN {tfn}. Filed {filing_date}. 
 Hearing listed for {hearing_date}."
```

**Technical (100 documents):**
Fields:
- `ip_address`: random valid IPv4 (avoid reserved ranges)
- `port`: random integer 1024-65535
- `serial_number`: format XXXXX-XXXXX-XXXXX, random alphanumeric
- `firmware_version`: format X.XX.X, random integers
- `api_key_prefix`: 8 random alphanumeric uppercase chars

Template variants:
```
"Device SN: {serial_number} connected from 
 {ip_address} on port {port}. 
 Firmware v{firmware_version}. 
 API key prefix: {api_key_prefix}."
```

#### Gradient subset (100 instances, 20 per type)

**random_digits (20):**
```python
# Generate 20 instances of 8-12 random digit strings
# Embed in neutral carrier: "The value is {digits}."
# target = the digit string
```

**formatted_identifiers (20):**
```python
# Mix of: dates (ISO 8601), phone numbers (+61-4XX-XXX-XXX),
# ISBNs (978-X-XX-XXXXXX-X format)
# Embed in neutral carrier: "Reference: {identifier}."
# target = the identifier
```

**alphanumeric_codes (20):**
```python
# Mix of: Australian license plates (XXX-XXX format),
# product codes (XX-XXXXX-XX), tracking numbers
# Embed in neutral carrier: "Code: {code}."
# target = the code
```

**rare_proper_nouns (20):**
```python
# Uncommon place names — use this fixed list to ensure rarity:
RARE_NOUNS = [
    "Ouagadougou", "Djibouti", "Ulaanbaatar", "Antananarivo",
    "Yamoussoukro", "Ngerulmud", "Funafuti", "Vaduz",
    "San Marino", "Naypyidaw", "Asmara", "Malabo",
    "Moroni", "Palikir", "Tarawa", "Honiara",
    "Port Vila", "Nuku'alofa", "Apia", "Suva"
]
# Embed in neutral carrier: "Located in {noun}."
# target = the proper noun
```

**natural_language (20):**
```python
# Take first 20 instances from SST2 validation split
# (datasets library, split="validation", seed=168)
# text = the review sentence
# target = the review sentence (full)
```

#### Validation checks after generation

```python
# After generating all documents, verify:
assert len(documents) == 400
assert all(len(doc["fields"]) >= 2 for doc in documents)
assert len(gradient) == 100
assert sum(1 for g in gradient if g["type"] == t) == 20 \
    for t in gradient_types

# Check no two documents have identical text
texts = [doc["text"] for doc in documents]
assert len(set(texts)) == len(texts), "Duplicate document texts found"

# Check all field values appear in their document text
for doc in documents:
    for field_name, field_value in doc["fields"].items():
        # strip formatting chars for BSB/TFN/ABN comparisons
        raw_value = field_value.replace("-", "").replace(" ", "")
        assert raw_value in doc["text"].replace("-", "").replace(" ", ""), \
            f"Field {field_name}={field_value} not found in doc {doc['id']}"
```

### SLURM: slurm/01_dataset.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/dataset_%j.log

module load python/3.11
source venv/bin/activate
python src/dataset.py --config config.yaml
echo "Dataset generation complete"
```

---

## Phase 2 — Document Encoding

### Script: src/unicode_utils.py

Implement all Unicode primitives here.
All other scripts import from this module.
Do not duplicate encoding logic anywhere else.

```python
"""
src/unicode_utils.py

All Unicode encoding/decoding primitives for GHOST.
Import this module everywhere. Never reimplement these functions.
"""

import unicodedata
from typing import List

# ── Unicode control characters ────────────────────────────────
RTL  = '\u202E'   # Right-to-Left Override
PDF  = '\u202C'   # Pop Directional Formatting
BIDI_CHARS = frozenset([RTL, PDF, '\u202A', '\u202B', '\u202D',
                         '\u2066', '\u2067', '\u2068', '\u2069'])

# ── Variation Selector ranges ─────────────────────────────────
VS_RANGE_1 = range(0xFE00, 0xFE10)   # VS1-VS16
VS_RANGE_2 = range(0xE0100, 0xE01F0) # VS17-VS256
VS_CODEPOINTS = frozenset(list(VS_RANGE_1) + list(VS_RANGE_2))


def byte_to_vs(byte: int) -> str:
    """Map a single byte (0-255) to a Variation Selector codepoint."""
    if byte < 16:
        return chr(0xFE00 + byte)
    else:
        return chr(0xE0100 + byte - 16)


def encode_payload_as_vs(payload: str) -> str:
    """
    Encode payload string as Variation Selector sequence.
    Returns string of VS characters only (no base character).
    """
    vs_chars = []
    for byte in payload.encode('utf-8'):
        vs_chars.append(byte_to_vs(byte))
    return ''.join(vs_chars)


def encode_char_with_vs(base_char: str, payload: str) -> str:
    """
    Attach VS-encoded payload to a base character.
    Human sees: base_char (unchanged)
    Tokenizer sees: base_char + VS sequence
    """
    return base_char + encode_payload_as_vs(payload)


def encode_bidi(field_value: str) -> str:
    """
    Apply bidi reversal encoding to a field value.
    Stores reversed sequence under RTL override.
    Human renderer displays original order.
    Model tokenizer ingests reversed sequence + control chars.
    """
    return RTL + field_value[::-1] + PDF


def encode_ghost(field_value: str, vs_sequences: dict) -> str:
    """
    Apply full GHOST encoding: bidi first, then VS injection.

    Args:
        field_value: original field string (e.g. "48271039")
        vs_sequences: dict mapping char_index -> VS payload string
                      produced by logprob-guided injection phase.
                      Keys are indices into the REVERSED sequence.

    Returns:
        GHOST-encoded string: RTL + VS-injected reversed chars + PDF

    The VS injection applies to the stored (reversed) sequence,
    not the rendered sequence. This ensures VS characters are
    adjacent to digits in the model's token stream.
    """
    reversed_field = field_value[::-1]
    encoded_chars = []
    for i, char in enumerate(reversed_field):
        if i in vs_sequences:
            encoded_chars.append(
                encode_char_with_vs(char, vs_sequences[i])
            )
        else:
            encoded_chars.append(char)
    return RTL + ''.join(encoded_chars) + PDF


def strip_vs(text: str) -> str:
    """Remove all Variation Selector characters from text."""
    return ''.join(c for c in text if ord(c) not in VS_CODEPOINTS)


def strip_bidi(text: str) -> str:
    """Remove all bidi control characters from text."""
    return ''.join(c for c in text if c not in BIDI_CHARS)


def strip_all(text: str) -> str:
    """Remove both VS and bidi control characters."""
    return strip_bidi(strip_vs(text))


def apply_nfkc(text: str) -> str:
    """Apply NFKC normalization (normalization attack)."""
    return unicodedata.normalize('NFKC', text)


def strip_think_tags(text: str) -> str:
    """
    Strip <think>...</think> blocks from DeepSeek-R1 outputs.
    Returns only the final answer after reasoning.
    """
    import re
    # Remove think blocks (including nested content)
    cleaned = re.sub(r'<think>.*?</think>', '', text,
                     flags=re.DOTALL)
    return cleaned.strip()


def show_codepoints(text: str, max_n: int = 30) -> str:
    """
    Return human-readable codepoint representation for debugging.
    Use this in logs and smoke tests, never in production paths.
    """
    entries = []
    for c in text[:max_n]:
        cp = ord(c)
        if cp in VS_CODEPOINTS:
            label = f"VS"
        elif c in BIDI_CHARS:
            label = f"BIDI"
        else:
            label = repr(c)
        entries.append(f"U+{cp:05X}[{label}]")
    if len(text) > max_n:
        entries.append(f"...+{len(text)-max_n}")
    return ' '.join(entries)
```

### Script: src/encode.py

```python
"""
src/encode.py

INPUTS:  data/raw/documents.json
         data/raw/gradient.json
         config.yaml
         Qwen2.5-7B-Instruct loaded locally for logprob queries

OUTPUTS: data/encoded/{condition}/documents.json
         data/encoded/{condition}/gradient.json
         for all conditions in config.yaml

CHECKPOINTING:
  After encoding each document, write checkpoint.
  On restart, skip already-encoded documents.
  Checkpoint files: data/encoded/{condition}/.checkpoint.json

CRITICAL: This is the most expensive phase.
  - Logprob queries for VS injection dominate runtime.
  - checkpoint after EVERY document without exception.
  - If job times out, resubmit — it will resume.
"""
```

#### Encoding logic per condition

```python
# ── clean ─────────────────────────────────────────────────────
# No change. Copy document text verbatim.
# encoded_text = doc["text"]
# encoded_fields = doc["fields"]  # unchanged ground truth

# ── bidi_only ─────────────────────────────────────────────────
# For each field value in the document:
#   apply encode_bidi(field_value) from unicode_utils
#   replace the field value occurrence in doc["text"]
#   with the bidi-encoded version
# Ground truth remains original field values
# (bidi encoding changes what model sees, not what human sees)

def encode_bidi_document(doc, unicode_utils):
    text = doc["text"]
    encoded_fields = {}
    for field_name, field_value in doc["fields"].items():
        encoded_value = unicode_utils.encode_bidi(field_value)
        text = text.replace(field_value, encoded_value, 1)
        encoded_fields[field_name] = field_value  # GT unchanged
    return {"text": text, "fields": encoded_fields}


# ── vs_only ───────────────────────────────────────────────────
# For each field value, for each character in the field value:
#   iteratively append VS characters
#   query Qwen2.5-7B logprob of original char given obfuscated
#   stop when logprob <= threshold_tau
# This requires the proxy model to be loaded.
# Returns the VS-injected field value string.
# Ground truth remains original field values.

def encode_vs_document(doc, proxy_model, proxy_tokenizer,
                       payload, threshold_tau, device):
    """
    For each field in doc, apply logprob-guided VS injection
    to each character.

    proxy_model: loaded Qwen2.5-7B model
    proxy_tokenizer: corresponding tokenizer
    payload: disruption payload string from config
    threshold_tau: logprob stopping threshold

    Returns dict with encoded text and original field GT.
    """
    text = doc["text"]
    for field_name, field_value in doc["fields"].items():
        encoded_field = ""
        for char in field_value:
            encoded_char = encode_char_with_vs_logprob(
                char, payload, proxy_model,
                proxy_tokenizer, threshold_tau, device
            )
            encoded_field += encoded_char
        text = text.replace(field_value, encoded_field, 1)
    return {"text": text, "fields": doc["fields"]}  # GT unchanged


def encode_char_with_vs_logprob(base_char, payload,
                                 model, tokenizer,
                                 threshold_tau, device):
    """
    Iteratively inject VS characters into base_char until
    logprob of original char falls below threshold_tau.

    Returns the VS-injected character string.
    """
    from unicode_utils import byte_to_vs, encode_payload_as_vs
    import torch

    current = base_char
    payload_bytes = payload.encode('utf-8')
    byte_idx = 0

    for _ in range(len(payload_bytes)):
        # Append next VS character
        vs_char = byte_to_vs(payload_bytes[byte_idx % len(payload_bytes)])
        candidate = current + vs_char

        # Query logprob of original base_char given candidate
        logprob = query_logprob(
            base_char, candidate, model, tokenizer, device
        )

        current = candidate
        byte_idx += 1

        if logprob <= threshold_tau:
            break

    return current


def query_logprob(target_token: str, context: str,
                  model, tokenizer, device) -> float:
    """
    Query the logprobability of target_token appearing
    in context using teacher-forced forward pass.

    Returns log probability as a float (negative value).
    Lower = model less likely to recover original token.
    """
    import torch

    # Tokenize context
    inputs = tokenizer(context, return_tensors="pt").to(device)

    # Tokenize target separately to get target token id
    target_ids = tokenizer(target_token,
                           add_special_tokens=False,
                           return_tensors="pt").input_ids

    if target_ids.shape[1] == 0:
        return float('-inf')  # untokenizable = already defeated

    target_id = target_ids[0, 0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for last position
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        logprob = log_probs[target_id].item()

    return logprob


# ── ghost ─────────────────────────────────────────────────────
# Step 1: apply bidi encoding (reverse the field value)
# Step 2: apply VS injection to each char of the REVERSED value
# (not the original — VS must be adjacent to stored chars)
# Ground truth remains original field values.

def encode_ghost_document(doc, proxy_model, proxy_tokenizer,
                          payload, threshold_tau, device):
    text = doc["text"]
    for field_name, field_value in doc["fields"].items():
        # Step 1: reverse for bidi storage
        reversed_value = field_value[::-1]

        # Step 2: VS inject on reversed chars
        encoded_reversed = ""
        for char in reversed_value:
            encoded_char = encode_char_with_vs_logprob(
                char, payload, proxy_model,
                proxy_tokenizer, threshold_tau, device
            )
            encoded_reversed += encoded_char

        # Step 3: wrap in bidi controls
        from unicode_utils import RTL, PDF
        encoded_field = RTL + encoded_reversed + PDF

        text = text.replace(field_value, encoded_field, 1)

    return {"text": text, "fields": doc["fields"]}  # GT unchanged


# ── ghost_nfkc ────────────────────────────────────────────────
# Apply ghost encoding, then apply NFKC normalization.
# This is the normalization attack condition.
# Produces what an adversary with NFKC preprocessing sees.
# Note: NFKC strips VS but preserves bidi controls.
# The resulting text has bidi controls but no VS fragmentation.

def encode_ghost_nfkc_document(ghost_encoded_doc):
    import unicodedata
    text = unicodedata.normalize('NFKC', ghost_encoded_doc["text"])
    return {"text": text, "fields": ghost_encoded_doc["fields"]}


# ── bae, textfooler, homochar ─────────────────────────────────
# Use TextAttack library for these baselines.
# Apply to FULL document text, not just field values.
# After encoding, re-extract field values from encoded text.
# If a field value is altered by the baseline (e.g. digit
# replaced by word), record the MODIFIED value as new GT
# for that condition and flag the instance.

def encode_with_textattack(doc, attack_name: str):
    """
    Apply TextAttack baseline to full document text.
    attack_name: one of "bae", "textfooler", "homochar"

    Returns encoded text and updated fields dict.
    Flags instances where field values were modified.
    """
    # TextAttack setup
    from textattack.attack_recipes import (
        BAEGarg2019,
        TextFoolerJin2019,
    )
    # HOMOCHAR uses character-level attack
    # Implement using textattack CharacterSubstitution
    # or direct homoglyph mapping if TextAttack version
    # does not include HOMOCHAR directly.

    # After attacking, check if original field values
    # still appear in encoded text.
    # If not, update GT for this instance and set flag.
    pass  # implement fully
```

#### Checkpointing implementation

```python
import json
import os

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint dict {doc_id: encoded_doc}."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: str, checkpoint: dict):
    """Atomically save checkpoint to avoid corruption."""
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(checkpoint, f)
    os.replace(tmp_path, checkpoint_path)


# Usage pattern in encode.py main loop:
# checkpoint = load_checkpoint(checkpoint_path)
# for doc in documents:
#     if doc["id"] in checkpoint:
#         continue  # already encoded
#     encoded = encode_xxx_document(doc, ...)
#     checkpoint[doc["id"]] = encoded
#     save_checkpoint(checkpoint_path, checkpoint)
```

### SLURM: slurm/02_encode.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_encode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/encode_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate

# Encode all conditions sequentially
# Most conditions are cheap; vs_only and ghost are expensive
# (logprob queries against Qwen2.5-7B)

python src/encode.py \
    --config config.yaml \
    --conditions clean bidi_only homochar

python src/encode.py \
    --config config.yaml \
    --conditions vs_only

python src/encode.py \
    --config config.yaml \
    --conditions ghost

python src/encode.py \
    --config config.yaml \
    --conditions ghost_nfkc bae textfooler

echo "All encoding complete"
```

---

## Phase 3 — Model Evaluation

### Script: src/evaluate.py

```python
"""
src/evaluate.py

INPUTS:  data/encoded/{condition}/documents.json  (all conditions)
         data/encoded/{condition}/gradient.json
         config.yaml

OUTPUTS: results/raw/{model_name}/{condition}/results.json

results.json structure (list of dicts):
{
  "doc_id": "fin_001",
  "domain": "financial",
  "field_name": "account_number",
  "ground_truth": "48271039",
  "model_response": "48271039",
  "exact_match": 1,
  "condition": "ghost",
  "model": "qwen25_7b",
  "is_proxy_model": true,
  "flagged": false    # true if baseline modified field GT
}

EVALUATION PROMPT (field extraction):
  "Read the following document carefully and report the exact 
   value of {FIELD_NAME}. Return only the value with no 
   additional text or explanation.\n\nDocument:\n{DOCUMENT_TEXT}"

TEMPERATURE: 0 for all models (deterministic output)
MAX_NEW_TOKENS: 50 for standard models, 2048 for DeepSeek
POST-PROCESSING:
  - Strip whitespace from model response
  - For DeepSeek: strip <think>...</think> blocks first
  - Exact match against ground truth (case-sensitive)
  - Normalise formatting chars in both GT and response
    before comparison (strip dashes and spaces for
    BSB/TFN/ABN comparisons)

CHECKPOINTING:
  Same pattern as encode.py.
  Checkpoint per model per condition.
  results/raw/{model}/{condition}/.checkpoint.json
"""
```

#### Local model inference

```python
def run_local_model(model_cfg: dict, condition: str,
                    documents: list, config: dict):
    """
    Load model, evaluate all documents for one condition,
    save results, unload model.

    CRITICAL: unload model and clear CUDA cache after each
    model to ensure next model loads cleanly.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_id = model_cfg["hf_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    results = []
    checkpoint = load_checkpoint(checkpoint_path)

    for doc in documents:
        for field_name, ground_truth in doc["fields"].items():
            instance_id = f"{doc['id']}_{field_name}"
            if instance_id in checkpoint:
                results.append(checkpoint[instance_id])
                continue

            prompt = build_field_extraction_prompt(
                doc["text"], field_name
            )
            response = generate_local(
                prompt, model, tokenizer, device,
                max_new_tokens=model_cfg["max_new_tokens"]
            )

            if model_cfg.get("strip_think_tags"):
                from unicode_utils import strip_think_tags
                response = strip_think_tags(response)

            response = response.strip()
            exact_match = normalise_and_compare(
                response, ground_truth
            )

            result = {
                "doc_id": doc["id"],
                "domain": doc["domain"],
                "field_name": field_name,
                "ground_truth": ground_truth,
                "model_response": response,
                "exact_match": int(exact_match),
                "condition": condition,
                "model": model_cfg["name"],
                "is_proxy_model": model_cfg.get("is_proxy", False),
            }
            results.append(result)
            checkpoint[instance_id] = result
            save_checkpoint(checkpoint_path, checkpoint)

    # Unload model
    del model
    torch.cuda.empty_cache()
    print(f"Unloaded {hf_id}")

    return results


def normalise_and_compare(response: str,
                          ground_truth: str) -> bool:
    """
    Exact match with formatting normalisation.
    Strips dashes, spaces for BSB/TFN/ABN comparisons.
    Case-sensitive.
    """
    def normalise(s):
        return s.strip().replace("-", "").replace(" ", "")
    return normalise(response) == normalise(ground_truth)
```

#### API model inference

```python
def run_api_model(model_cfg: dict, condition: str,
                  documents: list, config: dict):
    """
    Evaluate one API model on all documents for one condition.
    Implements exponential backoff for rate limits.
    Checkpoints after every instance.
    """
    import time

    provider = model_cfg["provider"]
    results = []
    checkpoint = load_checkpoint(checkpoint_path)

    for doc in documents:
        for field_name, ground_truth in doc["fields"].items():
            instance_id = f"{doc['id']}_{field_name}"
            if instance_id in checkpoint:
                results.append(checkpoint[instance_id])
                continue

            prompt = build_field_extraction_prompt(
                doc["text"], field_name
            )

            # API call with retry
            response = call_api_with_retry(
                provider, model_cfg["model_id"],
                prompt, model_cfg["max_tokens"],
                model_cfg["temperature"],
                max_retries=config["api_max_retries"],
                base_delay=config["api_retry_base_delay"]
            )

            response = response.strip()
            exact_match = normalise_and_compare(
                response, ground_truth
            )

            result = {
                "doc_id": doc["id"],
                "domain": doc["domain"],
                "field_name": field_name,
                "ground_truth": ground_truth,
                "model_response": response,
                "exact_match": int(exact_match),
                "condition": condition,
                "model": model_cfg["name"],
                "is_proxy_model": False,
            }
            results.append(result)
            checkpoint[instance_id] = result
            save_checkpoint(checkpoint_path, checkpoint)

            # Rate limiting
            time.sleep(60.0 / config["api_requests_per_minute"])

    return results


def call_api_with_retry(provider, model_id, prompt,
                        max_tokens, temperature,
                        max_retries, base_delay):
    """
    Call API with exponential backoff retry.
    Handles OpenAI, Anthropic, Google providers.
    Raises after max_retries exhausted.
    """
    import time

    for attempt in range(max_retries):
        try:
            if provider == "openai":
                return call_openai(model_id, prompt,
                                   max_tokens, temperature)
            elif provider == "anthropic":
                return call_anthropic(model_id, prompt,
                                      max_tokens, temperature)
            elif provider == "google":
                return call_google(model_id, prompt,
                                   max_tokens, temperature)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"API error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
```

#### Gemini refusal handling

```python
# Gemini 3.1 Pro may refuse to process GHOST-encoded strings.
# Record refusals explicitly — do not score as 0.
# Refusals are reported separately in results.

GEMINI_REFUSAL_PHRASES = [
    "I can't help",
    "I cannot process",
    "I'm not able to",
    "unable to assist",
]

def is_refusal(response: str) -> bool:
    response_lower = response.lower()
    return any(phrase.lower() in response_lower
               for phrase in GEMINI_REFUSAL_PHRASES)

# In result dict, add:
# "refusal": True/False
# Refusals excluded from FEA computation but
# reported as refusal_rate in Table 3 footnote.
```

### SLURM: slurm/03_eval_local.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_eval_local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/eval_local_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate

# Evaluate all local models sequentially
# Each model is loaded, evaluated, unloaded before next loads

python src/evaluate.py \
    --config config.yaml \
    --mode local \
    --models llama32_3b llama31_8b qwen25_3b \
             qwen25_7b deepseek_r1_14b \
    --conditions all

echo "Local evaluation complete"
```

### SLURM: slurm/04_eval_api.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_eval_api
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/eval_api_%j.log

module load python/3.11
source venv/bin/activate

# API keys must be set as environment variables
# Set these before submitting:
# export OPENAI_API_KEY=...
# export ANTHROPIC_API_KEY=...
# export GOOGLE_API_KEY=...

python src/evaluate.py \
    --config config.yaml \
    --mode api \
    --models gpt55 gpt56_sol claude_sonnet \
             claude_opus gemini_31_pro \
    --conditions all

echo "API evaluation complete"
```

---

## Phase 4 — In-Context Defense

### Script: src/inctx_defense.py

```python
"""
src/inctx_defense.py

Tests whether frontier models can denoise GHOST-encoded
text when explicitly informed of the encoding technique
and given worked examples.

INPUTS:  data/encoded/ghost/documents.json (first 20 financial)
         data/raw/documents.json (clean versions for examples)
         config.yaml

OUTPUTS: results/raw/{model}/inctx_defense/results.json

Models: gpt56_sol, claude_opus (strongest frontier models)
Shot counts: 5, 10, 20

SYSTEM PROMPT (inform model of encoding):
  "The following text has been obfuscated using Unicode 
   Variation Selectors injected into characters and Unicode 
   bidirectional controls that reverse the display order of 
   numbers. Characters may appear visually normal but contain 
   invisible Unicode characters, and number sequences may be 
   stored in reversed order. Please extract the exact value of 
   {FIELD_NAME} from the document, correcting for any Unicode 
   encoding you detect."

IN-CONTEXT EXAMPLES FORMAT:
  For each shot, provide:
    - GHOST-encoded document
    - Correct field value as ground truth
  
  Examples are drawn from the first N financial documents
  that are NOT part of the 20 evaluation instances.

EVALUATION:
  Same exact match metric as Phase 3.
  Report FEA per model per shot count.
  Compare against Phase 3 ghost condition FEA for same docs.

TABLE 5 FORMAT:
  Rows: After 5 shots / After 10 shots / After 20 shots
  Cols: GPT-5.6 Sol / Claude Opus 4.6
"""
```

### SLURM: slurm/05_inctx_defense.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_inctx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/inctx_%j.log

module load python/3.11
source venv/bin/activate
python src/inctx_defense.py --config config.yaml
echo "In-context defense evaluation complete"
```

---

## Phase 5 — Normalization Attack

### Script: src/norm_attack.py

```python
"""
src/norm_attack.py

Evaluates GHOST under three adversarial stripping conditions.
Tests the complementarity argument from Section 4.3.

INPUTS:  data/encoded/ghost/documents.json
         config.yaml

OUTPUTS: results/raw/{model}/norm_attack/results.json

Three attack conditions (applied programmatically,
no re-encoding needed — derive from ghost encoded docs):

1. strip_vs_only:
   Apply unicode_utils.strip_vs() to ghost-encoded text.
   Model receives: bidi controls still present, VS removed.
   Expected: model reads reversed digit sequence.
   FEA should be low (wrong digit order).

2. strip_bidi_only:
   Apply unicode_utils.strip_bidi() to ghost-encoded text.
   Model receives: VS fragmentation remains, no bidi reversal.
   Expected: model receives VS-fragmented original order digits.
   FEA should be low (VS fragmentation still active).

3. strip_both (NFKC attack):
   Apply unicode_utils.apply_nfkc() to ghost-encoded text.
   NOTE: NFKC strips VS but preserves bidi controls.
   So this is NOT strip_both — it's strip_vs + keep_bidi.
   For true strip_both: apply strip_all() not NFKC.
   Report BOTH:
     ghost_nfkc: apply_nfkc (VS stripped, bidi kept)
     ghost_strip_all: strip_all (both stripped)
   
   ghost_strip_all expected: model sees reversed digit sequence
   without VS fragmentation. Wrong value but clear text.
   Model must additionally know to reverse to recover original.

Run all three attack conditions through all 10 models.
This is cheap — no new encoding needed, just strip and evaluate.

TABLE 6 FORMAT:
  Rows: GHOST baseline / Strip VS / Strip Bidi / NFKC / Strip Both
  Cols: one per model
  Values: FEA (lower = better defense)
"""
```

### SLURM: slurm/06_norm_attack.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_norm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/norm_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate
python src/norm_attack.py --config config.yaml
echo "Normalization attack evaluation complete"
```

---

## Phase 6 — Prior Strength Gradient

### Script: src/gradient.py

```python
"""
src/gradient.py

Tests Hypothesis 1 (Prior Strength Threshold) across
five content types of increasing semantic prior strength.

INPUTS:  data/encoded/ghost/gradient.json
         data/encoded/clean/gradient.json
         config.yaml

OUTPUTS: results/raw/{model}/gradient/results.json

METRIC: Reconstruction similarity (not exact match).
  Exact match is too strict for natural language.
  Use cosine similarity via all-MiniLM-L6-v2.

PROMPT:
  "What does the following text say? Please repeat it back 
   exactly as you read it.\n\n{TEXT}"

EVALUATION:
  For each content type, for each instance:
    - Query model with GHOST-encoded version
    - Compute cosine similarity between model output
      and original clean target string
    - Average per content type

Models: all 10 (run local sequentially, API in parallel)

FIGURE 1 FORMAT:
  X-axis: content type (ordered by prior strength)
    random_digits | formatted_identifiers | alphanumeric_codes 
    | rare_proper_nouns | natural_language
  Y-axis: reconstruction similarity (0-1, lower = better defense)
  One line per model, 10 lines total
  This figure is the visual anchor of the prior strength hypothesis.

sentence_transformers setup:
  from sentence_transformers import SentenceTransformer
  sim_model = SentenceTransformer('all-MiniLM-L6-v2')

  def compute_similarity(response: str, target: str) -> float:
      embeddings = sim_model.encode([response, target])
      cos_sim = float(
          np.dot(embeddings[0], embeddings[1]) /
          (np.linalg.norm(embeddings[0]) * 
           np.linalg.norm(embeddings[1]))
      )
      return cos_sim
"""
```

### SLURM: slurm/07_gradient.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_gradient
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/gradient_%j.log

module load python/3.11 cuda/12.1
source venv/bin/activate
python src/gradient.py --config config.yaml
echo "Gradient evaluation complete"
```

---

## Phase 7 — Results Aggregation

### Script: src/metrics.py

```python
"""
src/metrics.py

Aggregates all raw results into paper-ready tables.

INPUTS:  results/raw/**/*.json
OUTPUTS: results/tables/table3_main.csv
         results/tables/table4_ablation.csv
         results/tables/table5_inctx.csv
         results/tables/table6_norm.csv
         results/tables/figure1_gradient.csv
         results/tables/summary_stats.txt

TABLE 3 — Main FEA Results:
  Rows: models (10 total + average row)
  Cols: clean | bae | textfooler | homochar | 
        vs_only | bidi_only | ghost
  Values: FEA (mean across all fields and documents)
  Separate columns for proxy-optimised (qwen25_7b)
  and transferred (all others)

TABLE 4 — Ablation:
  Rows: models (10 total)
  Cols: VS Only | Bidi Only | GHOST (combined)
  Values: FEA
  Include per-domain breakdown as supplementary:
    Financial / Medical / Legal / Technical

TABLE 5 — In-Context Defense:
  Rows: After 5 shots | After 10 shots | After 20 shots
  Cols: GPT-5.6 Sol | Claude Opus 4.6
  Values: FEA (on the 20 financial evaluation docs)
  Also report: baseline ghost FEA on same docs (row 0)

TABLE 6 — Normalization Attack:
  Rows: GHOST | Strip VS only | Strip Bidi only | 
        NFKC (strip VS) | Strip Both
  Cols: models (10 total)
  Values: FEA

FIGURE 1 — Prior Strength Gradient:
  CSV with columns:
    content_type | model | reconstruction_similarity
  Plotting done separately (include matplotlib script)

SUMMARY STATS (summary_stats.txt):
  - Mean FEA reduction: clean -> ghost across all models
  - Proxy vs transfer gap: qwen25_7b vs mean of others
  - Refusal rates by model and condition
  - Mean injection budget per digit type
  - Per-domain breakdown
  Print these in the terminal and save to file.
"""
```

### SLURM: slurm/08_metrics.sh

```bash
#!/bin/bash
#SBATCH --job-name=ghost_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=logs/metrics_%j.log

module load python/3.11
source venv/bin/activate
python src/metrics.py --config config.yaml
echo "Metrics aggregation complete"
cat results/tables/summary_stats.txt
```

---

## Master Submission Script

```bash
#!/bin/bash
# submit_all.sh
# Run from ghost/ directory after completing Phase 0 setup.
# Set API keys as environment variables before running.

set -e  # exit on any error

echo "Submitting GHOST experiment pipeline..."

# Check API keys are set
: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"
: "${ANTHROPIC_API_KEY:?Need ANTHROPIC_API_KEY}"
: "${GOOGLE_API_KEY:?Need GOOGLE_API_KEY}"

# Phase 1: Dataset
JOB1=$(sbatch --parsable slurm/01_dataset.sh)
echo "Phase 1 (dataset): job $JOB1"

# Phase 2: Encoding (depends on Phase 1)
JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    slurm/02_encode.sh)
echo "Phase 2 (encode): job $JOB2"

# Phase 3a: Local evaluation (depends on Phase 2)
JOB3A=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    slurm/03_eval_local.sh)
echo "Phase 3a (eval local): job $JOB3A"

# Phase 3b: API evaluation (depends on Phase 2)
JOB3B=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    --export=ALL,OPENAI_API_KEY,ANTHROPIC_API_KEY,GOOGLE_API_KEY \
    slurm/04_eval_api.sh)
echo "Phase 3b (eval API): job $JOB3B"

# Phase 4: In-context defense (depends on Phase 3b)
JOB4=$(sbatch --parsable \
    --dependency=afterok:$JOB3B \
    --export=ALL,OPENAI_API_KEY,ANTHROPIC_API_KEY \
    slurm/05_inctx_defense.sh)
echo "Phase 4 (in-context defense): job $JOB4"

# Phase 5: Normalization attack (depends on Phase 2)
# Run local + API in parallel
JOB5=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    slurm/06_norm_attack.sh)
echo "Phase 5 (normalization attack): job $JOB5"

# Phase 6: Prior strength gradient (depends on Phase 2)
JOB6=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    slurm/07_gradient.sh)
echo "Phase 6 (gradient): job $JOB6"

# Phase 7: Metrics (depends on all evaluation phases)
sbatch \
    --dependency=afterok:$JOB3A:$JOB3B:$JOB4:$JOB5:$JOB6 \
    slurm/08_metrics.sh
echo "Phase 7 (metrics): submitted with all dependencies"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
echo "Logs in: logs/"
```

---

## Smoke Tests

Run these before submitting to Spartan.
Each should complete in under 2 minutes on a CPU.

```bash
# smoke_test.sh — run locally before HPC submission

source venv/bin/activate

# Test 1: unicode_utils correctness
python - <<'EOF'
from src.unicode_utils import (
    encode_bidi, strip_bidi, strip_vs,
    strip_all, apply_nfkc, show_codepoints,
    encode_payload_as_vs, byte_to_vs
)

# Bidi encode/strip roundtrip
field = "48271039"
encoded = encode_bidi(field)
assert strip_bidi(encoded) == field[::-1], "Bidi strip failed"
print(f"Bidi encode: {show_codepoints(encoded)}")

# VS encode
vs_seq = encode_payload_as_vs("test")
assert len(vs_seq) == 4, "VS encode wrong length"
print(f"VS sequence: {show_codepoints(vs_seq)}")

# NFKC strips VS but keeps bidi
from unicode_utils import RTL, PDF
test_with_vs = RTL + "4" + vs_seq + "8" + PDF
nfkc_result = apply_nfkc(test_with_vs)
assert RTL in nfkc_result, "NFKC removed bidi"
# VS may or may not be stripped depending on codepoints
print("NFKC test passed")

print("TEST 1 PASSED: unicode_utils")
EOF

# Test 2: dataset generation (5 docs only)
python - <<'EOF'
import sys
sys.argv = ["dataset.py", "--config", "config.yaml", "--smoke"]
# Modify dataset.py to support --smoke flag that generates
# 5 documents per domain instead of 100
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dataset", "src/dataset.py"
)
EOF
echo "Run: python src/dataset.py --config config.yaml --smoke"

# Test 3: encoding (3 docs only)
echo "Run: python src/encode.py --config config.yaml \
  --smoke --conditions clean bidi_only ghost"

# Test 4: evaluation (3 docs, local model only)
echo "Run: python src/evaluate.py --config config.yaml \
  --smoke --mode local --models llama32_3b \
  --conditions clean ghost"

# Each script MUST support --smoke flag that:
# - Processes only 3-5 documents
# - Skips API calls (uses mock responses)
# - Completes in < 2 minutes on CPU
# - Verifies output file structure is correct
```

---

## Implementation Notes for Claude Code

### Implement in this exact order:

1. `requirements.txt` — install and verify all packages
2. `config.yaml` — single source of truth, no hardcoded values
3. `src/unicode_utils.py` — run smoke test before proceeding
4. `src/dataset.py` — run smoke test before proceeding
5. `src/encode.py` — implement checkpointing first, then encoding logic
6. `src/evaluate.py` — implement local models first, then API
7. `src/inctx_defense.py`
8. `src/norm_attack.py`
9. `src/gradient.py`
10. `src/metrics.py`
11. All SLURM scripts
12. `submit_all.sh`

### Never do any of the following:

- Hardcode model names, paths, or parameters outside config.yaml
- Load two local models simultaneously
- Skip checkpointing in encode.py or evaluate.py
- Use semantic similarity as the primary metric for main results
  (exact match only for field extraction)
- Score Gemini refusals as exact_match=0 without flagging separately
- Proceed to the next phase before smoke testing the current one
- Use temperature > 0 for any evaluation (deterministic outputs only)

### Always do the following:

- Read all parameters from config.yaml
- Write outputs to the exact paths specified in the directory structure
- Log progress to logs/ with timestamps
- Handle DeepSeek <think> tag stripping in every evaluation path
- Normalise formatting characters (dashes, spaces) before exact match
- Save partial results after every document (checkpointing)
- Print a brief summary to stdout after each phase completes

### Key design decisions to preserve:

- Qwen2.5-7B is BOTH proxy and target — report its results
  separately from the transferability analysis
- Bidi encoding reverses the stored sequence; VS injection applies
  to the REVERSED characters, not the original
- NFKC normalization strips VS but PRESERVES bidi controls
  — this is not the same as strip_all()
- Field ground truth never changes regardless of encoding condition
  (except for BAE/TextFooler/HOMOCHAR which may alter field values)
- Refusals (Gemini refusing to process) are not failures to be
  corrected — they are extraction failures for the adversary
  and defense successes for the paper
