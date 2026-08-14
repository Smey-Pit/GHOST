"""
src/verify_normalization_attack.py

Scratch driver (same status as verify_gemini_run.py -- not yet folded into
the committed src/norm_attack.py stub): applies NFC/NFD/NFKC/NFKD to the
real composed-GHOST example already on hand
(results/raw/verify_gemini_account_number.json, account_number =
5926847003, search tier converged 1 iteration, Hamming 10/10), then
queries all three frontier verify models (config.yaml's
agent_ensemble.frontier_verify_models: claude_sonnet, gpt56_sol,
gemini_31_pro) against each normalized variant plus the raw
(no-normalization) baseline.

CPU-only: no local HF model is loaded, only frontier API calls
(_call_anthropic/_call_openai/_call_google in adversary.py). Needs
ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY in the environment --
source ~/.bash_profile first if running non-interactively (see CLAUDE.md
gotchas).

Run with: python src/verify_normalization_attack.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml

from unicode_utils import apply_nfc, apply_nfd, apply_nfkc, apply_nfkd
from adversary import query_adversary, check_extraction

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
EXAMPLE_PATH = os.path.join(REPO_ROOT, "results", "raw", "verify_gemini_account_number.json")
OUT_PATH = os.path.join(REPO_ROOT, "results", "raw", "norm_attack_account_number.json")

NORMALIZATION_FORMS = {
    "raw": lambda t: t,
    "NFC": apply_nfc,
    "NFD": apply_nfd,
    "NFKC": apply_nfkc,
    "NFKD": apply_nfkd,
}


def main():
    with open(os.path.join(REPO_ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)

    with open(EXAMPLE_PATH) as f:
        example = json.load(f)

    field_name = example["field"]
    target = example["target"]
    ghost_encoding = example["encoding"]

    print(f"Field: {field_name}   Target: {target}")
    print(f"Source encoding: {EXAMPLE_PATH}\n")

    verify_models = config["agent_ensemble"]["frontier_verify_models"]

    results = {}
    for form_name, form_fn in NORMALIZATION_FORMS.items():
        transformed = form_fn(ghost_encoding)
        print(f"=== {form_name} (len={len(transformed)}) ===")
        results[form_name] = {"encoding": transformed, "models": {}}

        for model_name in verify_models:
            model_cfg = config["api_models"][model_name]
            response = query_adversary(
                encoded_text=transformed,
                field_name=field_name,
                model_id=model_cfg["model_id"],
                provider=model_cfg["provider"],
                max_tokens=model_cfg["max_tokens"],
            )
            verdict = check_extraction(response, target)
            results[form_name]["models"][model_name] = {
                "response": response,
                **verdict,
            }
            status = (
                "REFUSAL" if verdict["refusal"]
                else "EXTRACTED" if verdict["extracted"]
                else "PARTIAL" if verdict["partial"]
                else "DEFENDED"
            )
            print(f"  [{model_name}] {status:10s} response={response!r}")
        print()

    with open(OUT_PATH, "w") as f:
        json.dump(
            {"field": field_name, "target": target,
             "source_encoding": ghost_encoding, "results": results},
            f, indent=2,
        )
    print(f"Saved to {OUT_PATH}")

    print("\n=== Summary (rows=normalization form, cols=model) ===")
    header = ["form"] + verify_models
    print("  ".join(f"{h:12s}" for h in header))
    for form_name in NORMALIZATION_FORMS:
        row = [form_name]
        for model_name in verify_models:
            v = results[form_name]["models"][model_name]
            row.append(
                "REFUSAL" if v["refusal"]
                else "EXTRACTED" if v["extracted"]
                else "PARTIAL" if v["partial"]
                else "DEFENDED"
            )
        print("  ".join(f"{c:12s}" for c in row))


if __name__ == "__main__":
    main()
