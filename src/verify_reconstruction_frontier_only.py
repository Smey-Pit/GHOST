"""
Re-run just the frontier verify tier against the already-converged
reconstruction encoding (results/raw/verify_reconstruction_lyric.json),
with a larger max_tokens override -- config.yaml's per-model max_tokens
were calibrated for short field-extraction responses (10-12 chars), not
a ~127-character verbatim reconstruction, which needs far more output
budget (plus hidden reasoning tokens for gpt56_sol/gemini_31_pro).
Not part of the committed pipeline.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import yaml
from adversary import query_adversary, check_reconstruction, RECONSTRUCTION_PROMPT

RECONSTRUCTION_MIN_TOKENS = 4000

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    config = yaml.safe_load(f)

with open(
    os.path.join(os.path.dirname(__file__), "..", "results", "raw",
                 "verify_reconstruction_lyric.json")
) as f:
    d = json.load(f)

encoding = d["encoding"]
target = d["target"]
field_name = d["field"]

verify_results = {}
for name in config["agent_ensemble"]["frontier_verify_models"]:
    model_cfg = config["api_models"][name]
    max_tokens = max(model_cfg["max_tokens"], RECONSTRUCTION_MIN_TOKENS)
    response = query_adversary(
        encoded_text=encoding,
        field_name=field_name,
        model_id=model_cfg["model_id"],
        provider=model_cfg["provider"],
        max_tokens=max_tokens,
        prompt_template=RECONSTRUCTION_PROMPT,
    )
    verdict = check_reconstruction(response, target)
    verify_results[name] = {"response": response, **verdict}
    print(f"\n[{name}] max_tokens={max_tokens}")
    print(f"[{name}] response={response!r}")
    print(f"[{name}] similarity={verdict['similarity']:.3f} "
          f"reconstructed={verdict['extracted']} refusal={verdict['refusal']}")

d["verify_results"] = verify_results
out_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "raw",
    "verify_reconstruction_lyric.json",
)
with open(out_path, "w") as f:
    json.dump(d, f, indent=2)
print(f"\nSaved to {out_path}")
