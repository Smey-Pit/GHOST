"""
src/ghost_demo.py

Standalone-text entry point for GHOST: take arbitrary text + a sensitive
substring, run any of the real encoding conditions on it, and get back the
encoded text plus enough detail to show a human what happened. Built for a
demo site (see CLAUDE.md's "GHOST-Agent given the static pipeline's real
tools" era of this project for the mechanisms themselves) -- everything
here is a thin wrapper around src/encode.py's existing, tested per-condition
functions, NOT a reimplementation. Those functions are dataset-item-shaped
(`{"id", "text", "fields": {name: value}}`); `_wrap_as_item` synthesizes
that shape from a raw string so the real, already-verified encoding logic
runs unchanged.

GHOST-Agent (the LLM-driven search loop) is deliberately NOT wired in here
yet -- it needs a live local ensemble (GPU, ~15-20s/model/iteration) which
isn't demo-latency-friendly. See ghost_agent.run_ghost_agent for that path
when it's ready to be added.
"""

import hashlib
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from encode import (  # noqa: E402
    encode_bidi_only, encode_bidi_permute, encode_bidi_permute_sentence,
    encode_vs_only, encode_ghost, encode_ghost_permute,
    encode_ghost_permute_sentence, load_proxy_model,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

# mechanism key -> (needs_proxy, scope, label)
MECHANISMS = {
    "bidi_only":              (False, "field",    "Bidi reversal only"),
    "bidi_permute":            (False, "field",    "Bidi separable permutation"),
    "bidi_permute_sentence":    (False, "sentence", "Bidi permutation (whole sentence)"),
    "vs_only":                  (True,  "field",    "Variation-selector injection only"),
    "ghost":                    (True,  "field",    "GHOST (bidi reversal + VS injection)"),
    "ghost_permute":            (True,  "field",    "GHOST-permute (bidi permutation + VS injection)"),
    "ghost_permute_sentence":   (True,  "sentence", "GHOST-permute, whole sentence"),
}


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def _doc_id(text, field_value):
    """Deterministic pseudo doc id -- keeps derive_payload_bytes/permutation
    search seeds reproducible for the same (text, field_value) input."""
    h = hashlib.sha256(f"{text}\x00{field_value}".encode("utf-8")).hexdigest()
    return f"demo_{h[:16]}"


def _wrap_as_item(text, field_value):
    if field_value not in text:
        raise ValueError(
            f"field_value {field_value!r} does not appear verbatim in text"
        )
    return {"id": _doc_id(text, field_value), "text": text, "fields": {"target": field_value}}


def get_proxy(config, cache={}):
    """Loads the Qwen proxy once and keeps it resident (module-level cache)
    -- a demo server should call this at startup or on first VS-mechanism
    request, never per-request (see CLAUDE.md's load-time-dominates-cost
    lesson for exactly this kind of model)."""
    if "proxy" not in cache:
        cache["proxy"] = load_proxy_model(config)
    return cache["proxy"]


def encode_text(text, field_value, mechanism, config=None, proxy=None):
    """
    Run one GHOST encoding condition on arbitrary text.

    Args:
        text: full document/passage text
        field_value: the sensitive substring to obfuscate -- must appear
            verbatim in `text`. For sentence-scope mechanisms this is used
            only to locate the carrier sentence; the sentence itself (not
            just field_value) gets obfuscated.
        mechanism: one of MECHANISMS' keys
        config: parsed config.yaml (defaults to loading it fresh)
        proxy: an already-loaded ProxyModel (required for vs_only/ghost/
            ghost_permute/ghost_permute_sentence) -- get one via get_proxy()

    Returns dict: {mechanism, needs_proxy, scope, original_text,
                   encoded_text, field_value, vs_injection_iterations}
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism {mechanism!r}; choices: {list(MECHANISMS)}")
    needs_proxy, scope, _label = MECHANISMS[mechanism]
    if needs_proxy and proxy is None:
        raise ValueError(f"mechanism {mechanism!r} requires a loaded proxy (see get_proxy())")

    config = config or load_config()
    seed = config["seed"]
    salt = config["disruption_payload"]
    threshold_tau = config["threshold_tau"]
    trials = config.get("bidi_permute_search_trials", 500)

    item = _wrap_as_item(text, field_value)
    stats = []

    if mechanism == "bidi_only":
        out = encode_bidi_only(item)
    elif mechanism == "bidi_permute":
        out = encode_bidi_permute(item, trials, seed, salt)
    elif mechanism == "bidi_permute_sentence":
        out = encode_bidi_permute_sentence(item, trials, seed, salt)
    elif mechanism == "vs_only":
        out = encode_vs_only(item, proxy, seed, salt, threshold_tau, stats)
    elif mechanism == "ghost":
        out = encode_ghost(item, proxy, seed, salt, threshold_tau, stats)
    elif mechanism == "ghost_permute":
        out = encode_ghost_permute(item, proxy, seed, salt, threshold_tau, stats, trials)
    elif mechanism == "ghost_permute_sentence":
        out = encode_ghost_permute_sentence(item, proxy, seed, salt, threshold_tau, stats, trials)

    return {
        "mechanism": mechanism,
        "label": MECHANISMS[mechanism][2],
        "needs_proxy": needs_proxy,
        "scope": scope,
        "original_text": text,
        "encoded_text": out["text"],
        "field_value": field_value,
        "vs_injection_iterations": stats or None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--field_value", required=True)
    parser.add_argument("--mechanism", required=True, choices=list(MECHANISMS))
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    px = get_proxy(cfg) if MECHANISMS[args.mechanism][0] else None
    result = encode_text(args.text, args.field_value, args.mechanism, cfg, px)
    print("ORIGINAL:", result["original_text"])
    print("ENCODED :", result["encoded_text"])
    if result["vs_injection_iterations"]:
        print("VS iterations per char:", result["vs_injection_iterations"])
