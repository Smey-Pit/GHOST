"""
src/encode.py

Phase 2: document encoding across all conditions
(clean, bidi_only, bidi_permute, vs_only, ghost, ghost_permute, ghost_nfkc,
ghost_permute_nfkc, bae, textfooler, homochar).

bidi_permute / ghost_permute are the separable-permutation variant of
Mechanism 2 (see src/bidi_permute.py): instead of a flat RTL reversal,
each field is stored as a maximally-displaced separable permutation of
its own characters, reconstructed to the original by nested bidi
isolates/overrides. ghost_permute layers logprob-guided VS injection
on top, mirroring how ghost layers VS injection onto bidi_only's
reversal.

INPUTS:  data/raw/documents.json
         data/raw/gradient.json
         data/raw/calibration.json   (--calibrate mode only; never encoded
                                      into data/encoded/, used only to
                                      pick threshold_tau before a real run)
         config.yaml
         Qwen2.5-7B-Instruct loaded locally for logprob queries (vs_only, ghost)

OUTPUTS: data/encoded/{condition}/documents.json
         data/encoded/{condition}/gradient.json   (only for conditions that
                                                    Phase 6 actually reads:
                                                    clean, ghost)

CHECKPOINTING: after encoding each item, write checkpoint. On restart, skip
already-encoded items. Checkpoint files:
  data/encoded/{condition}/.checkpoint.{dataset}.json
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from unicode_utils import (  # noqa: E402
    RTL, PDF, encode_bidi, apply_nfkc, derive_payload_bytes,
)
from bidi_permute import (  # noqa: E402
    decompose, encode as bidi_permute_encode, find_target_permutation,
)

# Conditions that require the proxy model for logprob-guided VS injection.
VS_DEPENDENT_CONDITIONS = {"vs_only", "ghost", "ghost_permute"}
# Conditions derived from an already-encoded ghost document (no proxy needed).
# Maps derived condition -> the source condition its NFKC pass reads from.
DERIVED_CONDITIONS = {"ghost_nfkc": "ghost", "ghost_permute_nfkc": "ghost_permute"}
# Conditions using TextAttack baselines on full document text.
TEXTATTACK_CONDITIONS = {"bae", "textfooler", "homochar"}
# gradient.json is only ever consumed by gradient.py for these conditions.
GRADIENT_CONDITIONS = {"clean", "ghost", "ghost_permute"}


# ── Checkpointing ────────────────────────────────────────────────

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path, checkpoint):
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f)
    os.replace(tmp_path, checkpoint_path)


# ── Proxy model (logprob-guided VS injection) ────────────────────

class ProxyModel:
    """Wraps the proxy model used for the VS-injection stopping criterion."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def query_logprob(self, target_char, context):
        """
        Query the log-probability of target_char appearing in context
        using a teacher-forced forward pass over the last position.
        """
        import torch

        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        target_ids = self.tokenizer(
            target_char, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        if target_ids.shape[1] == 0:
            return float("-inf")  # untokenizable = already defeated
        target_id = target_ids[0, 0].item()

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[target_id].item()


def load_proxy_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = config["proxy_model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    return ProxyModel(model, tokenizer, device)


def encode_char_with_vs_logprob(base_char, payload_bytes, proxy, threshold_tau, max_iters=64):
    """
    Iteratively inject VS characters into base_char until the proxy's
    logprob of the original character falls below threshold_tau, or
    max_iters is reached (safety bound not present in the original plan —
    without it a character the proxy never gets uncertain about would
    loop until the payload's bytes are exhausted and then silently stop
    making progress while still re-querying every remaining byte).

    payload_bytes must already be unique to this character occurrence
    (see unicode_utils.derive_payload_bytes) -- callers must NOT reuse
    the same payload_bytes across characters, or the resulting VS
    sequences share a fixed, corpus-wide, string-matchable signature.
    """
    current = base_char
    iterations = 0
    for byte_idx in range(min(len(payload_bytes), max_iters)):
        vs_char = _byte_to_vs(payload_bytes[byte_idx])
        candidate = current + vs_char
        logprob = proxy.query_logprob(base_char, candidate)
        current = candidate
        iterations += 1
        if logprob <= threshold_tau:
            break
    return current, iterations


def _byte_to_vs(byte):
    from unicode_utils import byte_to_vs
    return byte_to_vs(byte)


def _derive_int_seed(seed, salt, *context):
    """
    Per-occurrence integer seed for find_target_permutation, derived the
    same keyed way as the VS payload. Without this, the permutation
    search picks its candidate purely from (search_seed, field length) --
    it never looks at the field's actual characters -- so a single fixed
    seed would give every field of the same length the IDENTICAL stored
    permutation across the whole corpus: not just detectable but directly
    invertible from a handful of examples.
    """
    b = derive_payload_bytes(seed, salt, *context, n_bytes=8)
    return int.from_bytes(b, "big")


# ── Per-condition encoding (documents / gradient share the same shape:
#    a dict of {field_name: field_value} to transform within `text`) ──

def _apply_field_transform(item, transform_fn):
    """
    Apply transform_fn(field_value, field_name) -> encoded_value to every
    field in item["fields"] (documents) or item["target"] (gradient
    instances; field_name passed as "target"), substituting the encoded
    value into item["text"].
    Returns a new dict; ground truth (fields/target) is left unchanged
    unless transform_fn also mutates it via the flag mechanism below.
    """
    text = item["text"]
    if "fields" in item:
        new_fields = {}
        for field_name, field_value in item["fields"].items():
            encoded_value = transform_fn(field_value, field_name)
            text = text.replace(field_value, encoded_value, 1)
            new_fields[field_name] = field_value
        return {**item, "text": text, "fields": new_fields}
    else:
        encoded_value = transform_fn(item["target"], "target")
        text = text.replace(item["target"], encoded_value, 1)
        return {**item, "text": text}


def encode_clean(item):
    return dict(item)


def encode_bidi_only(item):
    return _apply_field_transform(item, lambda field_value, field_name: encode_bidi(field_value))


def encode_vs_only(item, proxy, seed, salt, threshold_tau, stats):
    def transform(field_value, field_name):
        encoded_chars = []
        for char_index, char in enumerate(field_value):
            payload_bytes = derive_payload_bytes(
                seed, salt, "vs_only", item["id"], field_name, char_index
            )
            encoded_char, iters = encode_char_with_vs_logprob(char, payload_bytes, proxy, threshold_tau)
            encoded_chars.append(encoded_char)
            stats.append(iters)
        return "".join(encoded_chars)
    return _apply_field_transform(item, transform)


def encode_ghost(item, proxy, seed, salt, threshold_tau, stats):
    def transform(field_value, field_name):
        reversed_value = field_value[::-1]
        encoded_chars = []
        for char_index, char in enumerate(reversed_value):
            payload_bytes = derive_payload_bytes(
                seed, salt, "ghost", item["id"], field_name, char_index
            )
            encoded_char, iters = encode_char_with_vs_logprob(char, payload_bytes, proxy, threshold_tau)
            encoded_chars.append(encoded_char)
            stats.append(iters)
        return RTL + "".join(encoded_chars) + PDF
    return _apply_field_transform(item, transform)


def encode_bidi_permute(item, trials, seed, salt):
    """
    Mechanism 2 variant: instead of a flat RTL reversal, search for a
    separable permutation of the field value maximally displaced (by
    Hamming distance) from the original, then encode it via nested
    bidi isolates/overrides. See src/bidi_permute.py.

    The search seed is derived per (item, field), not reused globally --
    find_target_permutation picks its candidate purely by structure
    (Hamming distance from identity), never looking at the field's
    actual characters, so a single shared seed would give every field
    of the same length the IDENTICAL stored permutation across the whole
    corpus: directly invertible from a handful of examples, not just
    detectable.
    """
    def transform(field_value, field_name):
        if not field_value:
            return field_value
        field_seed = _derive_int_seed(seed, salt, "bidi_permute", item["id"], field_name)
        _, perm = find_target_permutation(field_value, trials=trials, seed=field_seed)
        tree = decompose(tuple(perm))
        return bidi_permute_encode(tree, field_value)
    return _apply_field_transform(item, transform)


def encode_ghost_permute(item, proxy, seed, salt, threshold_tau, stats, trials):
    """
    encode_bidi_permute + logprob-guided VS injection, mirroring how
    encode_ghost layers VS injection onto encode_bidi's flat reversal:
    VS is injected per-character in STORED (permuted) order, since that
    is the order a tokenizer actually ingests.
    """
    def transform(field_value, field_name):
        if not field_value:
            return field_value
        field_seed = _derive_int_seed(seed, salt, "ghost_permute", item["id"], field_name)
        _, perm = find_target_permutation(field_value, trials=trials, seed=field_seed)
        tree = decompose(tuple(perm))
        chars_by_original_pos = [None] * len(field_value)
        for orig_pos in perm:
            payload_bytes = derive_payload_bytes(
                seed, salt, "ghost_permute", item["id"], field_name, orig_pos
            )
            vs_char, iters = encode_char_with_vs_logprob(
                field_value[orig_pos], payload_bytes, proxy, threshold_tau
            )
            chars_by_original_pos[orig_pos] = vs_char
            stats.append(iters)
        return bidi_permute_encode(tree, chars_by_original_pos)
    return _apply_field_transform(item, transform)


def encode_ghost_nfkc(ghost_item):
    return {**ghost_item, "text": apply_nfkc(ghost_item["text"])}


def _project_span_through_diff(original, attacked, start, end):
    """
    Given a character span [start, end) in `original`, return the
    corresponding substring in `attacked` using a sequence alignment,
    so a field's post-attack value can be recovered even when the
    attack (BAE/TextFooler) replaced a word rather than leaving a
    clean 1:1 character mapping like homochar's.
    """
    import difflib

    matcher = difflib.SequenceMatcher(None, original, attacked)
    projected = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # Does this opcode's original-side range overlap [start, end)?
        overlap_start, overlap_end = max(i1, start), min(i2, end)
        if overlap_start >= overlap_end:
            continue
        if tag == "equal":
            offset = overlap_start - i1
            length = overlap_end - overlap_start
            projected.append(attacked[j1 + offset: j1 + offset + length])
        else:
            # replace/delete/insert: attribute the whole attacked-side
            # span for this opcode (best-effort; word-level attacks
            # don't have a clean char-for-char mapping).
            projected.append(attacked[j1:j2])
    return "".join(projected).strip()


def encode_textattack(item, attack_name, attack_wrapper):
    """
    Apply a TextAttack baseline to the full document text.
    If the attack alters characters within a field's own value, that
    field's ground truth is updated to the modified value and the
    instance is flagged (per plan: BAE/TextFooler/HOMOCHAR may alter
    the actual field value characters).
    """
    original_text = item["text"]
    attacked_text = attack_wrapper(original_text)

    result = {**item, "text": attacked_text}
    flagged = False
    if "fields" in item:
        new_fields = {}
        for field_name, field_value in item["fields"].items():
            raw_value = field_value.replace("-", "").replace(" ", "")
            raw_attacked = attacked_text.replace("-", "").replace(" ", "")
            if raw_value in raw_attacked:
                new_fields[field_name] = field_value
                continue

            flagged = True
            if attack_name == "homochar":
                # 1:1 character substitution — exact new value is known
                # directly, no alignment needed.
                new_fields[field_name] = _homochar_attack(field_value)
            else:
                start = original_text.find(field_value)
                if start == -1:
                    # Shouldn't happen (field values are always inserted
                    # verbatim by dataset.py), but don't fabricate a GT.
                    new_fields[field_name] = field_value
                else:
                    end = start + len(field_value)
                    new_fields[field_name] = _project_span_through_diff(
                        original_text, attacked_text, start, end
                    ) or field_value
        result["fields"] = new_fields
    result["flagged"] = flagged
    return result


# ── TextAttack wrappers ───────────────────────────────────────────

def _make_textattack_wrapper(attack_name):
    """
    Lazily build a callable(text) -> attacked_text for one of
    bae / textfooler / homochar. Constructed lazily (only when that
    condition is actually requested) since building an attack recipe
    loads its own models (e.g. BAE loads a BERT MLM).
    """
    if attack_name == "homochar":
        return _homochar_attack

    # Both BAEGarg2019 and TextFoolerJin2019 include a Universal Sentence
    # Encoder constraint (via tensorflow_hub). This process may also load
    # Qwen2.5-7B on GPU via PyTorch for vs_only/ghost in the same run, so
    # we force TF specifically onto CPU rather than hiding the GPU via
    # CUDA_VISIBLE_DEVICES (which would break the PyTorch proxy model).
    # Without this, USE's graph tries to JIT-compile on GPU and fails
    # with "libdevice not found" on this cluster's CUDA/XLA setup.
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    from textattack.attack_recipes import BAEGarg2019, TextFoolerJin2019
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # TextAttack recipes are built against a victim classifier; since we
    # only want the recipe's text transformation (not its attack-success
    # goal against a specific model), we wrap a small off-the-shelf
    # sentiment model purely to satisfy the recipe's constructor.
    victim_name = "distilbert-base-uncased-finetuned-sst-2-english"
    victim_model = AutoModelForSequenceClassification.from_pretrained(victim_name)
    victim_tokenizer = AutoTokenizer.from_pretrained(victim_name)
    wrapper = HuggingFaceModelWrapper(victim_model, victim_tokenizer)

    if attack_name == "bae":
        recipe = BAEGarg2019.build(wrapper)
    elif attack_name == "textfooler":
        recipe = TextFoolerJin2019.build(wrapper)
    else:
        raise ValueError(f"Unknown textattack condition: {attack_name}")

    def run(text):
        from textattack.shared import AttackedText
        result = recipe.attack(text, 0)
        try:
            return result.perturbed_text()
        except Exception:
            return text  # attack failed to find a perturbation; passthrough

    return run


_HOMOCHAR_MAP = {
    "0": "О",  # Cyrillic О
    "1": "І",  # Cyrillic І
    "3": "З",  # Cyrillic З
    "a": "а", "e": "е", "o": "о", "p": "р",
    "c": "с", "y": "у", "x": "х",
}


def _homochar_attack(text):
    """Character-level homoglyph substitution baseline."""
    return "".join(_HOMOCHAR_MAP.get(ch, ch) for ch in text)


# ── Calibration (--calibrate mode) ────────────────────────────────

def _char_logprob_trace(base_char, payload_bytes, proxy, max_iters=64):
    """
    Compute the FULL logprob trajectory for one character as VS bytes
    are appended one at a time, up to max_iters. Doing this once lets
    --calibrate sweep many candidate threshold_tau values without
    re-running the (expensive) forward passes per candidate.
    """
    current = base_char
    trace = []
    for byte_idx in range(min(len(payload_bytes), max_iters)):
        vs_char = _byte_to_vs(payload_bytes[byte_idx])
        current = current + vs_char
        trace.append(proxy.query_logprob(base_char, current))
    return trace


def _iters_to_threshold(trace, tau):
    """First 1-indexed position where the trace drops to <= tau, else len(trace)."""
    for i, logprob in enumerate(trace):
        if logprob <= tau:
            return i + 1
    return len(trace)


def run_calibration(config):
    """
    Run the logprob-guided VS injection pilot on the DISJOINT calibration
    split (data/raw/calibration.json). Computes the full per-character
    logprob trajectory once, then reports iteration-count stats for a
    sweep of candidate threshold_tau values so a value can be picked
    before running the main encode. Never writes into data/encoded/ and
    never touches documents.json.
    """
    with open(os.path.join(config["data_raw_dir"], "calibration.json")) as f:
        calibration = json.load(f)

    proxy = load_proxy_model(config)
    seed = config["seed"]
    salt = config["disruption_payload"]
    configured_tau = config["threshold_tau"]

    traces = []
    for doc in calibration:
        for field_name, field_value in doc["fields"].items():
            for char_index, char in enumerate(field_value):
                payload_bytes = derive_payload_bytes(
                    seed, salt, "calibration", doc["id"], field_name, char_index
                )
                traces.append(_char_logprob_trace(char, payload_bytes, proxy))

    print(f"Calibration pilot: {len(traces)} characters across {len(calibration)} docs "
          f"(per-character derived payload, not a single fixed string -- "
          f"see unicode_utils.derive_payload_bytes)")

    candidate_taus = sorted(set([configured_tau, -1.0, -2.0, -3.0, -5.0, -8.0, -10.0, -15.0, -20.0]))
    for tau in candidate_taus:
        stats = sorted(_iters_to_threshold(trace, tau) for trace in traces)
        n = len(stats)
        never_reached = sum(1 for trace in traces if all(lp > tau for lp in trace))
        marker = "  <- currently configured" if tau == configured_tau else ""
        print(f"tau={tau:>6.1f}  min={stats[0]:>2} p25={stats[n//4]:>2} "
              f"median={stats[n//2]:>2} p75={stats[3*n//4]:>2} max={stats[-1]:>2}  "
              f"never_reached={never_reached}/{n}{marker}")

    print()
    print("Guidance: pick the tau whose median lands around 3-5 iterations "
          "(the plan's target for 'reasonable'). If every tau in this sweep "
          "shows never_reached > 0, the payload or model may need revisiting "
          "rather than tau alone.")


# ── Main encode loop ──────────────────────────────────────────────

def encode_condition(condition, items, config, checkpoint_path, proxy=None, aux=None):
    checkpoint = load_checkpoint(checkpoint_path)
    results = []
    stats = []
    seed = config["seed"]
    salt = config["disruption_payload"]
    tau = config["threshold_tau"]
    bidi_permute_trials = config.get("bidi_permute_search_trials", 500)

    for item in items:
        if item["id"] in checkpoint:
            results.append(checkpoint[item["id"]])
            continue

        if condition == "clean":
            encoded = encode_clean(item)
        elif condition == "bidi_only":
            encoded = encode_bidi_only(item)
        elif condition == "bidi_permute":
            encoded = encode_bidi_permute(item, bidi_permute_trials, seed, salt)
        elif condition == "vs_only":
            encoded = encode_vs_only(item, proxy, seed, salt, tau, stats)
        elif condition == "ghost":
            encoded = encode_ghost(item, proxy, seed, salt, tau, stats)
        elif condition == "ghost_permute":
            encoded = encode_ghost_permute(item, proxy, seed, salt, tau, stats,
                                            bidi_permute_trials)
        elif condition in DERIVED_CONDITIONS:
            source_item = aux[item["id"]]
            encoded = encode_ghost_nfkc(source_item)
        elif condition in TEXTATTACK_CONDITIONS:
            encoded = encode_textattack(item, condition, aux)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        results.append(encoded)
        checkpoint[item["id"]] = encoded
        save_checkpoint(checkpoint_path, checkpoint)

    if stats:
        print(f"  [{condition}] mean VS-injection iterations/char: {sum(stats) / len(stats):.2f}")

    return results


def run_encode(config, conditions, datasets, item_limit=None, output_dir=None):
    with open(os.path.join(config["data_raw_dir"], "documents.json")) as f:
        documents = json.load(f)
    with open(os.path.join(config["data_raw_dir"], "gradient.json")) as f:
        gradient = json.load(f)

    if item_limit is not None:
        documents = documents[:item_limit]
        gradient = gradient[:item_limit]

    proxy = None
    if any(c in VS_DEPENDENT_CONDITIONS for c in conditions):
        proxy = load_proxy_model(config)

    base_dir = output_dir or config["data_encoded_dir"]

    def _load_derived_source(condition, dataset_name):
        # A derived (*_nfkc) condition is run as a SEPARATE invocation from
        # its source in the plan's own SLURM script (02_encode.sh), so it
        # must read the source's already-written output from disk rather
        # than rely on any in-memory state from this call.
        source_condition = DERIVED_CONDITIONS[condition]
        source_path = os.path.join(base_dir, source_condition, f"{dataset_name}.json")
        if not os.path.exists(source_path):
            raise RuntimeError(
                f"{condition} requires {source_path} to already exist — "
                f"encode the '{source_condition}' condition first."
            )
        with open(source_path) as f:
            items = json.load(f)
        return {item["id"]: item for item in items}

    for condition in conditions:
        print(f"Encoding condition: {condition}")
        cond_dir = os.path.join(base_dir, condition)
        os.makedirs(cond_dir, exist_ok=True)

        textattack_wrapper = _make_textattack_wrapper(condition) if condition in TEXTATTACK_CONDITIONS else None

        if "documents" in datasets:
            checkpoint_path = os.path.join(cond_dir, ".checkpoint.documents.json")
            aux = (_load_derived_source(condition, "documents")
                   if condition in DERIVED_CONDITIONS else textattack_wrapper)
            encoded_docs = encode_condition(condition, documents, config, checkpoint_path,
                                             proxy=proxy, aux=aux)
            with open(os.path.join(cond_dir, "documents.json"), "w") as f:
                json.dump(encoded_docs, f, indent=2)

        if "gradient" in datasets and condition in GRADIENT_CONDITIONS:
            checkpoint_path = os.path.join(cond_dir, ".checkpoint.gradient.json")
            aux = (_load_derived_source(condition, "gradient")
                   if condition in DERIVED_CONDITIONS else textattack_wrapper)
            encoded_grad = encode_condition(condition, gradient, config, checkpoint_path,
                                             proxy=proxy, aux=aux)
            with open(os.path.join(cond_dir, "gradient.json"), "w") as f:
                json.dump(encoded_grad, f, indent=2)

    print("Encoding complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=["documents", "gradient"],
                         choices=["documents", "gradient"])
    parser.add_argument("--calibrate", action="store_true",
                         help="Run the threshold_tau pilot on the disjoint calibration "
                              "split and exit; does not write to data/encoded/")
    parser.add_argument("--smoke", action="store_true",
                         help="Encode only the first 3 items per dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.calibrate:
        run_calibration(config)
        return

    conditions = args.conditions or config["encoding_conditions"]

    if args.smoke:
        # Never touches data/raw/ or data/encoded/ — reads the real raw
        # files but truncates in memory, and writes to a separate smoke
        # output dir so a 3-item "complete" checkpoint can never be
        # mistaken for a finished real run.
        run_encode(config, conditions, args.datasets, item_limit=3,
                   output_dir="data/encoded_smoke")
    else:
        run_encode(config, conditions, args.datasets)


if __name__ == "__main__":
    main()
