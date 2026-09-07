"""
src/ensemble.py

Search-tier adversary ensemble for GHOST-Agent (extends Task 2 of
GHOST_AGENT_TASKS.md; see config.yaml's agent_ensemble section).

Owns the load-once / query-many / unload lifecycle for the local HF
models in agent_ensemble.members -- reloading a 7-14B model per field
query would defeat the entire cost rationale for using an open-source
ensemble as a search-time proxy instead of frontier APIs. Models are
loaded ONE AT A TIME on the shared GPU and unloaded before the next,
mirroring src/encode.py's proxy-model convention.

This module answers exactly one question per call: does a supermajority
of the ensemble fail to extract the target from encoded_text? That
verdict drives the agent's stopping condition during search. It is
NEVER the paper's reported number -- config.yaml's
agent_ensemble.frontier_verify_models is the tier that gets reported,
and the gap between the two tiers should itself be measured and
reported (see project discussion: search-tier is a proxy, same status
as the Qwen logprob proxy in encode.py).
"""

import gc
import os
import sys
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(__file__))
from adversary import query_adversary_local, check_extraction  # noqa: E402
from prior_strength import score_prior_strength  # noqa: E402


# ── Model loading (real implementation; swappable for tests) ────────────

def load_local_model(hf_id: str, dtype: str = "bfloat16"):
    """
    Load a HF causal LM + tokenizer onto the shared GPU.

    Returns (tokenizer, model). For gated repos (meta-llama, mistralai)
    this relies on the standard huggingface_hub auth flow already
    configured in the environment -- not this function's concern.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map="auto", torch_dtype=torch_dtype,
    )
    return tokenizer, model


def unload_local_model(tokenizer, model) -> None:
    """Free a loaded model's GPU memory before loading the next one."""
    import torch

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ── Config resolution ─────────────────────────────────────────────────

def load_ensemble_config(config: dict) -> dict:
    """
    Resolve config.yaml's agent_ensemble section into a ready-to-use
    members list (local_models entries keyed by name) plus voting params.

    Args:
        config: the parsed config.yaml dict (see src/encode.py for the
                yaml.safe_load convention used elsewhere in this repo)

    Returns:
        dict with keys: members (list of {name, hf_id, max_new_tokens,
        dtype}), consensus_threshold (int), clean_floor_check (bool)
    """
    agent_ensemble = config["agent_ensemble"]
    local_models = config["local_models"]

    members = []
    for name in agent_ensemble["members"]:
        member_config = dict(local_models[name])
        member_config["name"] = name
        members.append(member_config)

    return {
        "members": members,
        "consensus_threshold": agent_ensemble["consensus_threshold"],
        "clean_floor_check": agent_ensemble["clean_floor_check"],
    }


# ── Ensemble orchestration ───────────────────────────────────────────────

def run_ensemble_query(
    encoded_text: str,
    field_name: str,
    ground_truth: str,
    members: list,
    consensus_threshold: int,
    clean_reference_text: Optional[str] = None,
    clean_reference_value: Optional[str] = None,
    clean_floor_check: bool = True,
    loader: Callable = load_local_model,
    unloader: Callable = unload_local_model,
    querier: Callable = query_adversary_local,
    checker: Callable = check_extraction,
    frontier_check_fn: Optional[Callable] = None,
) -> dict:
    """
    Run one field's encoded_text through every ensemble member, one at a
    time, and return a consensus verdict.

    A member's vote only counts if it passes the clean-floor check first
    (can it extract clean_reference_value from clean_reference_text at
    all?) -- otherwise a member's "failure" on encoded_text is as likely
    to be weak instruction-following as a working defense, which would
    corrupt the whole point of using real extraction failure as signal
    (see GHOST_AGENT_TASKS.md's rationale for replacing the logprob
    proxy). A refusal counts as "not extracted" here, consistent with
    check_extraction and with this repo's convention that refusals are a
    defense success.

    Args:
        encoded_text: the GHOST-encoded text to test
        field_name: the field being extracted
        ground_truth: the original (unencoded) field value
        members: list of dicts, each with at least
                 {"name": str, "hf_id": str, "max_new_tokens": int,
                 "dtype": str} -- i.e. load_ensemble_config's output
        consensus_threshold: number of valid (floor-check-passed)
                 members that must fail extraction to declare the
                 encoding defended
        clean_reference_text / clean_reference_value: an unencoded
                 sample used for the clean-floor check
        clean_floor_check: set False to skip (e.g. tests where the
                 mocked querier has no notion of "clean")
        loader/unloader/querier/checker: dependency-injection points so
                 this function is unit-testable without a GPU, and so
                 reconstruction-mode callers (Phase 6 gradient /
                 GHOST-Agent-on-natural-language) can swap in
                 query_adversary_local(prompt_template=RECONSTRUCTION_
                 PROMPT) + check_reconstruction without duplicating the
                 load/unload/floor-check lifecycle. Real field-extraction
                 callers use the defaults.
        frontier_check_fn: CALIBRATION-ONLY gate, None by default (every
                 real per-document/per-field eval run must leave this
                 unset). When given, a real frontier API model (see
                 src/calibrate_agent_search.py's make_frontier_check_fn)
                 is consulted, but ONLY on attempts the cheap local
                 ensemble already calls "defended" -- this is what lets
                 an agent's search loop learn to escalate PAST a
                 local-ensemble pass when a stronger held-out check would
                 still extract it, without spending an API call on every
                 iteration. This must never be wired into the actual
                 per-document GHOST-Agent search used for reported
                 results: doing so would mean the reported search
                 "succeeded" against the exact model whose defeat is
                 later claimed as a transferability result -- the same
                 proxy-grading-its-own-homework risk this file's module
                 docstring already flags for the search tier itself.
                 Use it only against the disjoint data/raw/calibration.json
                 split, to learn better STOPPING heuristics (e.g. "require
                 a margin past first local-ensemble pass") that then get
                 applied blind (frontier_check_fn=None again) on the real
                 eval set.

    Returns:
        dict with keys:
          defended: bool — did >= consensus_threshold valid members fail,
                    AND (if frontier_check_fn was given and triggered)
                    did that frontier check also fail to extract?
          n_valid: int — members that passed the clean-floor check
          n_failed: int — valid members that did not extract correctly
          per_member: list of per-member result dicts (includes the
                    frontier check, clearly labeled, when it ran)
          frontier_check: the frontier_check_fn result dict, only present
                    if it was actually called this attempt
    """
    if clean_floor_check and (
        clean_reference_text is None or clean_reference_value is None
    ):
        raise ValueError(
            "clean_floor_check requires clean_reference_text and "
            "clean_reference_value"
        )

    per_member = []

    for member in members:
        tokenizer, model = loader(member["hf_id"], member.get("dtype", "bfloat16"))
        max_new_tokens = member.get("max_new_tokens", 50)

        try:
            valid = True
            floor_result = None
            if clean_floor_check:
                floor_response = querier(
                    tokenizer, model, clean_reference_text,
                    field_name, max_new_tokens,
                )
                floor_result = checker(
                    floor_response, clean_reference_value,
                )
                valid = floor_result["extracted"]

            response = querier(
                tokenizer, model, encoded_text, field_name, max_new_tokens,
            )
            result = checker(response, ground_truth)

            per_member.append({
                "name": member["name"],
                "hf_id": member["hf_id"],
                "valid": valid,
                "floor_check": floor_result,
                "extracted": result["extracted"],
                "refusal": result["refusal"],
                "response": result["response"],
            })
        finally:
            unloader(tokenizer, model)

    valid_members = [m for m in per_member if m["valid"]]
    n_valid = len(valid_members)
    n_failed = sum(1 for m in valid_members if not m["extracted"])
    local_defended = n_failed >= consensus_threshold

    result = {
        "defended": local_defended,
        "n_valid": n_valid,
        "n_failed": n_failed,
        "per_member": per_member,
    }

    # Calibration-only escalation gate -- see docstring. Only spend a real
    # API call once the cheap tier already thinks it's won; if the local
    # ensemble hasn't converged yet there is nothing to escalate past.
    if local_defended and frontier_check_fn is not None:
        frontier_result = frontier_check_fn(encoded_text, field_name, ground_truth)
        result["frontier_check"] = frontier_result
        result["per_member"] = per_member + [{
            "name": f"{frontier_result['name']} [CALIBRATION-ONLY frontier gate]",
            "hf_id": None,
            "valid": True,
            "floor_check": None,
            "extracted": frontier_result["extracted"],
            "refusal": frontier_result["refusal"],
            "response": frontier_result["response"],
        }]
        if frontier_result["extracted"]:
            result["defended"] = False

    return result


# ── Prior-strength scoring ────────────────────────────────────────────

def run_ensemble_prior_strength(
    target: str,
    context: str,
    members: list,
    loader: Callable = load_local_model,
    unloader: Callable = unload_local_model,
    scorer: Callable = score_prior_strength,
) -> dict:
    """
    Score `target`'s prior-strength profile (see src/prior_strength.py)
    against every search-tier ensemble member, one at a time, and
    average the ratio across them.

    Computed ONCE per field on the RAW target, before any encoding --
    this characterizes the content itself (how much of it an LLM's own
    prior already explains, beyond generic compressibility), not any
    particular encoding attempt. Same load-one-at-a-time-then-unload
    lifecycle as run_ensemble_query, so cost is roughly one extra pass
    through the ensemble's load/unload cycle per field, not per
    iteration.

    Averaging the RATIO (not raw per-token logprob) across members is
    deliberate: the ratio is already normalized against each member's
    own surprisal via the same tokenizer-agnostic zlib reference, so an
    average across genuinely different tokenizer families (this
    ensemble's whole reason for existing) is comparing like with like in
    a way raw cross-tokenizer logprob averaging would not be.

    Args:
        target: the string being scored (e.g. a field value or sentence).
        context: non-empty preceding text (see prior_strength._per_token_
                 bits -- required so there's an anchor position to
                 predict the target's first token from).
        members: list of dicts as returned by load_ensemble_config
                 (at least {"name", "hf_id", "dtype"}).
        loader/unloader/scorer: dependency-injection points, same
                 rationale as run_ensemble_query -- unit-testable
                 without a GPU.

    Returns:
        dict with keys:
          mean_ratio: float — average prior-strength ratio across
                    members (near 0 = ensemble-wide familiarity/
                    memorization; near 1 = no ensemble member has an
                    edge over generic compression).
          per_member: list of {"name", "hf_id", "ratio", "nll_bits",
                    "compressed_bits", "min_k_bits"}.
    """
    per_member = []

    for member in members:
        tokenizer, model = loader(member["hf_id"], member.get("dtype", "bfloat16"))
        try:
            profile = scorer(tokenizer, model, target, context)
            per_member.append({
                "name": member["name"],
                "hf_id": member["hf_id"],
                **profile,
            })
        finally:
            unloader(tokenizer, model)

    mean_ratio = (
        sum(m["ratio"] for m in per_member) / len(per_member)
        if per_member else float("nan")
    )

    return {
        "mean_ratio": mean_ratio,
        "per_member": per_member,
    }
