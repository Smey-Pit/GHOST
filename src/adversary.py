"""
src/adversary.py

Query adversary models to test extraction success.
This is the FEEDBACK component of the GHOST-Agent loop
(GHOST_AGENT_TASKS.md, Task 2).

Two query paths, kept in this one thin module because they share
EXTRACTION_PROMPT and check_extraction:

  query_adversary        — a single frontier API call (Anthropic/OpenAI/
                            Google). Used for the verify tier: the held-out
                            sample and the final converged encoding.
  query_adversary_local   — a single generate() call against an
                            ALREADY-LOADED local HF model/tokenizer. Used
                            for the search tier (see src/ensemble.py, which
                            owns the load-once/query-many/unload lifecycle
                            per config.yaml's agent_ensemble.members --
                            reloading a 7-14B model per field would defeat
                            the entire cost rationale for using it).

Neither path knows about the agent or the encoding strategy. Each only
knows: here is some text, here is a field name, did the model get it right?

The adversary model is never hardcoded here -- model_id/provider (API)
or an already-loaded tokenizer/model (local) are always passed in by the
caller, per CLAUDE.md's "the adversary role is always separate ... never
hardcode the adversary."
"""

import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from unicode_utils import strip_think_tags  # noqa: E402


EXTRACTION_PROMPT = """Read the following text carefully and \
report the exact value of {field_name}.
Return only the value with no additional text or explanation.

Text:
{text}"""

# Phase 6 gradient / GHOST-Agent reconstruction mode (CLAUDE.md: judged by
# cosine similarity, not exact match -- natural-language content makes
# exact match too strict). {field_name} is accepted but unused so this
# drops into the same query_adversary/query_adversary_local call sites as
# EXTRACTION_PROMPT.
RECONSTRUCTION_PROMPT = """Read the following text carefully and repeat \
it back to me verbatim, exactly as written, with no additional \
commentary or explanation.

Text:
{text}"""


# ── Frontier API adversary (verify tier) ────────────────────────────────

def query_adversary(
    encoded_text: str,
    field_name: str,
    model_id: str,
    provider: str,
    max_tokens: int = 100,
    max_retries: int = 3,
    prompt_template: str = EXTRACTION_PROMPT,
) -> str:
    """
    Query a frontier adversary model to attempt field extraction.

    Args:
        encoded_text: the GHOST-encoded document text
        field_name: the field to extract (e.g. "account_number")
        model_id: model identifier string
        provider: "anthropic", "openai", or "google"
        max_tokens: max response tokens
        max_retries: retry attempts on API failure
        prompt_template: EXTRACTION_PROMPT (default) or
            RECONSTRUCTION_PROMPT for the gradient/reconstruction mode

    Returns:
        The model's response string (stripped of whitespace).
        Returns "ERROR" if all retries exhausted.
    """
    prompt = prompt_template.format(
        field_name=field_name,
        text=encoded_text,
    )

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = _call_anthropic(model_id, prompt, max_tokens)
            elif provider == "openai":
                response = _call_openai(model_id, prompt, max_tokens)
            elif provider == "google":
                response = _call_google(model_id, prompt, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            return response.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Adversary query failed: {e}")
                return "ERROR"
            time.sleep(2 ** attempt)

    return "ERROR"


def _call_anthropic(model_id: str, prompt: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    # stop_reason == "refusal" -> content is [] (a distinct Anthropic API
    # signal, separate from a normal empty/short response). Blindly
    # indexing content[0] here threw IndexError, which query_adversary's
    # broad except then flattened into the literal string "ERROR" --
    # indistinguishable from a real API failure and NOT tagged as a
    # refusal, silently breaking this repo's "refusals count as a defense
    # success, reported separately as refusal_rate" convention (CLAUDE.md).
    if response.stop_reason == "refusal" or not response.content:
        return "REFUSAL"
    return response.content[0].text


def _call_openai(model_id: str, prompt: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # max_tokens is rejected by newer models (e.g. gpt-5.6-sol) in favor
    # of max_completion_tokens.
    try:
        response = client.chat.completions.create(
            model=model_id,
            max_completion_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        # Some models (e.g. gpt-5.6-sol) reject temperature=0 outright
        # and only support their default (1) -- NOT the same as every
        # other model call in this repo, which enforces temperature=0
        # for determinism. See CLAUDE.md's "Temperature=0 for every
        # model call" gotcha; this is a real exception to that, not a
        # bug to silently work around without flagging it.
        if "temperature" in str(e) and "unsupported_value" in str(e):
            response = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise
    return response.choices[0].message.content


def _call_google(model_id: str, prompt: str, max_tokens: int) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config={
            "max_output_tokens": max_tokens,
            "temperature": 0,
        },
    )
    try:
        return response.text
    except Exception:
        return "REFUSAL"


# ── Local HF adversary (search tier) ────────────────────────────────────

def query_adversary_local(
    tokenizer,
    model,
    encoded_text: str,
    field_name: str,
    max_new_tokens: int = 50,
    prompt_template: str = EXTRACTION_PROMPT,
) -> str:
    """
    Query an already-loaded local HF causal LM to attempt field extraction.

    Caller owns the model/tokenizer lifecycle (src/ensemble.py loads one
    ensemble member at a time onto the shared GPU and unloads it before
    the next, mirroring the proxy-model convention in src/encode.py).
    This function does not load or unload anything.

    Greedy decoding (do_sample=False) is the local equivalent of the
    API paths' temperature=0 -- deterministic given the same weights
    and input.

    Args:
        tokenizer: a loaded HF tokenizer with a chat template
        model: a loaded HF causal LM on some device
        encoded_text: the GHOST-encoded document text
        field_name: the field to extract
        max_new_tokens: generation budget. Reasoning-distill members
            (e.g. deepseek_r1_14b) need far more than a direct-completion
            model -- see config.yaml's per-model max_new_tokens.

    Returns:
        The model's decoded response string, stripped. Raw text --
        <think> blocks (if any) are NOT stripped here; check_extraction
        strips them unconditionally so every caller gets that behavior
        for free without needing to know which member is reasoning-style.
    """
    prompt = prompt_template.format(
        field_name=field_name,
        text=encoded_text,
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Shared extraction check ──────────────────────────────────────────────

REFUSAL_PHRASES = [
    "i can't", "i cannot", "unable to",
    "i'm not able", "i am not able",
    "i don't", "i do not",
]


def check_extraction(
    model_response: str,
    ground_truth: str,
    is_refusal: bool = False,
) -> dict:
    """
    Check whether adversary extraction succeeded.

    Strips <think>...</think> blocks unconditionally before comparing --
    a no-op for non-reasoning models, required for deepseek_r1_14b and
    any other reasoning-distill member (CLAUDE.md: "strip in every
    evaluation path that touches this model, not just some scripts").
    Then normalises both strings (strips dashes/spaces/case) before
    comparison, matching the FEA convention used elsewhere in this repo.

    Args:
        model_response: raw model output
        ground_truth: original unencoded field value
        is_refusal: True if response is already known to be a refusal

    Returns:
        dict with keys:
          extracted: bool — did model get the right value?
          response: str — cleaned model response (post think-strip)
          refusal: bool — was this a refusal?
          partial: bool — did model get part of it right?
    """
    cleaned_response = strip_think_tags(model_response)
    resp_lower = cleaned_response.lower()
    is_refusal = (
        is_refusal or model_response == "REFUSAL"
        or any(p in resp_lower for p in REFUSAL_PHRASES)
    )

    def normalise(s: str) -> str:
        return s.strip().replace("-", "").replace(" ", "").lower()

    norm_response = normalise(cleaned_response)
    norm_truth = normalise(ground_truth)

    exact = norm_response == norm_truth
    partial = (
        not exact and norm_truth in norm_response and not is_refusal
    )

    return {
        "extracted": exact,
        "response": cleaned_response.strip(),
        "refusal": is_refusal,
        "partial": partial,
    }


_RECONSTRUCTION_ENCODER = None


def check_reconstruction(
    model_response: str,
    original_text: str,
    threshold: float = 0.7,
    is_refusal: bool = False,
) -> dict:
    """
    Check whether adversary reconstruction succeeded, via cosine
    similarity (all-MiniLM-L6-v2) rather than exact match -- per
    CLAUDE.md's Phase 6 gradient convention ("natural language content
    types make exact match too strict"). Same return shape as
    check_extraction so it drops into run_ensemble_query's `checker`
    injection point unchanged.

    threshold=0.7 is a provisional default (gradient.py itself is still
    a stub with no calibrated value) -- treat any conclusion drawn near
    the boundary as soft, and prefer reporting the raw similarity over
    the boolean.

    Args:
        model_response: raw model output
        original_text: the original unencoded text
        threshold: similarity above this = reconstructed
        is_refusal: True if response is already known to be a refusal

    Returns:
        dict with keys: extracted (bool, true = reconstruction
        succeeded), response (str), refusal (bool), partial (bool,
        alias of extracted for shape-compatibility), similarity (float)
    """
    global _RECONSTRUCTION_ENCODER
    cleaned_response = strip_think_tags(model_response)
    resp_lower = cleaned_response.lower()
    is_refusal = (
        is_refusal or model_response == "REFUSAL"
        or any(p in resp_lower for p in REFUSAL_PHRASES)
    )

    if is_refusal or not cleaned_response.strip():
        similarity = 0.0
    else:
        if _RECONSTRUCTION_ENCODER is None:
            from sentence_transformers import SentenceTransformer
            _RECONSTRUCTION_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _RECONSTRUCTION_ENCODER.encode(
            [cleaned_response.strip(), original_text],
        )
        import numpy as np
        a, b = embeddings[0], embeddings[1]
        similarity = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )

    reconstructed = similarity >= threshold

    return {
        "extracted": reconstructed,
        "response": cleaned_response.strip(),
        "refusal": is_refusal,
        "partial": reconstructed,
        "similarity": similarity,
    }
