"""
src/ghost_agent.py

GHOST self-improving agent (Task 5 of GHOST_AGENT_TASKS.md).

The agent loop:
  1. Build iteration prompt with reflection history
  2. Call the agent backbone (src/agent_backbone.py) with tools
  3. Execute the tool calls the agent makes (ghost_tools.py)
  4. Extract the agent's proposed encoding
  5. Query the ensemble panel (src/ensemble.py) -- NOT a single
     adversary, see reflection.py/ghost_agent_prompt.py for why
  6. Check the consensus verdict
  7. Update reflection memory
  8. Repeat or stop

The agent backbone reasons and proposes. The ensemble panel (configured
at runtime via config.yaml's agent_ensemble, resolved by
src/ensemble.py's load_ensemble_config) attempts extraction. These are
always separate models with separate roles -- the panel is never
hardcoded here, it's threaded through as ensemble_members/
consensus_threshold arguments.

DEVIATION FROM GHOST_AGENT_TASKS.md's CRITICAL RULE 5 (which fixed the
agent model to claude-sonnet-4-6): the agent role is now backbone-
configurable (src/agent_backbone.py, config.yaml's agent_backbones) so
backbone choice itself can be an ablation axis (Claude Sonnet/Haiku vs.
local DeepSeek-R1 distills), same in kind as the earlier single-
adversary -> ensemble-panel deviation. run_ghost_agent still defaults to
claude-sonnet-4-6 (AGENT_MODEL below) when no backbone/agent_backbone_name
is given, so every pre-existing call site (smoke tests, verify_*.py
scripts) is unaffected.
"""

import sys
import os
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(__file__))
from ghost_tools import TOOL_FUNCTIONS, tool_render, tool_get_stored, tool_hamming  # noqa: E402
from ghost_agent_prompt import (  # noqa: E402
    GHOST_AGENT_SYSTEM_PROMPT,
    build_iteration_prompt,
)
from ensemble import run_ensemble_query  # noqa: E402
from reflection import ReflectionMemory, Attempt  # noqa: E402
from ghost_tools_structural import tool_analyse_structure  # noqa: E402
import strategy_memory  # noqa: E402
import agent_backbone as _agent_backbone_mod  # noqa: E402
from agent_backbone import (  # noqa: E402
    TOOL_DEFINITIONS,
    execute_tool_call,
    run_agent_turn,
    AnthropicBackbone,
    DEFAULT_ANTHROPIC_MODEL as AGENT_MODEL,
)


# ── Principle distillation (GHOST_self_improving.md, Component B) ──────

DISTILLATION_PROMPT = """You just successfully encoded text to defeat \
a panel of models.

Content profile:
  Type: {content_type}
  Length: {n_chars} characters
  Sentences: {n_sentences}
  Has numbers: {has_numbers}

What worked (your own reasoning from that winning attempt):
  {agent_reasoning}

Iterations needed: {n_iterations}
Hamming distance achieved: {hamming}/{total_len}

Write ONE abstract principle (max 40 words) that generalises this
success to similar content. It must mention:
  1. What content profile this applies to
  2. What strategy to try first
  3. Why it works mechanistically

Do not mention the specific string. Generalise."""


def distil_principle(success_data: dict, backbone) -> str:
    """
    Call the agent backbone to distil an abstract, generalisable
    principle from one successful encoding attempt.

    `backbone` should be the SAME backbone instance run_ghost_agent is
    already using for this field (passed through, not a fresh one) --
    both because a second real backbone means a second set of real
    API credentials/GPU load to reason about, and because tests that
    inject a fake client need every model call in a run to go through
    that one fake, not silently open a live connection for this
    specific step. This also means distillation uses whichever backbone
    ran the field -- a local-model ablation run distils principles with
    the local model too, not a hardcoded Anthropic call.

    There is no structured bidi_config/vs_payload to cite here (the
    agent emits one already-combined <final_encoding> string, not
    separate config/payload fields -- see Attempt's "from_agent"
    placeholders in run_ghost_agent) -- so this prompts on the agent's
    own free-text reasoning instead, which is the only place that
    information actually lives.
    """
    return backbone.generate_text(
        DISTILLATION_PROMPT.format(**success_data), max_tokens=150,
    )


# TOOL_DEFINITIONS, execute_tool_call, and run_agent_turn now live in
# agent_backbone.py (imported above) -- re-exported here unchanged so
# existing direct imports (e.g. smoke_test_task5.py's
# `from ghost_agent import run_agent_turn, TOOL_DEFINITIONS`) keep working.


def extract_proposed_encoding(agent_text: str, target: str) -> Optional[str]:
    """
    Extract the agent's proposed encoding from its text response.

    The agent is instructed (ghost_agent_prompt.py) to wrap ONLY the
    raw encoded string in <final_encoding>...</final_encoding> tags.
    Delimiter-based extraction, not a text-marker + substring-match
    heuristic: a correctly obfuscated encoding contains none of the
    target's plaintext characters, so requiring target[:3] to appear
    literally (the original heuristic) would reject every genuinely
    obfuscated proposal and only accept ones contaminated with
    plaintext commentary on the same line -- confirmed as a real bug
    in this repo's Task 5 smoke test, not a hypothetical.

    Returns the encoding string or None if the tag is missing/empty.
    `target` is unused now but kept in the signature since callers
    already pass it and a future stricter check (e.g. requiring the
    tag's content to actually render back to target) may want it.
    """
    import re
    match = re.search(
        r"<final_encoding>(.*?)</final_encoding>", agent_text, re.DOTALL,
    )
    if match is None:
        return None
    encoding = match.group(1).strip()
    return encoding or None


def run_ghost_agent(
    target: str,
    field_name: str,
    ensemble_members: list,
    consensus_threshold: int,
    clean_floor_check: bool = True,
    max_iterations: int = 5,
    tool_budget_per_iter: int = 8,
    agent_response_max_tokens: int = 2000,
    verbose: bool = True,
    client=None,
    backbone=None,
    agent_backbone_name: Optional[str] = None,
    backbone_config: Optional[dict] = None,
    ensemble_query_fn: Callable = run_ensemble_query,
    doc_index: Optional[int] = None,
    use_memory: bool = True,
) -> dict:
    """
    Run the GHOST self-improving agent on a single field value.

    Args:
        target: the original text to protect
        field_name: human-readable field name
        ensemble_members: resolved member list, e.g.
            ensemble.load_ensemble_config(config)["members"]
        consensus_threshold: members that must fail to declare defended
        clean_floor_check: exclude members that can't read plain text
        max_iterations: maximum refinement iterations
        tool_budget_per_iter: max tool calls per iteration
        agent_response_max_tokens: per-API-call output token budget --
            see agent_backbone.run_agent_turn's response_max_tokens
            docstring; raise for targets much longer than a short
            numeric field
        verbose: print progress
        client: an Anthropic-SDK-shaped client (`.messages.create`).
            BACK-COMPAT ONLY, used when `backbone` and
            `agent_backbone_name` are both omitted: builds a default
            AnthropicBackbone(model_id=AGENT_MODEL, client=client), same
            as this function's original pre-backbone behaviour, so
            every existing call site (smoke tests, verify_*.py scripts)
            is unaffected. Ignored if `backbone` is given.
        backbone: a pre-constructed agent_backbone instance (see
            agent_backbone.py) -- pass this when running the SAME
            backbone across many fields/documents (e.g. convergence.py's
            ablation loop over a local model), since local backbones
            load weights once and reset() is called here per field, not
            reloaded. Caller owns close()/GPU lifecycle in this case.
        agent_backbone_name: alternative to `backbone` -- a key into
            `backbone_config`/config.yaml's agent_backbones.options,
            resolved fresh via agent_backbone.resolve_backbone() and
            closed automatically at the end of THIS call. Convenient
            for one-off runs; wasteful for a multi-field ablation loop
            over a local model (reloads weights every call) -- prefer
            `backbone` there instead.
        backbone_config: parsed config.yaml dict, required only when
            agent_backbone_name is given.
        ensemble_query_fn: defaults to ensemble.run_ensemble_query --
            overridable for tests so this loop is checkable without
            loading any real model weights.
        doc_index: caller's global document/field counter, purely for
            logging into the experience log (GHOST_self_improving.md's
            Component A/Question 4) -- has no effect on the loop itself.
        use_memory: gates ALL of strategy_memory.py's persistent-memory
            behaviour (load/retrieve/inject at start, experience-log
            append per iteration, distil+save on success). False
            reproduces this function's exact pre-memory behaviour --
            used by tests that inject a fake client/ensemble_query_fn
            and must not touch real files or make an extra real API
            call for distillation.

    Returns:
        dict with keys:
          success: bool — was extraction defeated?
          best_encoding: str — the best encoding found
          n_iterations: int — iterations used
          history: list — all attempts
          final_hamming: int — best hamming distance achieved
          n_principles_available: int — principles retrieved and
              injected at the start of this run (0 if use_memory=False)
    """
    owns_backbone = False
    if backbone is None:
        if agent_backbone_name is not None:
            if backbone_config is None:
                raise ValueError(
                    "backbone_config is required when agent_backbone_name is given"
                )
            backbone = _agent_backbone_mod.resolve_backbone(
                agent_backbone_name, backbone_config,
            )
        else:
            backbone = AnthropicBackbone(model_id=AGENT_MODEL, client=client)
        owns_backbone = True

    backbone.reset()

    try:
        return _run_ghost_agent_loop(
            target=target,
            field_name=field_name,
            ensemble_members=ensemble_members,
            consensus_threshold=consensus_threshold,
            clean_floor_check=clean_floor_check,
            max_iterations=max_iterations,
            tool_budget_per_iter=tool_budget_per_iter,
            agent_response_max_tokens=agent_response_max_tokens,
            verbose=verbose,
            backbone=backbone,
            ensemble_query_fn=ensemble_query_fn,
            doc_index=doc_index,
            use_memory=use_memory,
        )
    finally:
        if owns_backbone:
            backbone.close()


def _run_ghost_agent_loop(
    target: str,
    field_name: str,
    ensemble_members: list,
    consensus_threshold: int,
    clean_floor_check: bool,
    max_iterations: int,
    tool_budget_per_iter: int,
    agent_response_max_tokens: int,
    verbose: bool,
    backbone,
    ensemble_query_fn: Callable,
    doc_index: Optional[int],
    use_memory: bool,
) -> dict:
    """The actual iteration loop, split out of run_ghost_agent so backbone
    construction/reset/close (which differ between the default-Anthropic
    back-compat path and an externally-supplied/named backbone) stay in
    one place, wrapped in a single try/finally."""
    content_profile = tool_analyse_structure(target)

    system_prompt = GHOST_AGENT_SYSTEM_PROMPT
    n_principles_available = 0
    if use_memory:
        loaded_memory = strategy_memory.load_memory()
        principles = strategy_memory.retrieve_relevant_principles(
            target, loaded_memory,
        )
        n_principles_available = len(principles)
        principles_text = strategy_memory.format_principles_for_prompt(
            principles,
        )
        if principles_text:
            system_prompt = GHOST_AGENT_SYSTEM_PROMPT + "\n\n" + principles_text
        if verbose and principles:
            print(f"Loaded {len(principles)} relevant principle(s) from memory")

    memory = ReflectionMemory(target=target, field_name=field_name)
    best_encoding = None
    best_hamming = 0

    # The clean-floor check needs a plain-text sample containing the
    # SAME target value, unobfuscated -- it's testing "can this member
    # read this value at all", not testing a different dataset value.
    clean_reference_text = f"{field_name}: {target}"
    clean_reference_value = target

    if verbose:
        print(f"\n{'='*55}")
        print(f"GHOST-Agent: protecting '{target}' ({field_name})")
        print(f"Ensemble members: {[m['name'] for m in ensemble_members]}")
        print(f"Consensus threshold: {consensus_threshold}")
        print(f"Max iterations: {max_iterations}")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        reflection = memory.format_for_reflection()
        user_msg = build_iteration_prompt(
            target=target,
            field_name=field_name,
            reflection=reflection,
            iteration=iteration,
            budget_remaining=tool_budget_per_iter,
        )
        agent_text = backbone.run_turn(
            user_msg, system_prompt,
            tool_budget_per_iter=tool_budget_per_iter,
            response_max_tokens=agent_response_max_tokens,
        )

        if verbose:
            print(f"Agent reasoning:\n{agent_text[:300]}...")

        proposed = extract_proposed_encoding(agent_text, target)

        if proposed is None:
            if verbose:
                print("Could not extract encoding from agent text")
            continue

        rendered = tool_render(proposed)
        stored = tool_get_stored(proposed)
        dist = tool_hamming(stored, target)

        if verbose:
            print(f"Proposed encoding renders as: '{rendered}'")
            print(f"Stored sequence: '{stored}'")
            print(f"Hamming distance: {dist}")

        if dist > best_hamming:
            best_hamming = dist
            best_encoding = proposed

        # A resident local backbone (e.g. deepseek_r1_32b) and the
        # search-tier ensemble's own per-member GPU residency can both
        # need the card at once -- confirmed OOM on a real run when both
        # were held simultaneously. release_gpu()/reacquire_gpu() are
        # no-ops for AnthropicBackbone; see agent_backbone.py's
        # LocalReActBackbone docstring for why this round-trip exists.
        backbone.release_gpu()
        try:
            ensemble_result = ensemble_query_fn(
                encoded_text=proposed,
                field_name=field_name,
                ground_truth=target,
                members=ensemble_members,
                consensus_threshold=consensus_threshold,
                clean_reference_text=clean_reference_text,
                clean_reference_value=clean_reference_value,
                clean_floor_check=clean_floor_check,
            )
        finally:
            backbone.reacquire_gpu()

        if verbose:
            print(
                f"Ensemble result: {ensemble_result['n_failed']}/"
                f"{ensemble_result['n_valid']} valid members defeated"
            )
            print(f"Defended: {ensemble_result['defended']}")

        attempt = Attempt(
            iteration=iteration,
            bidi_config="from_agent",
            vs_payload="from_agent",
            encoding_description=agent_text[:200],
            encoded_text=proposed,
            rendered=rendered,
            stored=stored,
            hamming_dist=dist,
            ensemble_result=ensemble_result,
            agent_reasoning=agent_text[:500],
        )
        memory.add_attempt(attempt)

        if use_memory:
            import datetime as _datetime
            strategy_memory.append_experience({
                "timestamp": _datetime.datetime.now().isoformat(),
                "doc_index": doc_index,
                "field_name": field_name,
                "content_profile": content_profile,
                "bidi_config": attempt.bidi_config,
                "vs_payload": attempt.vs_payload,
                "hamming_dist": dist,
                "adversary_model": [m["name"] for m in ensemble_members],
                "adversary_response": [
                    m.get("response") for m in ensemble_result["per_member"]
                ],
                "extraction_succeeded": not ensemble_result["defended"],
                "iteration": iteration,
                "agent_reasoning": agent_text[:500],
            })

        result_msg = (
            f"Ensemble panel result: {ensemble_result['n_failed']}/"
            f"{ensemble_result['n_valid']} valid members defeated.\n"
            f"Consensus threshold: {consensus_threshold}.\n"
            f"{'DEFENDED' if ensemble_result['defended'] else 'NOT YET DEFENDED'}."
        )
        backbone.messages.append({"role": "user", "content": result_msg})

        if ensemble_result["defended"]:
            if verbose:
                print(
                    f"\nSUCCESS: Ensemble consensus defeated in "
                    f"{iteration} iterations!"
                )
            best_encoding = proposed

            if use_memory:
                try:
                    principle = distil_principle({
                        "content_type": content_profile["content_type"],
                        "n_chars": content_profile["n_chars"],
                        "n_sentences": content_profile["n_sentences"],
                        "has_numbers": content_profile["has_numbers"],
                        "agent_reasoning": agent_text[:500],
                        "n_iterations": iteration,
                        "hamming": dist,
                        "total_len": len(target),
                    }, backbone)
                except Exception as e:
                    if verbose:
                        print(f"Principle distillation failed: {e}")
                    principle = None

                if principle:
                    # NOTE: n_documents_processed is deliberately NOT
                    # touched here. run_ghost_agent operates per FIELD,
                    # not per document (one document has several
                    # fields), so incrementing it on every successful
                    # field-level distillation double(+)counted
                    # documents -- confirmed for real: 18 recorded after
                    # only 6 documents (23 field-runs, 18 successes)
                    # across two runs. The document-level count is
                    # convergence.py's job (it's the only caller that
                    # actually knows where one document ends and the
                    # next begins), incremented once per document after
                    # all of that document's fields are done.
                    loaded_memory = strategy_memory.add_principle(
                        memory=loaded_memory,
                        principle=principle,
                        content_profile=content_profile,
                        bidi_config=attempt.bidi_config,
                        vs_payload=attempt.vs_payload,
                        n_iterations=iteration,
                        adversary_model=",".join(
                            m["name"] for m in ensemble_members
                        ),
                    )
                    strategy_memory.save_memory(loaded_memory)
                    if verbose:
                        print(f"Distilled principle: {principle}")

            break

    return {
        "success": memory.succeeded,
        "best_encoding": best_encoding,
        "n_iterations": memory.n_attempts,
        "final_hamming": best_hamming,
        "history": memory.attempts,
        "target": target,
        "field_name": field_name,
        "n_principles_available": n_principles_available,
    }
