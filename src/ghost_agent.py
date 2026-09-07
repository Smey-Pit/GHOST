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
from ghost_tools import (  # noqa: E402
    TOOL_FUNCTIONS, tool_render, tool_get_stored, tool_hamming,
    set_proxy_model, clear_proxy_model,
    set_wider_context, clear_wider_context,
)
from ghost_agent_prompt import (  # noqa: E402
    GHOST_AGENT_SYSTEM_PROMPT,
    build_iteration_prompt,
)
from ensemble import run_ensemble_query  # noqa: E402
from reflection import ReflectionMemory, Attempt  # noqa: E402
from ghost_tools_structural import tool_analyse_structure  # noqa: E402
from adversary import REFUSAL_PHRASES  # noqa: E402
import strategy_memory  # noqa: E402
import agent_backbone as _agent_backbone_mod  # noqa: E402
from agent_backbone import (  # noqa: E402
    TOOL_DEFINITIONS,
    execute_tool_call,
    extract_config_from_tool_calls,
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


_MIN_PRINCIPLE_LENGTH = 20


def _is_valid_principle(text: Optional[str]) -> bool:
    """
    Reject distil_principle output that isn't actually a usable
    principle before it ever reaches strategy_memory.add_principle.

    Real bug found 2026-08-19: distil_principle's return value was
    treated as valid the moment it was a non-empty string -- on 2/10
    real examples the distillation call itself refused ("I can't help
    with this request.") or apologized instead of producing a
    principle, and `if principle:` let a plain refusal string through
    to be persisted into results/strategy_memory.json as if it were
    real learned strategy (confirmed directly by reading the file after
    that run). Reuses adversary.REFUSAL_PHRASES -- the same substring
    list already used to detect a refusal from an ADVERSARY response,
    since a distillation-call refusal reads the same way. Also rejects
    a degenerate too-short result, which a genuine principle (a
    template-filled sentence or two) should never be.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < _MIN_PRINCIPLE_LENGTH:
        return False
    lower = stripped.lower()
    if any(p in lower for p in REFUSAL_PHRASES):
        return False
    return True


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
    field_ground_truth: Optional[str] = None,
    frontier_check_fn: Optional[Callable] = None,
    proxy=None,
    wider_context: Optional[str] = None,
    require_unanimous: bool = False,
    prior_strength_fn: Optional[Callable] = None,
    min_sentence_scope_hamming_fraction: float = 0.3,
) -> dict:
    """
    Run the GHOST self-improving agent on a single field value.

    Args:
        target: the text the agent is asked to obfuscate. Historically
            always a bare field value -- can now also be a whole sentence
            containing a field (sentence-level obfuscation), in which case
            pass field_ground_truth (below) so extraction is graded
            against the field, not the sentence.
        field_name: human-readable field name
        field_ground_truth: the actual value an adversary must recover to
            count as a successful extraction. None (default) means
            field_ground_truth == target -- EXACTLY today's pre-existing
            behavior for every current caller (bare-field scope), zero
            change. Pass this explicitly when target is a SENTENCE
            containing the field: clean_reference_value and the
            ground_truth passed to ensemble_query_fn both use this value
            instead of target, so "was the field recovered" is graded
            correctly even though the agent is obfuscating (and the
            adversary is reading) the surrounding sentence too.
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
        frontier_check_fn: CALIBRATION-ONLY, None by default -- forwarded
            unchanged to ensemble_query_fn's own frontier_check_fn (see
            ensemble.run_ensemble_query's docstring for the full
            rationale). Every real per-document/per-field eval run must
            leave this None; only src/calibrate_agent_search.py (run
            against the disjoint data/raw/calibration.json split) should
            ever pass one. Passing this against the actual eval/Track A
            data would optimize the search directly against whichever
            frontier model is checked, invalidating that model's later
            transferability number -- exactly the risk already flagged
            for qwen25_7b (encoding-time proxy) and deepseek_r1_14b
            (ensemble member vs. agent backbone conflict).
        proxy: an already-loaded encode.ProxyModel (or None). When given,
            registers it via ghost_tools.set_proxy_model() before the
            loop and clears it in `finally` -- this is what makes
            ghost_tools.py's bidi_permute/encode_vs_logprob/
            combine_permute tools actually usable to the agent (see
            ghost_tools.py's own docstring on why these need a live
            proxy). Caller still owns loading/unloading the proxy model
            itself and must not let it collide on GPU with the
            search-tier ensemble or a local backbone -- this function
            only manages the ghost_tools module-level registration, not
            the model's GPU residency.
        use_memory: gates ALL of strategy_memory.py's persistent-memory
            behaviour (load/retrieve/inject at start, experience-log
            append per iteration, distil+save on success). False
            reproduces this function's exact pre-memory behaviour --
            used by tests that inject a fake client/ensemble_query_fn
            and must not touch real files or make an extra real API
            call for distillation.
        wider_context: optional wider text (e.g. the sentence containing
            the current field) the agent may request via its
            widen_scope() tool when its current target can't reach full
            consensus on its own. None (default) means widen_scope()
            returns an ERROR result and the agent must keep working with
            `target` -- reproduces this function's exact pre-widening
            behaviour for every existing caller. Registered via
            ghost_tools.set_wider_context() before the loop, cleared in
            `finally`, same lifecycle as `proxy` below. field_ground_truth
            must still be the real field value in this case (the agent
            widening mid-run does not change what counts as a successful
            extraction).
        require_unanimous: if True, the loop only stops on FULL consensus
            (every valid ensemble member defeated), not just
            consensus_threshold -- the previous binary "3-of-4 clears it"
            rule let the agent stop on iteration 1 essentially every real
            run tested so far, with no margin and no incentive to ever
            retry a seed or widen scope. False (default) reproduces the
            exact pre-existing stopping rule. When True and the agent
            only reaches consensus_threshold (not full consensus), the
            loop keeps going and the per-iteration result message tells
            the agent its margin is thin -- this is what actually gives
            widen_scope()/seed-retry a reason to fire.
        prior_strength_fn: optional (target, context) -> dict callable,
            e.g. functools.partial(ensemble.run_ensemble_prior_strength,
            members=ensemble_members, loader=..., unloader=...) -- same
            "caller pre-binds everything except the two real arguments"
            convention as frontier_check_fn. None (default) skips prior-
            strength scoring entirely -- zero cost, zero behavior change
            for every existing caller. When given, called ONCE before
            the loop (it characterizes the raw target's content, not
            any particular encoding attempt -- see src/prior_strength.py)
            and merged into content_profile as "prior_strength_ratio"/
            "prior_strength_detail", which then flows into the
            experience log and any principle distilled from a successful
            attempt. A real error here (e.g. a member fails to load) is
            caught and logged, not raised -- this is a research signal,
            not something that should crash a field's search.
        min_sentence_scope_hamming_fraction: only enforced when
            field_ground_truth is given (sentence scope) -- an attempt
            whose Hamming distance is below this fraction of
            len(effective_target) is rejected and retried (no Attempt
            recorded), same treatment as a proposed encoding that fails
            to parse. Catches a real failure mode found 2026-08-19: the
            agent satisfying "render back to the full target" by
            copying the entire surrounding sentence as exact verbatim
            plaintext and only obfuscating the field itself -- field-
            scope wearing sentence-scope's clothing. Default 0.3 sits
            well below every genuine sentence-scope attempt observed so
            far (92-100%) and well above the two real bugged cases found
            (8-9%). Ignored entirely for bare-field-scope calls
            (field_ground_truth=None) -- zero behavior change there.

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
    if proxy is not None:
        set_proxy_model(proxy)
    if wider_context is not None:
        set_wider_context(wider_context)

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
            field_ground_truth=field_ground_truth,
            frontier_check_fn=frontier_check_fn,
            wider_context=wider_context,
            require_unanimous=require_unanimous,
            prior_strength_fn=prior_strength_fn,
            min_sentence_scope_hamming_fraction=min_sentence_scope_hamming_fraction,
        )
    finally:
        if owns_backbone:
            backbone.close()
        if proxy is not None:
            clear_proxy_model()
        if wider_context is not None:
            clear_wider_context()


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
    field_ground_truth: Optional[str] = None,
    frontier_check_fn: Optional[Callable] = None,
    wider_context: Optional[str] = None,
    require_unanimous: bool = False,
    prior_strength_fn: Optional[Callable] = None,
    min_sentence_scope_hamming_fraction: float = 0.3,
) -> dict:
    """The actual iteration loop, split out of run_ghost_agent so backbone
    construction/reset/close (which differ between the default-Anthropic
    back-compat path and an externally-supplied/named backbone) stay in
    one place, wrapped in a single try/finally."""
    content_profile = tool_analyse_structure(target)

    if prior_strength_fn is not None:
        try:
            prior_strength_result = prior_strength_fn(target, f"{field_name}: ")
            content_profile = {
                **content_profile,
                "prior_strength_ratio": prior_strength_result["mean_ratio"],
                "prior_strength_detail": prior_strength_result,
            }
            if verbose:
                print(
                    "Prior-strength ratio (ensemble mean): "
                    f"{prior_strength_result['mean_ratio']:.4f}"
                )
        except Exception as e:
            if verbose:
                print(f"Prior-strength scoring failed (continuing without it): {e}")

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

    # field_ground_truth decouples "what the agent obfuscates" (target)
    # from "what an extraction must recover to count as success" -- these
    # are the same string for the original bare-field-value scope
    # (field_ground_truth defaults to target below), but must differ for
    # sentence-level scope: the agent obfuscates the whole SENTENCE, while
    # extraction is still graded against just the field's value. Captured
    # BEFORE the None-collapse below so the clean_reference_text branch
    # can tell "caller omitted this" apart from "caller explicitly passed
    # a value equal to target" (an edge case the collapsed value alone
    # can't distinguish).
    is_sentence_scope = field_ground_truth is not None
    field_ground_truth = field_ground_truth or target

    # The clean-floor check needs a plain-text sample containing the
    # SAME target value, unobfuscated -- it's testing "can this member
    # read this value at all", not testing a different dataset value.
    # Sentence-scope: the sentence already reads naturally and contains
    # the field, so no artificial "field_name: " prefix is needed (unlike
    # the bare-field case, where the prefix is what gives the model any
    # field-name context at all).
    clean_reference_text = target if is_sentence_scope else f"{field_name}: {target}"
    clean_reference_value = field_ground_truth

    # effective_target/effective_clean_reference_text are what's ACTUALLY
    # encoded/graded this iteration -- start out equal to target/
    # clean_reference_text, but flip once (never back) if the agent calls
    # widen_scope() and a wider_context was supplied. field_ground_truth/
    # clean_reference_value never change on a widen -- what counts as a
    # successful extraction is always the real field value, regardless of
    # how much surrounding text got obfuscated to protect it.
    effective_target = target
    effective_clean_reference_text = clean_reference_text
    scope_widened = False
    final_success = False

    if verbose:
        print(f"\n{'='*55}")
        print(f"GHOST-Agent: protecting '{target}' ({field_name})")
        print(f"Ensemble members: {[m['name'] for m in ensemble_members]}")
        print(f"Consensus threshold: {consensus_threshold}")
        print(f"Max iterations: {max_iterations}")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Reset the backbone's own message history every iteration --
        # otherwise each iteration's full ReAct tool-call transcript
        # (up to tool_budget_per_iter round-trips) accumulates forever
        # across the whole field run, since backbone.reset() above is
        # only called once before this loop starts. On a sentence-scope
        # target with VS injection this blew past Claude's 200k-token
        # context limit (247k tokens) on iteration 1 of a real Track A
        # pilot run -- the encoded string alone is ~6x the plain
        # sentence length and gets echoed back in every tool result.
        # Safe to reset per-iteration because cross-iteration context
        # already flows through `reflection` below, not through the
        # backbone's own conversation state.
        backbone.reset()

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

        tool_calls = getattr(backbone, "last_tool_calls", [])
        bidi_config, vs_payload = extract_config_from_tool_calls(tool_calls)

        # A widen_scope() call this turn means the agent's OWN
        # <final_encoding> for this turn (if any) targets wider_context,
        # not the original target -- everything downstream (hamming,
        # rendering, ensemble ground truth text, reflection's TARGET
        # label) must switch to match what actually got encoded. Only
        # flips once: a second widen_scope() call this run is a no-op
        # (there's nothing wider configured to escalate to further).
        if (
            wider_context is not None
            and not scope_widened
            and any(name == "widen_scope" for name, _ in tool_calls)
        ):
            scope_widened = True
            effective_target = wider_context
            effective_clean_reference_text = wider_context
            memory.target = wider_context
            best_hamming = 0
            best_encoding = None
            if verbose:
                print("Agent widened scope to the supplied wider context.")

        proposed = extract_proposed_encoding(agent_text, effective_target)

        if proposed is None:
            if verbose:
                print("Could not extract encoding from agent text")
            continue

        rendered = tool_render(proposed)
        stored = tool_get_stored(proposed)
        dist = tool_hamming(stored, effective_target)

        if verbose:
            print(f"Proposed encoding renders as: '{rendered}'")
            print(f"Stored sequence: '{stored}'")
            print(f"Hamming distance: {dist}")

        # Real bugs found 2026-08-19 (a real sentence-scope batch run):
        # nothing here previously checked that the proposed encoding
        # actually covers the FULL target. Two ways that went wrong,
        # both now rejected outright (no Attempt recorded, same
        # silent-retry treatment as `proposed is None` above -- max_
        # iterations bounds how many times this can happen):
        #   1. rendered != effective_target -- either a straight length
        #      mismatch (the agent encoded only a substring, e.g. the
        #      bare field inside a sentence-scope target -- tool_hamming
        #      then returns -1, which never exceeds best_hamming's
        #      initial 0, so this used to silently "succeed" with a
        #      nonsensical hamming=0 once the ensemble failed to read a
        #      short, context-free garbled fragment), OR a same-LENGTH
        #      but content-corrupted render (found independently by
        #      verify_prior_strength_frontier.py's own round-trip check
        #      on a real multi-line tabular document -- root cause not
        #      yet isolated, but rendered != target catches it either way).
        #   2. is_sentence_scope and the displacement is far too small to
        #      be a genuine sentence-scope attempt -- e.g. the agent
        #      copies the ENTIRE surrounding sentence as exact verbatim
        #      plaintext and only obfuscates the field itself. This
        #      passes check #1 (rendered == target, lengths trivially
        #      match) and produces a small but non-zero Hamming distance
        #      that looks superficially like "weak but valid" obfuscation
        #      -- two real cases showed dist=12 against a ~140-char
        #      sentence (8-9%), against 92-100% for every genuine
        #      sentence-scope attempt in the same batch. This is field-
        #      scope wearing sentence-scope's clothing, not a real test
        #      of the mechanism this run/caller asked for.
        if rendered != effective_target:
            if verbose:
                print(
                    f"REJECTED: proposed encoding renders as {len(rendered)} "
                    f"chars, target is {len(effective_target)} chars, or "
                    f"content differs despite matching length -- it does not "
                    f"render back to the exact full target. Retrying."
                )
            continue

        if (
            is_sentence_scope
            and effective_target
            and dist / len(effective_target) < min_sentence_scope_hamming_fraction
        ):
            if verbose:
                print(
                    f"REJECTED: only {dist}/{len(effective_target)} characters "
                    f"({dist / len(effective_target):.1%}) displaced -- below "
                    f"min_sentence_scope_hamming_fraction="
                    f"{min_sentence_scope_hamming_fraction:.0%}. Sentence scope "
                    f"requires obfuscating the surrounding text too, not just "
                    f"the field. Retrying."
                )
            continue

        if dist > best_hamming:
            best_hamming = dist
            best_encoding = proposed

        # A resident local backbone (e.g. deepseek_r1_32b) and the
        # search-tier ensemble's own per-member GPU residency can both
        # need the card at once -- confirmed OOM on a real run when both
        # were held simultaneously. release_gpu()/reacquire_gpu() are
        # no-ops for AnthropicBackbone; see agent_backbone.py's
        # LocalReActBackbone docstring for why this round-trip exists.
        ensemble_kwargs = dict(
            encoded_text=proposed,
            field_name=field_name,
            ground_truth=field_ground_truth,
            members=ensemble_members,
            consensus_threshold=consensus_threshold,
            clean_reference_text=effective_clean_reference_text,
            clean_reference_value=clean_reference_value,
            clean_floor_check=clean_floor_check,
        )
        # Only threaded through when explicitly given (calibration runs) --
        # omitted otherwise so a custom ensemble_query_fn (e.g. a test
        # fake, or a reconstruction-mode swap-in) that doesn't accept this
        # kwarg keeps working unchanged.
        if frontier_check_fn is not None:
            ensemble_kwargs["frontier_check_fn"] = frontier_check_fn

        backbone.release_gpu()
        try:
            ensemble_result = ensemble_query_fn(**ensemble_kwargs)
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
            bidi_config=bidi_config,
            vs_payload=vs_payload,
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
                # Structured, per-attempt fields for later correlation
                # analysis (e.g. "does prior_strength_ratio predict
                # hamming_pct/frontier_outcome") -- see CLAUDE.md's
                # prior-strength scoping discussion. prior_strength_ratio
                # is None when prior_strength_fn wasn't given this run;
                # vs_threshold_tau is not yet captured from tool calls
                # (extract_config_from_tool_calls doesn't record it) --
                # a known gap, not silently pretended away.
                "prior_strength_ratio": content_profile.get("prior_strength_ratio"),
                "hamming_pct": (dist / len(effective_target)) if effective_target else None,
                "vs_threshold_tau": None,
                "frontier_outcome": ensemble_result.get("frontier_check"),
                "adversary_model": [m["name"] for m in ensemble_members],
                "adversary_response": [
                    m.get("response") for m in ensemble_result["per_member"]
                ],
                "extraction_succeeded": not ensemble_result["defended"],
                "iteration": iteration,
                "agent_reasoning": agent_text[:500],
            })

        # is_full_consensus is independent of consensus_threshold -- it's
        # ALL valid members failing, not just enough of them. n_valid/
        # n_failed are computed purely from the local ensemble in
        # ensemble.py (unaffected by the calibration-only frontier gate
        # appending its own per_member row), so this stays meaningful
        # even when frontier_check_fn is set.
        is_full_consensus = (
            ensemble_result["n_valid"] > 0
            and ensemble_result["n_failed"] == ensemble_result["n_valid"]
        )
        # should_stop is the ACTUAL stopping bar for this run --
        # ensemble_result["defended"] alone (consensus_threshold, e.g.
        # 3-of-4) is what the previous version of this loop stopped on
        # unconditionally, which is exactly why the agent converged in 1
        # iteration on essentially every real run tested and never had a
        # reason to retry a seed or call widen_scope(): the bar it was
        # actually optimizing for was trivially cleared on the first
        # attempt almost every time.
        # AND with ensemble_result["defended"] so a calibration-only
        # frontier_check_fn overturn (ensemble.py sets "defended" back to
        # False when the frontier model still extracts, regardless of
        # local unanimity) is never ignored -- is_full_consensus alone
        # only reflects the LOCAL ensemble's n_failed/n_valid and knows
        # nothing about a frontier overturn, so using it bare would let
        # require_unanimous stop on "full local consensus" even after a
        # real frontier model just proved it could still extract. When
        # require_unanimous=False this reduces to exactly
        # ensemble_result["defended"] (unchanged pre-existing behavior),
        # since consensus_threshold <= n_valid makes "defended" already
        # imply is_full_consensus whenever it's actually True.
        should_stop = ensemble_result["defended"] and (
            is_full_consensus if require_unanimous else True
        )

        result_msg = (
            f"Ensemble panel result: {ensemble_result['n_failed']}/"
            f"{ensemble_result['n_valid']} valid members defeated.\n"
            f"Consensus threshold: {consensus_threshold}"
            f"{' (unanimous defeat required)' if require_unanimous else ''}.\n"
            f"{'DEFENDED' if should_stop else 'NOT YET DEFENDED'}."
        )
        if ensemble_result["defended"] and not should_stop:
            result_msg += (
                "\nNote: you cleared the basic consensus threshold but NOT "
                "every valid panel member -- your margin is thin, and a "
                "held-out model outside this panel could still succeed "
                "where these failed narrowly. Before accepting this, retry "
                "combine_permute/combine with a DIFFERENT seed or payload, "
                "or call widen_scope() if you have exhausted that."
            )
        backbone.messages.append({"role": "user", "content": result_msg})

        if should_stop:
            final_success = True
            if verbose:
                print(
                    f"\nSUCCESS: Ensemble "
                    f"{'unanimity' if require_unanimous else 'consensus'} "
                    f"defeated in {iteration} iterations!"
                )
            best_encoding = proposed

            if use_memory:
                final_content_profile = (
                    tool_analyse_structure(effective_target)
                    if scope_widened else content_profile
                )
                try:
                    principle = distil_principle({
                        "content_type": final_content_profile["content_type"],
                        "n_chars": final_content_profile["n_chars"],
                        "n_sentences": final_content_profile["n_sentences"],
                        "has_numbers": final_content_profile["has_numbers"],
                        "agent_reasoning": agent_text[:500],
                        "n_iterations": iteration,
                        "hamming": dist,
                        "total_len": len(effective_target),
                    }, backbone)
                except Exception as e:
                    if verbose:
                        print(f"Principle distillation failed: {e}")
                    principle = None

                if principle is not None and not _is_valid_principle(principle):
                    if verbose:
                        print(
                            f"Distillation produced no usable principle "
                            f"(refusal or too short) -- discarding: {principle!r}"
                        )
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
                        content_profile=final_content_profile,
                        bidi_config=attempt.bidi_config,
                        vs_payload=attempt.vs_payload,
                        n_iterations=iteration,
                        adversary_model=",".join(
                            m["name"] for m in ensemble_members
                        ),
                        # Deliberately content_profile (the ORIGINAL
                        # target's score), not final_content_profile --
                        # prior-strength was only ever computed once, on
                        # `target`, before any widen_scope() call.
                        # Re-scoring the widened target would need
                        # another real ensemble load/unload pass, not
                        # done here; this value should be read as "the
                        # pre-widen field's prior strength" whenever
                        # scope_widened is True, not the sentence's.
                        prior_strength_ratio=content_profile.get("prior_strength_ratio"),
                    )
                    strategy_memory.save_memory(loaded_memory)
                    if verbose:
                        print(f"Distilled principle: {principle}")

            break

    return {
        # final_success reflects should_stop (the require_unanimous-aware
        # bar this run actually stopped on), NOT memory.succeeded --
        # memory.succeeded is True the moment any attempt clears the bare
        # consensus_threshold, which under require_unanimous=True can be
        # true on an iteration where this loop correctly judged the
        # margin too thin and kept going.
        "success": final_success,
        "best_encoding": best_encoding,
        "n_iterations": memory.n_attempts,
        "final_hamming": best_hamming,
        "history": memory.attempts,
        "target": target,
        "field_name": field_name,
        "n_principles_available": n_principles_available,
        "scope_widened": scope_widened,
        "final_target": effective_target,
    }
