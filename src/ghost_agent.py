"""
src/ghost_agent.py

GHOST self-improving agent (Task 5 of GHOST_AGENT_TASKS.md).

The agent loop:
  1. Build iteration prompt with reflection history
  2. Call the agent model (claude-sonnet-4-6, per config.yaml's
     api_models.claude_sonnet -- fixed per CRITICAL RULE 5 in the task
     doc) with tools
  3. Execute the tool calls the agent makes (ghost_tools.py)
  4. Extract the agent's proposed encoding
  5. Query the ensemble panel (src/ensemble.py) -- NOT a single
     adversary, see reflection.py/ghost_agent_prompt.py for why
  6. Check the consensus verdict
  7. Update reflection memory
  8. Repeat or stop

The agent (claude-sonnet-4-6) reasons and proposes. The ensemble panel
(configured at runtime via config.yaml's agent_ensemble, resolved by
src/ensemble.py's load_ensemble_config) attempts extraction. These are
always separate models with separate roles -- the panel is never
hardcoded here, it's threaded through as ensemble_members/
consensus_threshold arguments.
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


AGENT_MODEL = "claude-sonnet-4-6"  # fixed per CRITICAL RULE 5; the panel
                                    # being tested against is the thing
                                    # that's configurable, never this


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


def distil_principle(success_data: dict, client=None) -> str:
    """
    Call the agent model to distil an abstract, generalisable principle
    from one successful encoding attempt.

    `client` defaults to a real anthropic.Anthropic() but should be the
    SAME client instance run_ghost_agent is already using (passed
    through, not a fresh one) -- both because a second real client
    means a second set of real API credentials/costs to reason about,
    and because tests that inject a fake client need every model call
    in a run to go through that one fake, not silently open a live
    connection for this specific step.

    There is no structured bidi_config/vs_payload to cite here (the
    agent emits one already-combined <final_encoding> string, not
    separate config/payload fields -- see Attempt's "from_agent"
    placeholders in run_ghost_agent) -- so this prompts on the agent's
    own free-text reasoning instead, which is the only place that
    information actually lives.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": DISTILLATION_PROMPT.format(**success_data),
        }],
    )
    return response.content[0].text.strip()


# ── Anthropic tool definitions ────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "render",
        "description": (
            "Render a bidi-encoded string to see what a human "
            "reader sees. Always call this to verify your "
            "encoding before reporting it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "encoding": {
                    "type": "string",
                    "description": "The encoded string to render",
                },
            },
            "required": ["encoding"],
        },
    },
    {
        "name": "get_stored",
        "description": (
            "Get the stored character sequence from an encoded "
            "string. This is what an LLM tokenizer ingests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "encoding": {
                    "type": "string",
                    "description": "The encoded string",
                },
            },
            "required": ["encoding"],
        },
    },
    {
        "name": "hamming",
        "description": (
            "Compute Hamming distance between two strings. "
            "Higher = more different = better obfuscation. "
            "Use this to evaluate your encoding quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "s1": {
                    "type": "string",
                    "description": "First string (stored sequence)",
                },
                "s2": {
                    "type": "string",
                    "description": "Second string (original target)",
                },
            },
            "required": ["s1", "s2"],
        },
    },
    {
        "name": "encode_bidi",
        "description": (
            "Apply a bidi encoding configuration to text. "
            "Available configs: full_rtl, rli_split:N, "
            "rli_split_v2:N, three_block:N:M, nested:N:M, "
            "double_rli:N"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to encode",
                },
                "config": {
                    "type": "string",
                    "description": (
                        "Config string e.g. 'full_rtl' or "
                        "'rli_split:3' or 'three_block:2:5'"
                    ),
                },
            },
            "required": ["text", "config"],
        },
    },
    {
        "name": "encode_vs",
        "description": (
            "Inject invisible Variation Selector characters "
            "into text. Fragments LLM tokenization without "
            "affecting human readability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to inject VS into",
                },
                "payload": {
                    "type": "string",
                    "description": (
                        "Payload string determining VS codepoints. "
                        "Try 'ghost', 'ghost_v2', 'aaaaaa', etc."
                    ),
                },
            },
            "required": ["text", "payload"],
        },
    },
    {
        "name": "combine",
        "description": (
            "Apply BOTH bidi encoding and VS injection. "
            "This is the strongest obfuscation -- use this "
            "as your primary tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to encode",
                },
                "bidi_config": {
                    "type": "string",
                    "description": "Bidi configuration string",
                },
                "vs_payload": {
                    "type": "string",
                    "description": "VS payload string",
                },
            },
            "required": ["text", "bidi_config", "vs_payload"],
        },
    },
]


# ── Agent execution ───────────────────────────────────────────

def execute_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a single tool call and return string result."""
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return f"ERROR: unknown tool {tool_name}"
    try:
        result = fn(**tool_input)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


def run_agent_turn(
    client, messages: list, system_prompt: str, max_tool_rounds: int = 8,
    response_max_tokens: int = 2000,
) -> tuple:
    """
    Run one agent turn: call the agent model with tools, execute all
    tool calls, feed back the results, and repeat until the model
    stops calling tools (stop_reason != "tool_use") or max_tool_rounds
    is hit.

    response_max_tokens=2000 was calibrated against short numeric
    targets (10-12 chars). A VS-injected encoding is roughly
    len(target) * (VS chars per char + 1), so a ~125-char natural-
    language target produces an encoding + commentary that can blow
    through 2000 tokens mid-string -- the model gets cut off before
    writing the closing </final_encoding> tag, and
    extract_proposed_encoding then (correctly) reports no tag found.
    Confirmed against the real API: a 20-tool-round run on a lyric
    sentence reached the tag but was truncated inside the encoding at
    exactly 2000 tokens. Callers with longer targets should raise this.

    Every `tool_use` block MUST be answered by a `tool_result` block in
    the very next message -- the Anthropic API rejects a subsequent
    call otherwise ("tool_use ids were found without tool_result
    blocks"). A fixed one-continuation-round version of this function
    silently left a second round of tool_use blocks unanswered
    whenever the model chose to call tools again after its first
    round; that corrupted `messages` for the rest of the run, only
    surfacing as a 400 error on the NEXT api call (confirmed against
    the real API, not just the mocked smoke tests).

    `client` only needs a `.messages.create(...)` method matching the
    Anthropic SDK's interface -- this is a deliberate seam so tests can
    pass a fake client without a real API key.

    Returns:
        (updated_messages, agent_final_text)
    """
    final_text = ""

    for _ in range(max_tool_rounds):
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=response_max_tokens,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "text":
                final_text = block.text
            elif block.type == "tool_use":
                result = execute_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})

    return messages, final_text


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
            see run_agent_turn's response_max_tokens docstring; raise
            for targets much longer than a short numeric field
        verbose: print progress
        client: an Anthropic-SDK-shaped client (`.messages.create`).
            Defaults to a real `anthropic.Anthropic()` -- overridable
            for tests so this loop is checkable without an API key.
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
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

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
    messages = []
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
        messages.append({"role": "user", "content": user_msg})

        messages, agent_text = run_agent_turn(
            client, messages, system_prompt,
            max_tool_rounds=tool_budget_per_iter,
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
        messages.append({"role": "user", "content": result_msg})

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
                    }, client=client)
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
