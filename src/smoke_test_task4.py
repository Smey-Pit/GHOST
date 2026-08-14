"""
smoke_test_task4.py

Same structure as the original Task 4 smoke test in
GHOST_AGENT_TASKS.md, plus checks that the prompt reflects the
ensemble-panel framing (Task 2/3's deviation from the spec's single-
adversary design) rather than a lone "the adversary."
"""

from ghost_agent_prompt import (
    GHOST_AGENT_SYSTEM_PROMPT,
    build_iteration_prompt,
)

# Check prompt is substantial
assert len(GHOST_AGENT_SYSTEM_PROMPT) > 1000
assert "render" in GHOST_AGENT_SYSTEM_PROMPT
assert "hamming" in GHOST_AGENT_SYSTEM_PROMPT
assert "combine" in GHOST_AGENT_SYSTEM_PROMPT
assert "bidi" in GHOST_AGENT_SYSTEM_PROMPT

# Check it describes the panel/ensemble mechanism, not a lone adversary
assert "panel" in GHOST_AGENT_SYSTEM_PROMPT.lower()
assert "supermajority" in GHOST_AGENT_SYSTEM_PROMPT.lower()
assert "different tokenizers" in GHOST_AGENT_SYSTEM_PROMPT.lower()

# Check it instructs delimiter-based reporting, not a loose text marker
assert "<final_encoding>" in GHOST_AGENT_SYSTEM_PROMPT
assert "</final_encoding>" in GHOST_AGENT_SYSTEM_PROMPT

# Check iteration prompt construction
prompt1 = build_iteration_prompt(
    target="48271039",
    field_name="account_number",
    reflection="No attempts yet.",
    iteration=1,
    budget_remaining=20,
)
assert "48271039" in prompt1
assert "account_number" in prompt1

prompt2 = build_iteration_prompt(
    target="48271039",
    field_name="account_number",
    reflection="Attempt 1 failed...",
    iteration=2,
    budget_remaining=15,
)
assert "ITERATION 2" in prompt2
assert "Attempt 1 failed" in prompt2
assert "panel" in prompt2.lower()

prompt3 = build_iteration_prompt(
    target="48271039",
    field_name="account_number",
    reflection="No attempts yet.",
    iteration=1,
    budget_remaining=20,
)
assert "<final_encoding>" in prompt3

print("TASK 4 SMOKE TEST PASSED")
