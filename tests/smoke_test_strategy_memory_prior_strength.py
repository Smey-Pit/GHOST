"""
tests/smoke_test_strategy_memory_prior_strength.py

Verifies add_principle's new prior_strength_ratio field WITHOUT touching
any real file (add_principle operates on an in-memory dict the caller
owns -- load_memory/save_memory are never called here). No GPU/API.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import strategy_memory  # noqa: E402

CONTENT_PROFILE = {
    "content_type": "numeric",
    "n_chars": 10,
    "n_sentences": 1,
    "has_numbers": True,
    "complexity": "low",
}

memory = {"version": 1, "n_documents_processed": 0, "principles": []}

# -- Test 1: creation stores prior_strength_ratio --
memory = strategy_memory.add_principle(
    memory=memory,
    principle="Full RTL reversal + VS injection defeats numeric fields.",
    content_profile=CONTENT_PROFILE,
    bidi_config="full_rtl",
    vs_payload="ghost",
    n_iterations=1,
    adversary_model="llama31_8b,mistral7b,deepseek_r1_14b,deepseek_llm_7b",
    prior_strength_ratio=0.92,
)
assert len(memory["principles"]) == 1, memory
assert memory["principles"][0]["prior_strength_ratio"] == 0.92, memory
print("Test 1 passed: creation stores prior_strength_ratio")

# -- Test 2: strengthening an existing matching principle does NOT
# overwrite the original prior_strength_ratio with the new call's value
# (per add_principle's docstring: set once at creation, not averaged) --
memory = strategy_memory.add_principle(
    memory=memory,
    principle="Full RTL reversal + VS injection defeats numeric fields.",
    content_profile=CONTENT_PROFILE,
    bidi_config="full_rtl",
    vs_payload="ghost",
    n_iterations=1,
    adversary_model="llama31_8b,mistral7b,deepseek_r1_14b,deepseek_llm_7b",
    prior_strength_ratio=0.05,  # very different value -- must be ignored
)
assert len(memory["principles"]) == 1, "should strengthen, not duplicate"
assert memory["principles"][0]["prior_strength_ratio"] == 0.92, (
    "strengthening an existing principle must not overwrite its "
    "original prior_strength_ratio"
)
assert memory["principles"][0]["supporting_evidence"] == 2, memory
print("Test 2 passed: strengthen path leaves prior_strength_ratio untouched")

# -- Test 3: omitting prior_strength_ratio (every pre-existing call site)
# defaults to None, not a crash or a fabricated value --
memory2 = {"version": 1, "n_documents_processed": 0, "principles": []}
memory2 = strategy_memory.add_principle(
    memory=memory2,
    principle="Sentence-scope obfuscation for long text.",
    content_profile={
        "content_type": "multi_sentence", "n_chars": 150,
        "n_sentences": 3, "has_numbers": False, "complexity": "high",
    },
    bidi_config="bidi_permute(trials=2000,seed=0)",
    vs_payload="ghost",
    n_iterations=1,
    adversary_model="llama31_8b",
)
assert memory2["principles"][0]["prior_strength_ratio"] is None, memory2
print("Test 3 passed: omitted prior_strength_ratio defaults to None")

print("STRATEGY_MEMORY PRIOR-STRENGTH SMOKE TEST PASSED")
