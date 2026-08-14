"""
smoke_test_task3.py

Adapted from the original Task 3 smoke test in GHOST_AGENT_TASKS.md to
use ensemble_result (src/ensemble.py's run_ensemble_query output)
instead of a single adversary_response -- see reflection.py's module
docstring for why.
"""

from reflection import ReflectionMemory, Attempt

mem = ReflectionMemory(target="48271039", field_name="account_number")
assert mem.n_attempts == 0
assert not mem.succeeded
assert mem.best_attempt is None

# Attempt 1: weak encoding, ensemble mostly extracts it (1/4 defeated,
# below the consensus_threshold of 3 from config.yaml)
ensemble_result_1 = {
    "defended": False,
    "n_valid": 4,
    "n_failed": 1,
    "per_member": [
        {"name": "llama31_8b", "valid": True, "refusal": False, "extracted": True, "response": "93017284"},
        {"name": "mistral7b", "valid": True, "refusal": False, "extracted": True, "response": "93017284"},
        {"name": "deepseek_r1_14b", "valid": True, "refusal": False, "extracted": True, "response": "93017284"},
        {"name": "deepseek_llm_7b", "valid": True, "refusal": False, "extracted": False, "response": "garbled"},
    ],
}

att1 = Attempt(
    iteration=1,
    bidi_config="full_rtl",
    vs_payload="ghost",
    encoding_description="simple full reversal",
    encoded_text="placeholder",
    rendered="48271039",
    stored="93017284",
    hamming_dist=7,
    ensemble_result=ensemble_result_1,
    agent_reasoning="tried simple reversal, most of the ensemble read the stored order",
)
mem.add_attempt(att1)

assert mem.n_attempts == 1
assert not mem.succeeded  # only 1/4 defeated, below threshold
assert att1.n_failed == 1
assert set(att1.members_that_extracted) == {"llama31_8b", "mistral7b", "deepseek_r1_14b"}

formatted = mem.format_for_reflection()
assert "48271039" in formatted
assert "full_rtl" in formatted
assert "llama31_8b" in formatted
assert "1/4 valid members defeated" in formatted
assert "llama31_8b" in formatted.split("INSIGHT")[-1]  # repeat offenders named
print(formatted)

# Attempt 2: stronger encoding, 3/4 defeated -- crosses threshold
ensemble_result_2 = {
    "defended": True,
    "n_valid": 4,
    "n_failed": 3,
    "per_member": [
        {"name": "llama31_8b", "valid": True, "refusal": False, "extracted": False, "response": "garbled"},
        {"name": "mistral7b", "valid": True, "refusal": True, "extracted": False, "response": "I cannot determine this"},
        {"name": "deepseek_r1_14b", "valid": True, "refusal": False, "extracted": False, "response": "wrong"},
        {"name": "deepseek_llm_7b", "valid": True, "refusal": False, "extracted": True, "response": "48271039"},
    ],
}

att2 = Attempt(
    iteration=2,
    bidi_config="rli_split:4",
    vs_payload="ghost_v2",
    encoding_description="RLI split with stronger VS",
    encoded_text="placeholder2",
    rendered="48271039",
    stored="03927184",
    hamming_dist=8,
    ensemble_result=ensemble_result_2,
    agent_reasoning="tried rli split with longer VS payload",
)
mem.add_attempt(att2)

assert mem.n_attempts == 2
assert mem.succeeded  # 3/4 defeated, crosses threshold=3
assert mem.best_attempt is att2  # higher n_failed than att1
assert att2.extraction_defeated

formatted2 = mem.format_for_reflection()
assert "NOTE: 1 attempt(s) already crossed" in formatted2
print(formatted2)

print("TASK 3 SMOKE TEST PASSED")
