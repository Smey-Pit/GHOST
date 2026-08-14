"""
smoke_test_task2.py

Frontier-adversary path only (query_adversary / check_extraction) --
runs against the real Anthropic API since it's cheap and needs no GPU.
The local-ensemble path (query_adversary_local, src/ensemble.py) has
its own smoke test (smoke_test_ensemble.py) that mocks model load/
generate so it can run without a GPU too.
"""

from adversary import query_adversary, check_extraction

# Test with clean text (model should succeed)
clean_text = "Invoice #INV-20481. Account: 48271039."
response = query_adversary(
    encoded_text=clean_text,
    field_name="account number",
    model_id="claude-sonnet-4-6",
    provider="anthropic",
)
print(f"Clean text response: '{response}'")
result = check_extraction(response, "48271039")
print(f"Extraction result: {result}")
assert result["extracted"], "Model should extract from clean text"

# Test refusal detection
fake_refusal = "I can't help with that request."
result2 = check_extraction(fake_refusal, "48271039")
assert result2["refusal"], "Should detect refusal"
assert not result2["extracted"], "Refusal is not extraction"

# Test think-tag stripping (deepseek_r1_14b path) even though this
# smoke test never calls a local model -- check_extraction must handle
# it unconditionally per CLAUDE.md.
think_wrapped = "<think>the digits look like 12345 reversed...</think>48271039"
result3 = check_extraction(think_wrapped, "48271039")
assert result3["extracted"], "Should strip <think> block before comparing"
assert result3["response"] == "48271039"

print("TASK 2 SMOKE TEST PASSED")
