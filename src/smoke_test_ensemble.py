"""
smoke_test_ensemble.py

Tests src/ensemble.py's config resolution and consensus-voting logic
WITHOUT a GPU or any downloaded model weights, by injecting fake
loader/unloader/querier functions. This is deliberately separate from
smoke_test_task2.py (which hits the real Anthropic API) so the voting
logic can be verified now and re-run unchanged once real weights are
available -- only the injected functions change, the assertions don't.
"""

import yaml
from ensemble import load_ensemble_config, run_ensemble_query

CONFIG_PATH = "../config.yaml"


def fake_loader(hf_id, dtype):
    return (hf_id, hf_id)  # no real tokenizer/model needed


def fake_unloader(tokenizer, model):
    pass


# ── Test 1: config resolution reads agent_ensemble + local_models ──────

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

resolved = load_ensemble_config(config)
names = [m["name"] for m in resolved["members"]]
assert names == ["llama31_8b", "mistral7b", "deepseek_r1_14b", "deepseek_llm_7b"], names
assert resolved["consensus_threshold"] == 3
assert resolved["clean_floor_check"] is True
for m in resolved["members"]:
    assert "hf_id" in m and "max_new_tokens" in m
print("Test 1 passed: config resolution")


# ── Test 2: 3-of-4 failures crosses threshold=3 -> defended ─────────────

def querier_3_fail_1_succeed(tokenizer, model, text, field_name, max_new_tokens):
    # Every member passes the clean floor check (echoes the clean value),
    # then on the "real" query only deepseek_llm_7b gets it right.
    if text == "CLEAN_TEXT":
        return "48271039"
    if model == "deepseek-ai/deepseek-llm-7b-chat":
        return "48271039"
    return "wrong_guess"


result = run_ensemble_query(
    encoded_text="ENCODED_TEXT",
    field_name="account_number",
    ground_truth="48271039",
    members=resolved["members"],
    consensus_threshold=resolved["consensus_threshold"],
    clean_reference_text="CLEAN_TEXT",
    clean_reference_value="48271039",
    clean_floor_check=True,
    loader=fake_loader,
    unloader=fake_unloader,
    querier=querier_3_fail_1_succeed,
)
assert result["n_valid"] == 4, result
assert result["n_failed"] == 3, result
assert result["defended"] is True, result
print("Test 2 passed: 3-of-4 failure crosses threshold=3")


# ── Test 3: only 2-of-4 fail -> NOT defended ─────────────────────────────

def querier_2_fail_2_succeed(tokenizer, model, text, field_name, max_new_tokens):
    if text == "CLEAN_TEXT":
        return "48271039"
    if model in ("deepseek-ai/deepseek-llm-7b-chat", "mistralai/Mistral-7B-Instruct-v0.3"):
        return "48271039"
    return "wrong_guess"


result2 = run_ensemble_query(
    encoded_text="ENCODED_TEXT",
    field_name="account_number",
    ground_truth="48271039",
    members=resolved["members"],
    consensus_threshold=resolved["consensus_threshold"],
    clean_reference_text="CLEAN_TEXT",
    clean_reference_value="48271039",
    clean_floor_check=True,
    loader=fake_loader,
    unloader=fake_unloader,
    querier=querier_2_fail_2_succeed,
)
assert result2["n_valid"] == 4, result2
assert result2["n_failed"] == 2, result2
assert result2["defended"] is False, result2
print("Test 3 passed: 2-of-4 failure does NOT cross threshold=3")


# ── Test 4: clean-floor check excludes a broken member's vote ───────────

def querier_one_member_cant_read_clean(tokenizer, model, text, field_name, max_new_tokens):
    # llama31_8b fails even on CLEAN text -- its vote must be excluded,
    # even though it also "fails" on the encoded text.
    if model == "meta-llama/Llama-3.1-8B-Instruct":
        return "garbage" if text == "CLEAN_TEXT" else "wrong_guess"
    if text == "CLEAN_TEXT":
        return "48271039"
    return "wrong_guess"  # remaining 3 all defeat extraction on encoded text


result3 = run_ensemble_query(
    encoded_text="ENCODED_TEXT",
    field_name="account_number",
    ground_truth="48271039",
    members=resolved["members"],
    consensus_threshold=resolved["consensus_threshold"],
    clean_reference_text="CLEAN_TEXT",
    clean_reference_value="48271039",
    clean_floor_check=True,
    loader=fake_loader,
    unloader=fake_unloader,
    querier=querier_one_member_cant_read_clean,
)
assert result3["n_valid"] == 3, result3  # llama31_8b excluded
assert result3["n_failed"] == 3, result3
assert result3["defended"] is True, result3
broken = [m for m in result3["per_member"] if m["name"] == "llama31_8b"][0]
assert broken["valid"] is False
print("Test 4 passed: clean-floor check excludes a broken member's vote")


# ── Test 5: refusal counts as "not extracted" for voting purposes ───────

def querier_refusal(tokenizer, model, text, field_name, max_new_tokens):
    if text == "CLEAN_TEXT":
        return "48271039"
    return "I cannot determine this value."


result4 = run_ensemble_query(
    encoded_text="ENCODED_TEXT",
    field_name="account_number",
    ground_truth="48271039",
    members=resolved["members"],
    consensus_threshold=resolved["consensus_threshold"],
    clean_reference_text="CLEAN_TEXT",
    clean_reference_value="48271039",
    clean_floor_check=True,
    loader=fake_loader,
    unloader=fake_unloader,
    querier=querier_refusal,
)
assert result4["n_failed"] == 4, result4
assert all(m["refusal"] for m in result4["per_member"]), result4
assert result4["defended"] is True, result4
print("Test 5 passed: refusal counts toward defended, same as extraction failure")

print("ENSEMBLE SMOKE TEST PASSED")
