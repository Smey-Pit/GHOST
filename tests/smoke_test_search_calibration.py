"""
Smoke test for the frontier calibration-gate mechanism added to
src/ensemble.py, src/adversary.py, and src/calibrate_agent_search.py.

No GPU, no real API calls -- everything here is mocked/faked. Verifies:
  1. ensemble.run_ensemble_query's frontier_check_fn gate: only called
     when the local ensemble already says defended; flips defended to
     False if the frontier check extracts, leaves it True otherwise;
     appends a clearly-labeled row to per_member.
  2. adversary.make_frontier_check_fn builds a callable that calls
     query_adversary with the right model_id/provider from a config
     dict, and check_extraction on the result.
  3. calibrate_agent_search.run_calibration's comparison/CSV logic works
     against a fake calibration.json and a fake run_ghost_agent_fn (so
     no real agent loop, ensemble, or backbone is exercised here --
     that's already covered by smoke_test_task5.py /
     smoke_test_agent_backbone.py).
"""

import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ensemble  # noqa: E402
import adversary  # noqa: E402
import calibrate_agent_search  # noqa: E402


# ── 1. run_ensemble_query's frontier_check_fn gate ──────────────────────

def _fake_loader(hf_id, dtype="bfloat16"):
    return None, None


def _fake_unloader(tokenizer, model):
    pass


def _fake_querier(tokenizer, model, text, field_name, max_new_tokens):
    # Floor check queries clean_reference_text ("account_number: 12345")
    # and must pass; the real encoded-text query ("encoded") must fail --
    # distinguishing on the input text is what lets one fake querier
    # drive both checks correctly through the real check_extraction.
    if "12345" in text:
        return "12345"
    return "some wrong response"


_fake_checker_all_fail = adversary.check_extraction


members = [
    {"name": f"member_{i}", "hf_id": f"fake/{i}", "max_new_tokens": 10}
    for i in range(4)
]

# All 4 local members fail -> local tier says defended (consensus 3).
result_no_gate = ensemble.run_ensemble_query(
    encoded_text="encoded", field_name="account_number", ground_truth="12345",
    members=members, consensus_threshold=3,
    clean_reference_text="account_number: 12345", clean_reference_value="12345",
    clean_floor_check=True,
    loader=_fake_loader, unloader=_fake_unloader,
    querier=_fake_querier, checker=_fake_checker_all_fail,
)
assert result_no_gate["defended"] is True
assert "frontier_check" not in result_no_gate
print("run_ensemble_query: no frontier_check_fn given -> unaffected OK")

frontier_calls = []


def _fake_frontier_extracts(encoded_text, field_name, ground_truth):
    frontier_calls.append((encoded_text, field_name, ground_truth))
    return {"name": "fake_frontier", "extracted": True, "refusal": False, "response": ground_truth}


result_gated_overturned = ensemble.run_ensemble_query(
    encoded_text="encoded", field_name="account_number", ground_truth="12345",
    members=members, consensus_threshold=3,
    clean_reference_text="account_number: 12345", clean_reference_value="12345",
    clean_floor_check=True,
    loader=_fake_loader, unloader=_fake_unloader,
    querier=_fake_querier, checker=_fake_checker_all_fail,
    frontier_check_fn=_fake_frontier_extracts,
)
assert result_gated_overturned["defended"] is False, "frontier extraction must overturn local 'defended'"
assert len(frontier_calls) == 1
assert any("CALIBRATION-ONLY" in m["name"] for m in result_gated_overturned["per_member"])
print("run_ensemble_query: frontier_check_fn extracts -> overturns defended=False OK")


def _fake_frontier_fails(encoded_text, field_name, ground_truth):
    return {"name": "fake_frontier", "extracted": False, "refusal": False, "response": "wrong"}


result_gated_confirmed = ensemble.run_ensemble_query(
    encoded_text="encoded", field_name="account_number", ground_truth="12345",
    members=members, consensus_threshold=3,
    clean_reference_text="account_number: 12345", clean_reference_value="12345",
    clean_floor_check=True,
    loader=_fake_loader, unloader=_fake_unloader,
    querier=_fake_querier, checker=_fake_checker_all_fail,
    frontier_check_fn=_fake_frontier_fails,
)
assert result_gated_confirmed["defended"] is True
print("run_ensemble_query: frontier_check_fn also fails -> stays defended=True OK")


calls_before = len(frontier_calls)
result_local_not_defended = ensemble.run_ensemble_query(
    # Only 2 members and threshold=3 -- local tier can never reach
    # 'defended' regardless of per-member outcome, so frontier_check_fn
    # must not even be called (cost control).
    encoded_text="encoded", field_name="account_number", ground_truth="12345",
    members=members[:2], consensus_threshold=3,
    clean_reference_text="account_number: 12345", clean_reference_value="12345",
    clean_floor_check=True,
    loader=_fake_loader, unloader=_fake_unloader,
    querier=_fake_querier, checker=_fake_checker_all_fail,
    frontier_check_fn=_fake_frontier_extracts,
)
assert result_local_not_defended["defended"] is False
assert len(frontier_calls) == calls_before, "frontier_check_fn must not be called when local tier isn't defended"
print("run_ensemble_query: local tier not defended -> frontier_check_fn skipped (cost control) OK")


# ── 2. adversary.make_frontier_check_fn ─────────────────────────────────

fake_config = {
    "api_models": {
        "gpt56_sol": {"provider": "openai", "model_id": "gpt-5.6-sol", "max_tokens": 2000},
    }
}

_query_calls = []


def _fake_query_adversary(encoded_text, field_name, model_id, provider, max_tokens):
    _query_calls.append((model_id, provider, max_tokens))
    return "12345"


_orig_query_adversary = adversary.query_adversary
adversary.query_adversary = _fake_query_adversary
try:
    check_fn = adversary.make_frontier_check_fn("gpt56_sol", fake_config)
    out = check_fn("some encoded text", "account_number", "12345")
finally:
    adversary.query_adversary = _orig_query_adversary

assert _query_calls == [("gpt-5.6-sol", "openai", 2000)]
assert out == {"name": "gpt56_sol", "extracted": True, "refusal": False, "response": "12345"}
print("adversary.make_frontier_check_fn: resolves config + calls query_adversary/check_extraction OK")


# ── 3. calibrate_agent_search.run_calibration ───────────────────────────

fake_calibration_docs = [
    {
        "id": "cal_test_001", "domain": "financial",
        "fields": {"account_number": "12345", "reference_number": "REF-1"},
    },
    {
        "id": "cal_test_002", "domain": "medical",
        "fields": {"patient_id": "PID-9"},
    },
]

with tempfile.TemporaryDirectory() as tmpdir:
    calibration_path = os.path.join(tmpdir, "calibration.json")
    with open(calibration_path, "w") as f:
        json.dump(fake_calibration_docs, f)

    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(
            "agent_ensemble:\n"
            "  consensus_threshold: 3\n"
            "  clean_floor_check: true\n"
            "  members: []\n"
            "local_models: {}\n"
            "api_models:\n"
            "  gpt56_sol:\n"
            "    provider: openai\n"
            "    model_id: gpt-5.6-sol\n"
            "    max_tokens: 2000\n"
        )

    output_dir = os.path.join(tmpdir, "tables")

    call_log = []

    def _fake_run_ghost_agent_fn(target, field_name, frontier_check_fn=None, **kwargs):
        # Baseline call (no frontier_check_fn) always "succeeds" in 1
        # iteration; gated call (frontier_check_fn given) needs 3 to
        # simulate the frontier gate forcing extra iterations.
        call_log.append((target, field_name, frontier_check_fn is not None))
        if frontier_check_fn is None:
            return {"success": True, "n_iterations": 1, "final_hamming": 4}
        return {"success": True, "n_iterations": 3, "final_hamming": 5}

    results = calibrate_agent_search.run_calibration(
        calibration_path=calibration_path,
        config_path=config_path,
        output_dir=output_dir,
        n_fields=3,
        max_iterations=5,
        frontier_model_keys=["gpt56_sol"],
        run_ghost_agent_fn=_fake_run_ghost_agent_fn,
    )

    assert len(results) == 3, f"expected 3 fields (n_fields cap), got {len(results)}"
    assert [r["field_name"] for r in results] == [
        "account_number", "reference_number", "patient_id",
    ], "must respect n_fields cap across documents, not per-document"
    assert all(r["iterations_delta"] == 2 for r in results)
    assert all(r["hamming_delta"] == 1 for r in results)

    # Each field should have triggered exactly one baseline call (no
    # frontier_check_fn) and one gated call (frontier_check_fn given).
    assert sum(1 for _, _, gated in call_log if not gated) == 3
    assert sum(1 for _, _, gated in call_log if gated) == 3

    out_csv = os.path.join(output_dir, "agent_search_calibration.csv")
    assert os.path.exists(out_csv)
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

print("calibrate_agent_search.run_calibration: n_fields cap, delta computation, CSV write OK")

print("SMOKE TEST search_calibration PASSED")
