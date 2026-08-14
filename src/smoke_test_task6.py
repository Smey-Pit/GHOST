"""
smoke_test_task6.py

Verifies convergence.py's orchestration (sampling, static-baseline
computation, CSV writing, per-domain summary) WITHOUT a GPU or a real
Anthropic API key, by injecting a fake run_ghost_agent_fn. Runs on the
real data/raw/documents.json so the sampling/field-iteration logic is
exercised against real data shape, just with a stubbed agent.
"""

import csv
import os
import shutil

from convergence import run_convergence_analysis

TMP_OUTPUT_DIR = "../results/tables/_smoke_test_task6"


def fake_run_ghost_agent(
    target, field_name, ensemble_members, consensus_threshold,
    clean_floor_check, max_iterations, verbose,
):
    # Deterministic stand-in: "succeeds" for odd-length targets, to get
    # a mix of True/False in the summary without needing real inference.
    success = len(target) % 2 == 1
    return {
        "success": success,
        "best_encoding": f"FAKE_ENCODING({target})",
        "n_iterations": 2 if success else max_iterations,
        "final_hamming": len(target) - 1,
        "history": [],
        "target": target,
        "field_name": field_name,
    }


if os.path.isdir(TMP_OUTPUT_DIR):
    shutil.rmtree(TMP_OUTPUT_DIR)

results = run_convergence_analysis(
    documents_path="../data/raw/documents.json",
    config_path="../config.yaml",
    n_samples=3,
    max_iterations=2,
    output_dir=TMP_OUTPUT_DIR,
    run_ghost_agent_fn=fake_run_ghost_agent,
)

assert len(results) > 0, "No results produced"
assert "agent_success" in results[0]
assert "agent_iterations" in results[0]
assert "static_hamming" in results[0]

# Field count per doc varies by domain (financial has 5, legal has 4,
# etc. -- see data/raw/documents.json) so just check we got a
# plausible multi-field, multi-domain result set, not a fixed count.
assert len(results) >= 3, f"expected at least 1 field per doc, got {len(results)}"
assert len({r["doc_id"] for r in results}) == 3, "expected exactly 3 sampled docs"

for row in results:
    expected_success = len(row["field_value"]) % 2 == 1
    assert row["agent_success"] == expected_success, row

out_path = os.path.join(TMP_OUTPUT_DIR, "agent_convergence.csv")
assert os.path.isfile(out_path), "CSV not written"
with open(out_path) as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)
assert len(csv_rows) == len(results)
assert "agent_success" in csv_rows[0]

shutil.rmtree(TMP_OUTPUT_DIR)

print(f"Smoke test ran {len(results)} field evaluations")
print("TASK 6 SMOKE TEST PASSED")
