import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_operations_verifier_is_explicitly_not_a_match_soak(tmp_path):
    output = tmp_path / "operations.json"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/verify_product_operations.py"),
            "--tasks", "20", "--out", str(output),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"]
    assert report["task_count"] == 20
    assert report["submission_attempts"] == 40
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["claim_boundary"]["closes_100_match_soak_gate"] is False
    assert all(report["checks"].values())
