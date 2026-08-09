import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_integrated_web_recovery_verifier_has_honest_boundaries(tmp_path):
    output = tmp_path / "web-recovery.json"
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/verify_web_recovery.py"),
            "--out", str(output),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] and all(report["checks"].values())
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["formal_experiment_executed"] is False
    assert any("RPO/RTO" in item for item in report["limitations"])
