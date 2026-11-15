import json
import os
from pathlib import Path
import subprocess
import sys

from gfs_core.bootstrap.application import CoreApplication

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_status_cli_returns_machine_readable_ranking():
    output = CoreApplication(PROJECT_ROOT).run_cli([
        "status", "--top", "3", "--json",
    ])
    rows = json.loads(output)
    assert [row["team"] for row in rows] == ["Argentina", "Brazil", "France"]
    assert len(rows) == 3


def test_monte_carlo_module_entrypoint_handles_all_team_names():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gfs_core",
            "--base-dir",
            str(PROJECT_ROOT),
            "monte-carlo",
            "-n",
            "1",
            "--seed",
            "42",
            "--json",
        ],
        cwd=PROJECT_ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    rows = json.loads(completed.stdout)
    assert len(rows) == 48
    assert rows[0]["team"] == "Brazil"
    assert any(row["team"] == "Curaçao" for row in rows)
