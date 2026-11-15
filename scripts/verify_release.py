from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gfs-studio" / "src"))
sys.path.insert(0, str(ROOT / "gfs-core" / "src"))

from gfs_studio.plugins.evidence.release.current import source_identity


def run(command, cwd) -> int:
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    core = root / "gfs-core"
    studio = root / "gfs-studio"
    checks = {}
    commands = {}
    with tempfile.TemporaryDirectory(prefix="gfs-release-") as temporary:
        for name, project in (("core", core), ("studio", studio)):
            checks[f"{name}_tests"] = run(
                [sys.executable, "-m", "pytest", "tests", "-q"], project,
            ) == 0
            checks[f"{name}_ruff"] = run([
                "uvx", "--from", "ruff==0.15.17", "ruff", "check",
                "src", "tests", "--output-format", "concise",
            ], project) == 0
            checks[f"{name}_wheel"] = run([
                "uv", "-q", "build", "--offline", "--wheel", ".",
                "--out-dir", temporary,
            ], project) == 0
            commands[name] = {
                key: checks[f"{name}_{key}"]
                for key in ("tests", "ruff", "wheel")
            }
    deployment_process = subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_deployment.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    deployment = json.loads(deployment_process.stdout)
    checks["static_deployment"] = (
        deployment_process.returncode == 0 and deployment["static_verified"]
    )
    code_ready = all(checks.values())
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity(root),
        "code_ready": code_ready,
        "checks": checks,
        "commands": commands,
        "deployment": deployment,
        "scope": "current_source_build_not_independent_reproduction_or_participant_evidence",
    }
    destination = args.out or studio / "evidence" / "current" / "build_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary_path.write_text(encoded, encoding="utf-8")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(encoded, end="")
    return 0 if code_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
