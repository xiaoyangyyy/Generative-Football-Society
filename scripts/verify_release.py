from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gfs-studio" / "src"))
sys.path.insert(0, str(ROOT / "gfs-core" / "src"))

from gfs_core.kernel.project import source_identities, source_identity


def run(command, cwd, *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=cwd, check=False, capture_output=capture, text=True,
    )


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
    with tempfile.TemporaryDirectory(prefix="gfs-release-") as temporary_name:
        temporary = Path(temporary_name)
        for name, project in (("core", core), ("studio", studio)):
            checks[f"{name}_tests"] = run(
                [sys.executable, "-m", "pytest", "tests", "-q"], project,
            ).returncode == 0
            checks[f"{name}_ruff"] = run([
                "uvx", "--from", "ruff==0.15.17", "ruff", "check",
                "src", "tests", "--output-format", "concise",
            ], project).returncode == 0
            checks[f"{name}_wheel"] = run([
                "uv", "-q", "build", "--wheel", ".",
                "--out-dir", temporary,
            ], project).returncode == 0
            commands[name] = {
                key: checks[f"{name}_{key}"]
                for key in ("tests", "ruff", "wheel")
            }
        wheels = sorted(temporary.glob("*.whl"))
        checks["two_wheels_built"] = len(wheels) == 2
        venv = temporary / "installed"
        checks["clean_venv_created"] = run([
            sys.executable, "-m", "venv", str(venv),
        ], temporary).returncode == 0
        installed_python = (
            venv / "Scripts" / "python.exe" if os.name == "nt"
            else venv / "bin" / "python"
        )
        checks["clean_wheel_install"] = checks["two_wheels_built"] and run([
            str(installed_python), "-m", "pip", "install",
            *(str(wheel) for wheel in wheels),
        ], temporary).returncode == 0
        checks["installed_core_micro_match"] = checks["clean_wheel_install"] and run([
            str(installed_python), "-c",
            (
                "from gfs_core.bootstrap.application import CoreApplication; "
                f"result=CoreApplication({str(core)!r}).run_micro_match(fast=True); "
                "assert result.ticks > 0; print('installed-core-micro-ok')"
            ),
        ], temporary).returncode == 0
        checks["installed_studio_release_route"] = checks["clean_wheel_install"] and run([
            str(installed_python), "-m", "gfs_studio", "--base-dir", str(studio),
            "release", "status",
        ], temporary).returncode == 0
        supply = run([
            sys.executable, str(root / "scripts" / "build_plugin_supply_chain.py"),
        ], root, capture=True)
        try:
            supply_report = json.loads(supply.stdout)
        except ValueError:
            supply_report = {"passed": False, "error": "invalid verifier output"}
        checks["plugin_supply_chain"] = (
            supply.returncode == 0 and supply_report.get("passed") is True
        )
        deployment_process = run([
            sys.executable, str(root / "scripts" / "verify_deployment.py"),
        ], root, capture=True)
        try:
            deployment = json.loads(deployment_process.stdout)
        except ValueError:
            deployment = {"static_verified": False, "errors": ["invalid verifier output"]}
        checks["static_deployment"] = (
            deployment_process.returncode == 0
            and deployment.get("static_verified") is True
        )
    code_ready = all(checks.values())
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_identity": source_identity(root),
        "source_identities": source_identities(root),
        "code_ready": code_ready,
        "checks": checks,
        "commands": commands,
        "supply_chain": supply_report,
        "deployment": deployment,
        "scope": "current_source_build_clean_wheel_install_not_external_research_evidence",
    }
    destination = args.out or studio / "evidence" / "current" / "build_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary_path.write_text(encoded, encoding="utf-8", newline="\n")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(encoded, end="")
    return 0 if code_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
