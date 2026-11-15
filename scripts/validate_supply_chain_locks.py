#!/usr/bin/env python3
"""Validate both hashed lockfiles with real uv and pip dry runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_reproducibility_matrix import PROFILES
from src.infrastructure import file_sha256
from src.infrastructure.supply_chain import (
    canonical_package_name,
    parse_hashed_lock,
)

EXCLUDE_NEWER = "2026-08-10T00:00:00Z"
LINUX_PLATFORMS = (
    "manylinux_2_28_x86_64",
    "manylinux_2_27_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _render(command: list[str]) -> str:
    executable = Path(command[0])
    first = (
        "uv" if executable.stem.casefold() == "uv"
        else "python" if executable.resolve() == Path(sys.executable).resolve()
        else command[0]
    )
    return " ".join([first, *command[1:]])


def _planned_packages(output: str) -> set[str]:
    for line in reversed(output.splitlines()):
        if not line.startswith("Would install "):
            continue
        return {
            canonical_package_name(item.rsplit("-", 1)[0])
            for item in line.removeprefix("Would install ").split()
        }
    return set()


def _diagnostic(result: subprocess.CompletedProcess[str]) -> str | None:
    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout).strip()[-4000:] or None


def _commands(profile: dict[str, Any], uv: str) -> tuple[list[str], list[str], list[str]]:
    lock = str(profile["lock"])
    input_file = str(profile["input"])
    python = str(profile["python"])
    platform = str(profile["platform"])
    resolve = [
        uv, "pip", "compile", input_file, "--output-file", lock,
        "--python-version", python, "--python-platform", platform,
        "--torch-backend", "cpu", "--only-binary", ":all:", "--generate-hashes",
        "--no-annotate", "--emit-index-url", "--emit-index-annotation",
        "--exclude-newer", EXCLUDE_NEWER, "--no-cache", "--no-python-downloads",
    ]
    uv_dry_run = [
        uv, "pip", "sync", lock, "--python-version", python,
        "--python-platform", platform, "--require-hashes", "--dry-run", "--system",
        "--torch-backend", "cpu", "--index-strategy", "unsafe-best-match",
    ]
    abi = "cp" + python.replace(".", "")
    pip_dry_run = [
        sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
        "--require-hashes", "--only-binary=:all:", "--python-version", python,
        "--implementation", "cp", "--abi", abi,
        "--target", "<temporary-directory>", "-r", lock,
    ]
    platform_index = pip_dry_run.index("--python-version")
    targets = LINUX_PLATFORMS if profile["id"].startswith("linux-") else ("win_amd64",)
    pip_dry_run[platform_index:platform_index] = [
        part for target in targets for part in ("--platform", target)
    ]
    return resolve, uv_dry_run, pip_dry_run


def validate_profile(profile: dict[str, Any], uv: str) -> dict[str, Any]:
    lock = ROOT / str(profile["lock"])
    packages = parse_hashed_lock(lock.read_text(encoding="utf-8"))
    locked_names = {row["name"] for row in packages}
    resolve, uv_command, pip_command = _commands(profile, uv)
    uv_result = _run(uv_command)
    with tempfile.TemporaryDirectory(prefix="gfs-lock-validation-") as temporary:
        actual_pip_command = [
            temporary if value == "<temporary-directory>" else value
            for value in pip_command
        ]
        pip_result = _run(actual_pip_command)
    planned = _planned_packages(pip_result.stdout)
    uv_passed = uv_result.returncode == 0
    pip_passed = pip_result.returncode == 0 and planned == locked_names
    observations = {
        "all_packages_exact": len(packages) == len(locked_names) == 64,
        "all_packages_have_sha256": all(row["sha256"] for row in packages),
        "uv_target_resolution_passed": uv_passed,
        "pip_require_hashes_target_plan_passed": pip_passed,
        "would_install_package_count": len(planned),
        "environment_installed": False,
        "container_built": False,
    }
    passed = all(observations[key] for key in (
        "all_packages_exact", "all_packages_have_sha256",
        "uv_target_resolution_passed", "pip_require_hashes_target_plan_passed",
    ))
    report = {
        "schema_version": 1,
        "verification": (
            "gfs_python312_linux_target_lock"
            if profile["id"].startswith("linux-")
            else "gfs_python313_windows_reference_lock"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "passed_resolution_and_hash_dry_run" if passed else "validation_failed"
        ),
        "passed": passed,
        "target": {
            "python": profile["python"],
            "platform": (
                "x86_64 manylinux compatible through 2.28"
                if profile["id"].startswith("linux-")
                else "x86_64-pc-windows-msvc"
            ),
            "torch_backend": "cpu",
            "package_count": len(packages),
        },
        "commands": {
            "resolve": _render(resolve),
            "uv_dry_run": _render(uv_command),
            "pip_dry_run": _render(pip_command),
        },
        "artifact_sha256": {
            str(profile[key]): file_sha256(ROOT / str(profile[key]))
            for key in ("input", "lock", "sbom")
        },
        "observations": observations,
        "external_package_index_calls_made": True,
        "provider_api_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "limitations": [
            "The dry runs validate the target install plan and recorded distribution hashes without installing packages or executing imports in the target environment.",
            "A successful container build and target-runtime import test remain separate deployment gates.",
        ],
    }
    diagnostics = {
        "uv": _diagnostic(uv_result),
        "pip": _diagnostic(pip_result),
    }
    if not pip_passed and pip_result.returncode == 0:
        diagnostics["pip"] = "The pip install plan does not exactly match the 64 locked packages."
    if any(diagnostics.values()):
        report["diagnostics"] = {
            key: value for key, value in diagnostics.items() if value
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the observed validation reports")
    parser.add_argument(
        "--profile", choices=[profile["id"] for profile in PROFILES],
        help="Validate only one profile; by default both are validated",
    )
    args = parser.parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required for the non-installing lock validation")
    selected = [
        profile for profile in PROFILES
        if args.profile is None or profile["id"] == args.profile
    ]
    reports = [validate_profile(profile, uv) for profile in selected]
    for profile, report in zip(selected, reports, strict=True):
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.write:
            output = ROOT / str(profile["validation"])
            output.write_text(rendered, encoding="utf-8", newline="\n")
        print(rendered, end="")
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
