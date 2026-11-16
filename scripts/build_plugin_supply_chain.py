#!/usr/bin/env python3
"""Compile, verify, and project the two plugin runtime supply chains."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_supply_chain_locks import _planned_packages
from src.infrastructure import file_sha256
from src.infrastructure.supply_chain import (
    build_cyclonedx_sbom,
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
PROFILES = (
    {
        "id": "core",
        "project": "gfs-core",
        "lock": "gfs-core/requirements-linux-py312.lock",
        "input": "gfs-core/requirements-linux-py312.in",
        "sbom": "gfs-core/supply-chain/sbom.cdx.json",
        "validation": "gfs-core/supply-chain/lock-validation.json",
        "torch": False,
    },
    {
        "id": "studio",
        "project": "gfs-studio",
        "lock": "gfs-studio/requirements-linux-py312.lock",
        "input": "gfs-studio/requirements-linux-py312.in",
        "sbom": "gfs-studio/supply-chain/sbom.cdx.json",
        "validation": "gfs-studio/supply-chain/lock-validation.json",
        "torch": True,
    },
)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, encoding="utf-8",
        errors="backslashreplace", timeout=1800, check=False,
    )


def _project_identity(project: str) -> tuple[str, str]:
    document = tomllib.loads(
        (ROOT / project / "pyproject.toml").read_text(encoding="utf-8")
    )
    return document["project"]["name"], document["project"]["version"]


def _expected_sbom(profile: dict) -> dict:
    name, version = _project_identity(profile["project"])
    return build_cyclonedx_sbom(
        ROOT / profile["lock"], project_name=name, project_version=version,
        resolution_target="CPython 3.12 / x86_64 manylinux_2_28 / CPU",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def _compile(profile: dict, uv: str) -> None:
    command = [
        uv, "pip", "compile", profile["input"], "--output-file", profile["lock"],
        "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28",
        "--only-binary", ":all:", "--generate-hashes", "--no-annotate",
        "--emit-index-url", "--emit-index-annotation", "--exclude-newer",
        EXCLUDE_NEWER, "--no-cache", "--no-python-downloads",
    ]
    if profile["torch"]:
        command.extend(["--torch-backend", "cpu"])
    result = _run(command)
    if result.returncode:
        raise RuntimeError(
            f"{profile['id']} lock compilation failed: "
            + (result.stderr or result.stdout)[-4000:]
        )
    if profile["torch"]:
        lock = ROOT / profile["lock"]
        text = lock.read_text(encoding="utf-8")
        index = "--extra-index-url https://download.pytorch.org/whl/cpu"
        if index not in text:
            text = text.replace("--index-url https://pypi.org/simple\n", (
                "--index-url https://pypi.org/simple\n" + index + "\n"
            ), 1)
            lock.write_text(text, encoding="utf-8", newline="\n")


def _validate(profile: dict, uv: str) -> dict:
    packages = parse_hashed_lock(
        (ROOT / profile["lock"]).read_text(encoding="utf-8")
    )
    locked = {row["name"] for row in packages}
    uv_command = [
        uv, "pip", "sync", profile["lock"], "--python-version", "3.12",
        "--python-platform", "x86_64-manylinux_2_28", "--require-hashes",
        "--dry-run", "--system", "--index-strategy", "unsafe-best-match",
    ]
    if profile["torch"]:
        uv_command.extend(["--torch-backend", "cpu"])
    uv_result = _run(uv_command)
    pip_command = [
        sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
        "--require-hashes", "--only-binary=:all:",
    ]
    for target in LINUX_PLATFORMS:
        pip_command.extend(["--platform", target])
    pip_command.extend([
        "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312",
        "-r", profile["lock"],
    ])
    with tempfile.TemporaryDirectory(prefix=f"gfs-{profile['id']}-lock-") as target:
        pip_command.extend(["--target", target])
        pip_result = _run(pip_command)
    planned = _planned_packages(pip_result.stdout)
    passed = (
        uv_result.returncode == 0
        and pip_result.returncode == 0
        and planned == locked
        and all(row["sha256"] for row in packages)
    )
    report = {
        "schema_version": 1,
        "verification": f"gfs_{profile['id']}_linux_py312_hashed_lock",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed_resolution_and_hash_dry_run" if passed else "failed",
        "passed": passed,
        "package_count": len(packages),
        "observations": {
            "uv_target_resolution_passed": uv_result.returncode == 0,
            "pip_require_hashes_target_plan_passed": (
                pip_result.returncode == 0 and planned == locked
            ),
            "would_install_package_count": len(planned),
            "environment_installed": False,
        },
        "artifact_sha256": {
            profile[key]: file_sha256(ROOT / profile[key])
            for key in ("input", "lock", "sbom")
        },
        "external_package_index_calls_made": True,
        "provider_api_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
    }
    if not passed:
        report["diagnostics"] = {
            "uv": (
                None if uv_result.returncode == 0
                else (uv_result.stderr or uv_result.stdout)[-4000:]
            ),
            "pip": (
                None if pip_result.returncode == 0 and planned == locked
                else (pip_result.stderr or pip_result.stdout)[-4000:]
            ),
        }
    return report


def verify_current() -> dict:
    reports = []
    for profile in PROFILES:
        paths = {key: ROOT / profile[key] for key in ("input", "lock", "sbom", "validation")}
        exists = all(path.is_file() for path in paths.values())
        packages = (
            parse_hashed_lock(paths["lock"].read_text(encoding="utf-8"))
            if paths["lock"].is_file() else []
        )
        validation = (
            json.loads(paths["validation"].read_text(encoding="utf-8"))
            if paths["validation"].is_file() else {}
        )
        hashes = validation.get("artifact_sha256", {})
        checks = {
            "artifacts_exist": exists,
            "hashed_exact_closure": bool(packages) and all(
                row["sha256"] for row in packages
            ),
            "sbom_current": exists and json.loads(
                paths["sbom"].read_text(encoding="utf-8")
            ) == _expected_sbom(profile),
            "validation_passed": validation.get("passed") is True,
            "validation_hashes_current": exists and all(
                hashes.get(profile[key]) == file_sha256(paths[key])
                for key in ("input", "lock", "sbom")
            ),
        }
        reports.append({
            "id": profile["id"], "package_count": len(packages),
            "checks": checks, "passed": all(checks.values()),
        })
    return {
        "schema_version": 1,
        "passed": all(report["passed"] for report in reports),
        "profiles": reports,
        "external_calls_made": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--profile", choices=[profile["id"] for profile in PROFILES])
    args = parser.parse_args(argv)
    uv = shutil.which("uv")
    if (args.compile or args.validate) and uv is None:
        parser.error("uv is required")
    selected = [
        profile for profile in PROFILES
        if args.profile is None or profile["id"] == args.profile
    ]
    for profile in selected:
        if args.compile:
            _compile(profile, uv)
        if args.write:
            _write_json(ROOT / profile["sbom"], _expected_sbom(profile))
        if args.validate:
            report = _validate(profile, uv)
            if args.write:
                _write_json(ROOT / profile["validation"], report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["passed"]:
                return 1
    report = verify_current() if not (args.compile or args.write or args.validate) else None
    if report is not None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
