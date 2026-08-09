#!/usr/bin/env python3
"""Verify the supported Linux and Windows hashed reproducibility profiles."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402
from src.infrastructure.supply_chain import (  # noqa: E402
    build_cyclonedx_sbom,
    parse_hashed_lock,
)


PROFILES = (
    {
        "id": "linux-py312-x86_64-cpu",
        "python": "3.12",
        "platform": "x86_64-manylinux_2_28",
        "input": "requirements-linux-py312.in",
        "lock": "requirements-linux-py312.lock",
        "validation": "data/evaluation/target_lock_validation_v1.json",
        "sbom": "data/evaluation/supply_chain_sbom_v1.cdx.json",
        "resolution_target": (
            "CPython 3.12 / x86_64 manylinux_2_28 / CPU"
        ),
    },
    {
        "id": "windows-py313-x86_64-cpu",
        "python": "3.13",
        "platform": "x86_64-pc-windows-msvc",
        "input": "requirements-windows-py313.in",
        "lock": "requirements-windows-py313.lock",
        "validation": "data/evaluation/windows_lock_validation_v1.json",
        "sbom": "data/evaluation/supply_chain_sbom_windows_py313_v1.cdx.json",
        "resolution_target": (
            "CPython 3.13 / Windows x86_64 / CPU"
        ),
    },
)
EXACT = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _direct_dependencies() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies: dict[str, str] = {}
    for row in project["project"]["dependencies"]:
        match = EXACT.fullmatch(row)
        if not match:
            raise ValueError(f"dependency is not exact: {row}")
        dependencies[_canonical(match.group(1))] = match.group(2)
    return dependencies


def _project_identity() -> tuple[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    return str(project["name"]), str(project["version"])


def verify_reproducibility_matrix() -> dict:
    expected_direct = _direct_dependencies()
    project_name, project_version = _project_identity()
    profile_reports: list[dict] = []
    identity: dict[str, str] = {}
    for profile in PROFILES:
        paths = {
            key: ROOT / str(profile[key])
            for key in ("input", "lock", "validation", "sbom")
        }
        all_exist = all(path.is_file() for path in paths.values())
        if not all_exist:
            profile_reports.append({
                "id": profile["id"],
                "passed": False,
                "checks": {"all_artifacts_exist": False},
            })
            continue
        validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
        stored_sbom = json.loads(paths["sbom"].read_text(encoding="utf-8"))
        packages = parse_hashed_lock(paths["lock"].read_text(encoding="utf-8"))
        locked = {
            row["name"]: str(row["version"]).split("+", 1)[0]
            for row in packages
        }
        expected_sbom = build_cyclonedx_sbom(
            paths["lock"],
            project_name=project_name,
            project_version=project_version,
            resolution_target=str(profile["resolution_target"]),
        )
        observations = validation.get("observations") or {}
        hashes = validation.get("artifact_sha256") or {}
        checks = {
            "all_artifacts_exist": True,
            "lock_has_64_exact_hashed_packages": (
                len(packages) == 64
                and len({row["name"] for row in packages}) == 64
                and all(row["sha256"] for row in packages)
            ),
            "direct_dependencies_match": all(
                locked.get(name) == version
                for name, version in expected_direct.items()
            ),
            "validation_passed_both_resolvers": (
                validation.get("passed") is True
                and validation.get("status")
                == "passed_resolution_and_hash_dry_run"
                and observations.get("uv_target_resolution_passed") is True
                and observations.get("pip_require_hashes_target_plan_passed") is True
                and observations.get("would_install_package_count") == 64
                and observations.get("environment_installed") is False
            ),
            "validation_hashes_are_current": all(
                hashes.get(str(profile[key])) == file_sha256(paths[key])
                for key in ("input", "lock", "sbom")
            ),
            "cyclonedx_sbom_is_current": stored_sbom == expected_sbom,
            "safe_indices_are_explicit": all(
                marker in paths["lock"].read_text(encoding="utf-8")
                for marker in (
                    "--index-url https://pypi.org/simple",
                    "--extra-index-url https://download.pytorch.org/whl/cpu",
                )
            ),
        }
        for path in paths.values():
            identity[str(path.relative_to(ROOT)).replace("\\", "/")] = file_sha256(path)
        profile_reports.append({
            "id": profile["id"],
            "python": profile["python"],
            "platform": profile["platform"],
            "package_count": len(packages),
            "checks": checks,
            "passed": all(checks.values()),
        })
    passed = len(profile_reports) == 2 and all(
        row.get("passed") is True for row in profile_reports
    )
    return {
        "schema_version": 1,
        "verification": "gfs_supported_reproducibility_profile_matrix",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "profile_count": len(profile_reports),
        "profiles": profile_reports,
        "artifact_sha256": identity,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "scope": (
            "Reference deployment and development profiles; this does not claim "
            "support for every Python/OS combination allowed by pyproject.toml."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_reproducibility_matrix()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
