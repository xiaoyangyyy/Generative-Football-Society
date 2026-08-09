#!/usr/bin/env python3
"""Verify the code-only reproduction release contract without running compute."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_paper_package import verify_paper_package  # noqa: E402
from src.infrastructure import file_sha256  # noqa: E402


LOCK_CONTRACT = ROOT / "data/evaluation/dependency_lock_contract_v1.json"
LICENSE_REGISTRY = ROOT / "data/evaluation/data_license_registry_v1.json"
REPRODUCTION_MANIFEST = ROOT / "data/evaluation/reproduction_manifest_v1.json"
ENVIRONMENT = ROOT / "data/evaluation/reproduction_environment_v1.json"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
DOCKERFILE = ROOT / "Dockerfile"

EXPECTED_SOURCE_IDS = {
    "statsbomb_open_data",
    "metrica_sample_data",
    "skillcorner_open_data",
    "sportec_idsse",
    "transfermarkt_dataset_mirror",
}
EXACT_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _confined_file(relative: str) -> bool:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return False
    return candidate.is_file()


def _parse_exact_requirements(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in text.splitlines():
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        match = EXACT_REQUIREMENT.fullmatch(row)
        if not match:
            raise ValueError(f"requirement is not an exact direct pin: {row}")
        name = _normalized_name(match.group(1))
        if name in parsed:
            raise ValueError(f"duplicate direct requirement: {name}")
        parsed[name] = match.group(2)
    return parsed


def _license_checks(registry: dict) -> dict[str, bool]:
    policy = registry.get("policy") or {}
    sources = registry.get("sources") or []
    identifiers = [str(item.get("source_id") or "") for item in sources]
    decisions = {str(item.get("archive_decision") or "") for item in sources}
    return {
        "license_registry_schema": (
            registry.get("schema_version") == 1
            and registry.get("registry_id")
            == "gfs-known-external-data-license-decisions-v1"
        ),
        "known_provider_coverage_is_exact": (
            set(identifiers) == EXPECTED_SOURCE_IDS
            and len(identifiers) == len(set(identifiers))
        ),
        "license_policy_is_default_deny": (
            policy.get("default") == "exclude_from_distribution"
            and policy.get("unknown_or_ambiguous_terms")
            == "exclude_from_distribution"
            and policy.get("source_access_is_not_redistribution_permission") is True
            and policy.get("blanket_redistribution_approved") is False
        ),
        "all_source_manifests_exist": all(
            item.get("local_manifests")
            and all(_confined_file(str(path)) for path in item["local_manifests"])
            for item in sources
        ),
        "every_source_has_review_evidence_and_reason": all(
            item.get("license_status")
            and item.get("verification_status")
            and item.get("terms_observed")
            and item.get("reason")
            for item in sources
        ),
        "unapproved_external_data_is_excluded": (
            decisions == {"exclude_external_data"}
            and all(item.get("archive_eligible") is False for item in sources)
        ),
    }


def _observed_direct_packages(expected: dict[str, str]) -> dict[str, dict]:
    observed: dict[str, dict] = {}
    for name, expected_version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        observed[name] = {
            "expected": expected_version,
            "observed": actual,
            "matches": actual == expected_version,
        }
    return observed


def verify_reproduction_release() -> dict:
    contract = _read_json(LOCK_CONTRACT)
    registry = _read_json(LICENSE_REGISTRY)
    manifest = _read_json(REPRODUCTION_MANIFEST)
    environment = _read_json(ENVIRONMENT)
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requirements = _parse_exact_requirements(
        REQUIREMENTS.read_text(encoding="utf-8")
    )
    project_requirements = _parse_exact_requirements(
        "\n".join(pyproject["project"]["dependencies"])
    )
    expected = {
        _normalized_name(name): str(version)
        for name, version in contract.get("direct_dependencies", {}).items()
    }
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    artifacts = manifest.get("artifacts") or {}
    commands = manifest.get("commands") or []

    checks: dict[str, bool] = {
        "dependency_contract_schema_and_scope": (
            contract.get("schema_version") == 1
            and contract.get("lock_level") == "exact_direct_dependencies_only"
            and contract.get("full_transitive_hash_lock") is False
            and len(expected) == 8
        ),
        "requirements_are_exact_and_match_contract": requirements == expected,
        "pyproject_dependencies_match_direct_lock": project_requirements == expected,
        "python_support_window_is_bounded": (
            pyproject["project"]["requires-python"]
            == contract.get("python_constraint")
            == ">=3.11,<3.14"
        ),
        "build_backend_is_exactly_pinned": (
            pyproject["build-system"]["requires"]
            == [contract.get("build_system", {}).get("requirement")]
            and pyproject["build-system"]["build-backend"]
            == contract.get("build_system", {}).get("backend")
        ),
        "identity_inputs_exist_and_are_confined": all(
            _confined_file(str(path)) for path in contract.get("identity_inputs") or []
        ),
        "container_uses_exact_direct_lock": all(
            marker in dockerfile
            for marker in (
                "ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm",
                "COPY pyproject.toml requirements.txt README.md ./",
                "python -m pip install -r requirements.txt",
                "python -m pip install --no-deps .",
            )
        ),
        "container_direct_distributions_were_observed": (
            contract.get("index_observation", {}).get("target")
            == "CPython 3.12 Linux wheel or universal wheel"
            and contract.get("index_observation", {}).get(
                "all_direct_distributions_available"
            )
            is True
            and contract.get("index_observation", {}).get("packages")
            == {name: True for name in expected}
        ),
        "incomplete_lock_is_not_overclaimed": (
            contract.get("container_images_digest_pinned") is False
            and len(contract.get("limitations") or []) >= 4
            and environment.get("status")
            == "exact_direct_lock_partial_environment"
            and manifest.get("environment", {}).get("full_transitive_hash_lock")
            is False
        ),
        "release_artifacts_are_registered": all(
            artifacts.get(key) == value
            for key, value in {
                "dependency_lock_contract": "data/evaluation/dependency_lock_contract_v1.json",
                "data_license_registry": "data/evaluation/data_license_registry_v1.json",
                "release_verifier": "scripts/verify_reproduction_release.py",
            }.items()
        ),
        "release_audit_is_read_only": any(
            row.get("id") == "reproduction_release_audit"
            and row.get("command")
            == "python scripts/verify_reproduction_release.py"
            and row.get("effect") == "read_only"
            and row.get("runs_simulation") is False
            and row.get("explicit_authority_required") is False
            for row in commands
        ),
        "release_manifest_remains_preexecution": (
            manifest.get("stage") == "preexecution_auditable_package"
            and manifest.get("data_distribution", {}).get("license_review_complete")
            is False
            and manifest.get("archive", {}).get("status") == "not_built"
        ),
    }
    checks.update(_license_checks(registry))
    checks["paper_package_still_passes"] = verify_paper_package()["passed"]

    observed = _observed_direct_packages(expected)
    identity_paths = [
        *(str(path) for path in contract.get("identity_inputs") or []),
        "data/evaluation/dependency_lock_contract_v1.json",
        "data/evaluation/data_license_registry_v1.json",
        "data/evaluation/reproduction_manifest_v1.json",
    ]
    identity = {
        path: file_sha256(ROOT / path)
        for path in identity_paths
        if _confined_file(path)
    }
    readiness = {
        "runtime_direct_dependencies_match": all(
            item["matches"] for item in observed.values()
        ),
        "full_transitive_hash_lock": False,
        "container_images_digest_pinned": False,
        "container_build_verified": False,
        "external_data_archive_approved": False,
        "confirmatory_results_available": False,
        "independent_reproduction_available": False,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_code_only_reproduction_release_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_code_contract" if passed else "failed",
        "passed": passed,
        "release_ready": passed and all(readiness.values()),
        "dependency_lock_level": contract.get("lock_level"),
        "known_external_source_count": len(registry.get("sources") or []),
        "archive_eligible_external_source_count": sum(
            item.get("archive_eligible") is True
            for item in registry.get("sources") or []
        ),
        "checks": checks,
        "readiness": readiness,
        "observed_direct_packages": observed,
        "artifact_sha256": identity,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "limitations": [
            "exact direct versions are pinned, but transitive wheels and hashes are not",
            "container tags are fixed, but image digests and a successful build are unverified",
            "all known external data remains excluded from a release archive pending source-specific approval",
            "confirmatory execution and independent reproduction have not been performed",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_reproduction_release()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
