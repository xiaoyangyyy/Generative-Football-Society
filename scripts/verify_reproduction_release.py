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
from scripts.verify_container_images import verify_container_images  # noqa: E402
from scripts.verify_data_release import verify_data_release  # noqa: E402
from scripts.verify_local_runtime import verify_local_runtime  # noqa: E402
from scripts.verify_reproducibility_matrix import (  # noqa: E402
    verify_reproducibility_matrix,
)
from src.infrastructure import file_sha256  # noqa: E402
from src.infrastructure.supply_chain import (  # noqa: E402
    build_cyclonedx_sbom,
    parse_hashed_lock,
)


LOCK_CONTRACT = ROOT / "data/evaluation/dependency_lock_contract_v1.json"
LICENSE_REGISTRY = ROOT / "data/evaluation/data_license_registry_v1.json"
REPRODUCTION_MANIFEST = ROOT / "data/evaluation/reproduction_manifest_v1.json"
ENVIRONMENT = ROOT / "data/evaluation/reproduction_environment_v1.json"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
DOCKERFILE = ROOT / "Dockerfile"
TARGET_LOCK_INPUT = ROOT / "requirements-linux-py312.in"
TARGET_LOCK = ROOT / "requirements-linux-py312.lock"
TARGET_VALIDATION = ROOT / "data/evaluation/target_lock_validation_v1.json"
SBOM = ROOT / "data/evaluation/supply_chain_sbom_v1.cdx.json"
WINDOWS_LOCK = ROOT / "requirements-windows-py313.lock"
WINDOWS_VALIDATION = ROOT / "data/evaluation/windows_lock_validation_v1.json"
WINDOWS_SBOM = ROOT / "data/evaluation/supply_chain_sbom_windows_py313_v1.cdx.json"

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
    by_id = {str(item.get("source_id") or ""): item for item in sources}
    sportec = by_id.get("sportec_idsse") or {}
    excluded = [
        item for identifier, item in by_id.items() if identifier != "sportec_idsse"
    ]
    return {
        "license_registry_schema": (
            registry.get("schema_version") == 1
            and registry.get("registry_id")
            == "gfs-source-specific-data-license-decisions-v2"
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
            and policy.get("source_specific_review_complete") is True
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
        "source_specific_license_decisions_are_bounded": (
            sportec.get("archive_eligible") is True
            and sportec.get("archive_decision") == "include_derived_sportec_subset"
            and sportec.get("license_status") == "CC-BY-4.0"
            and all(item.get("archive_eligible") is False for item in excluded)
            and all(
                item.get("archive_decision") == "exclude_external_data"
                for item in excluded
            )
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


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


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
    lock_text = TARGET_LOCK.read_text(encoding="utf-8")
    locked_packages = parse_hashed_lock(lock_text)
    locked = {row["name"]: row for row in locked_packages}
    validation = _read_json(TARGET_VALIDATION)
    sbom = _read_json(SBOM)
    expected_sbom = build_cyclonedx_sbom(
        TARGET_LOCK,
        project_name=str(pyproject["project"]["name"]),
        project_version=str(pyproject["project"]["version"]),
    )
    artifacts = manifest.get("artifacts") or {}
    commands = manifest.get("commands") or []
    matrix_report = verify_reproducibility_matrix()
    local_runtime_report = verify_local_runtime()
    image_report = verify_container_images()
    data_release_report = verify_data_release()

    checks: dict[str, bool] = {
        "dependency_contract_schema_and_scope": (
            contract.get("schema_version") == 1
            and contract.get("lock_level")
            == "supported_profile_transitive_hash_lock_matrix"
            and contract.get("full_transitive_hash_lock") is True
            and contract.get("cross_platform_lock_matrix") is True
            and contract.get("container_images_digest_pinned") is True
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
        "target_lock_is_complete_and_hashed": (
            len(locked_packages) == contract.get("target_lock", {}).get("package_count")
            == 64
            and all(row["sha256"] for row in locked_packages)
            and contract.get("target_lock", {}).get("all_packages_exact") is True
            and contract.get("target_lock", {}).get("all_packages_hashed") is True
        ),
        "target_lock_contains_direct_contract": all(
            name in locked and _base_version(str(locked[name]["version"])) == version
            for name, version in expected.items()
        ),
        "target_lock_has_safe_index_and_portability_contract": (
            "--index-url https://pypi.org/simple" in lock_text
            and "--extra-index-url https://download.pytorch.org/whl/cpu" in lock_text
            and locked.get("torch", {}).get("version") == "2.12.1+cpu"
            and {
                name: locked.get(name, {}).get("version")
                for name in ("tzdata", "colorama")
            }
            == contract.get("target_lock", {}).get("portability_shims")
        ),
        "target_lock_validation_matches_artifacts": (
            validation.get("passed") is True
            and validation.get("status") == "passed_resolution_and_hash_dry_run"
            and validation.get("target", {}).get("package_count") == 64
            and validation.get("observations", {}).get(
                "pip_require_hashes_target_plan_passed"
            )
            is True
            and all(
                _confined_file(path)
                and validation.get("artifact_sha256", {}).get(path)
                == file_sha256(ROOT / path)
                for path in (
                    "requirements-linux-py312.in",
                    "requirements-linux-py312.lock",
                    "data/evaluation/supply_chain_sbom_v1.cdx.json",
                )
            )
        ),
        "cyclonedx_sbom_is_current_and_complete": (
            sbom == expected_sbom
            and sbom.get("bomFormat") == "CycloneDX"
            and sbom.get("specVersion") == "1.6"
            and len(sbom.get("components") or []) == 64
            and len((sbom.get("dependencies") or [])[0].get("dependsOn") or [])
            == 64
        ),
        "supported_profile_matrix_is_current": (
            matrix_report.get("passed") is True
            and matrix_report.get("profile_count") == 2
            and WINDOWS_LOCK.is_file()
            and WINDOWS_VALIDATION.is_file()
            and WINDOWS_SBOM.is_file()
        ),
        "local_runtime_import_smoke_passes": (
            local_runtime_report.get("passed") is True
            and all(local_runtime_report.get("checks", {}).values())
        ),
        "container_image_digest_contract_is_current": (
            image_report.get("passed") is True
            and all(image_report.get("checks", {}).values())
        ),
        "container_uses_transitive_hash_lock": all(
            marker in dockerfile
            for marker in (
                "ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7",
                "COPY pyproject.toml requirements.txt requirements-linux-py312.lock README.md ./",
                "python -m pip install --require-hashes -r requirements-linux-py312.lock",
                "python -m pip install --no-build-isolation --no-deps .",
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
        "target_scope_and_remaining_limits_are_honest": (
            contract.get("container_images_digest_pinned") is True
            and len(contract.get("limitations") or []) >= 4
            and environment.get("status")
            == "supported_profile_matrix_runtime_verified"
            and manifest.get("environment", {}).get("full_transitive_hash_lock")
            is True
            and manifest.get("environment", {}).get("cross_platform_lock_matrix")
            is True
            and manifest.get("environment", {}).get("container_build_verified")
            is False
        ),
        "release_artifacts_are_registered": all(
            artifacts.get(key) == value
            for key, value in {
                "dependency_lock_contract": "data/evaluation/dependency_lock_contract_v1.json",
                "data_license_registry": "data/evaluation/data_license_registry_v1.json",
                "release_verifier": "scripts/verify_reproduction_release.py",
                "target_runtime_lock": "requirements-linux-py312.lock",
                "target_lock_input": "requirements-linux-py312.in",
                "target_lock_validation": "data/evaluation/target_lock_validation_v1.json",
                "supply_chain_sbom": "data/evaluation/supply_chain_sbom_v1.cdx.json",
                "windows_runtime_lock": "requirements-windows-py313.lock",
                "windows_lock_validation": "data/evaluation/windows_lock_validation_v1.json",
                "windows_supply_chain_sbom": "data/evaluation/supply_chain_sbom_windows_py313_v1.cdx.json",
                "reproducibility_matrix_verification": "data/evaluation/reproducibility_matrix_verification_v1.json",
                "local_runtime_verification": "data/evaluation/local_runtime_verification_v1.json",
                "container_image_contract": "data/evaluation/container_image_contract_v1.json",
                "container_image_verification": "data/evaluation/container_image_verification_v1.json",
                "container_runtime_verifier": "scripts/verify_container_runtime.py",
                "data_release_manifest": "data/evaluation/data_release_manifest_v1.json",
                "data_release_verification": "data/evaluation/data_release_verification_v1.json",
                "data_release_builder": "scripts/build_data_release_archive.py",
                "data_release_verifier": "scripts/verify_data_release.py",
                "idsse_attribution": "docs/IDSSE_ATTRIBUTION.md",
                "sbom_builder": "scripts/build_supply_chain_sbom.py",
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
            is True
            and manifest.get("archive", {}).get("status")
            == "manifest_addressed_subset_ready"
            and data_release_report.get("passed") is True
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
        "data/evaluation/target_lock_validation_v1.json",
        "data/evaluation/windows_lock_validation_v1.json",
        "data/evaluation/local_runtime_verification_v1.json",
        "data/evaluation/reproducibility_matrix_verification_v1.json",
        "data/evaluation/container_image_verification_v1.json",
        "data/evaluation/data_release_manifest_v1.json",
        "data/evaluation/data_release_verification_v1.json",
        "docs/IDSSE_ATTRIBUTION.md",
        "scripts/build_data_release_archive.py",
        "scripts/verify_data_release.py",
    ]
    identity = {
        path: file_sha256(ROOT / path)
        for path in identity_paths
        if _confined_file(path)
    }
    readiness = {
        "runtime_direct_dependencies_match": (
            all(item["matches"] for item in observed.values())
            and local_runtime_report.get("passed") is True
        ),
        "full_transitive_hash_lock": True,
        "cross_platform_lock_matrix": matrix_report.get("passed") is True,
        "cyclonedx_sbom_current": True,
        "container_images_digest_pinned": image_report.get("passed") is True,
        "container_build_verified": False,
        "external_data_archive_approved": data_release_report.get("passed") is True,
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
            "the verified matrix covers named Linux deployment and Windows development reference profiles, not every permitted Python/OS combination",
            "container image digests are fixed, but a successful build is unverified on this Docker-less host",
            "only the DFL-authorized CC BY 4.0 IDSSE subset is approved; four other external sources remain excluded",
            "the licensed supplement boundary does not audit or rewrite repository history",
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
