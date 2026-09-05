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
from scripts.product_validation_study import (  # noqa: E402
    protocol_report as product_validation_protocol_report,
)
from scripts.product_value_study import (  # noqa: E402
    protocol_report as product_value_protocol_report,
)
from scripts.academic_replication_study import (  # noqa: E402
    protocol_report as academic_replication_protocol_report,
)
from scripts.run_production_validation import (  # noqa: E402
    protocol_report as production_validation_protocol_report,
)
from scripts.verify_container_images import verify_container_images  # noqa: E402
from scripts.verify_data_release import verify_data_release  # noqa: E402
from scripts.verify_local_runtime import verify_local_runtime  # noqa: E402
from scripts.verify_independent_reproduction import (  # noqa: E402
    protocol_report as independent_reproduction_protocol_report,
)
from scripts.finalize_paper import (  # noqa: E402
    protocol_report as paper_finalization_protocol_report,
)
from scripts.verify_security_closure import (  # noqa: E402
    protocol_report as security_closure_protocol_report,
)
from scripts.build_excellence_evidence_kit import verify_kit  # noqa: E402
from scripts.verify_reproducibility_matrix import (  # noqa: E402
    verify_reproducibility_matrix,
)
from src.infrastructure import file_sha256  # noqa: E402
from src.infrastructure.supply_chain import (  # noqa: E402
    build_cyclonedx_sbom,
    parse_hashed_lock,
)
from src.product.excellence import validate_scoring_contract  # noqa: E402


LOCK_CONTRACT = ROOT / "data/evaluation/dependency_lock_contract_v1.json"
LICENSE_REGISTRY = ROOT / "data/evaluation/data_license_registry_v1.json"
REPRODUCTION_MANIFEST = ROOT / "data/evaluation/reproduction_manifest_v1.json"
ENVIRONMENT = ROOT / "data/evaluation/reproduction_environment_v1.json"
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
DOCKERFILE = ROOT / "Dockerfile"
TARGET_LOCK_INPUT = ROOT / "requirements-linux-py312.in"
TARGET_LOCK = ROOT / "requirements-linux-py312.lock"
CI_TOOL_INPUT = ROOT / "requirements-ci-linux-py312.in"
CI_TOOL_LOCK = ROOT / "requirements-ci-linux-py312.lock"
TARGET_VALIDATION = ROOT / "data/evaluation/target_lock_validation_v1.json"
SBOM = ROOT / "data/evaluation/supply_chain_sbom_v1.cdx.json"
WINDOWS_LOCK = ROOT / "requirements-windows-py313.lock"
WINDOWS_VALIDATION = ROOT / "data/evaluation/windows_lock_validation_v1.json"
WINDOWS_SBOM = ROOT / "data/evaluation/supply_chain_sbom_windows_py313_v1.cdx.json"
CI_ACTION_LOCK = ROOT / "data/evaluation/ci_action_lock_v1.json"
PRODUCT_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/product_validation_protocol_verification_v1.json"
)
PRODUCTION_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/production_validation_protocol_verification_v1.json"
)
PRODUCT_VALUE_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/product_value_validation_protocol_verification_v1.json"
)
ACADEMIC_REPLICATION_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/academic_action_replication_protocol_verification_v2.json"
)
INDEPENDENT_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/independent_action_reproduction_protocol_verification_v2.json"
)
PAPER_FINALIZATION_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/action_paper_finalization_protocol_verification_v2.json"
)
SECURITY_CLOSURE_PROTOCOL_REPORT = (
    ROOT / "data/evaluation/security_closure_protocol_verification_v2.json"
)
EXCELLENCE_SCORING_CONTRACT = (
    ROOT / "data/evaluation/excellence_scoring_contract_v1.json"
)
EXCELLENCE_EVIDENCE_KIT_REPORT = (
    ROOT / "data/evaluation/excellence_evidence_kit_verification_v1.json"
)

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
            and policy.get("unknown_or_ambiguous_terms") == "exclude_from_distribution"
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


def _ci_workflow_is_locked(action_lock: dict, workflow: str) -> bool:
    actions = action_lock.get("actions") or []
    return (
        action_lock.get("schema_version") == 1
        and action_lock.get("contract_id")
        == "gfs-github-actions-immutable-references-v1"
        and {row.get("id") for row in actions}
        == {
            "checkout",
            "setup_python",
            "upload_artifact",
            "attest_build_provenance",
        }
        and all(
            re.fullmatch(
                r"[0-9a-f]{40}",
                str(row.get("resolved_commit") or ""),
            )
            and workflow.count(f"{row.get('repository')}@{row.get('resolved_commit')}")
            == row.get("expected_workflow_occurrences")
            for row in actions
        )
        and all(
            marker in workflow
            for marker in (
                "--require-hashes -r requirements-linux-py312.lock",
                "--require-hashes -r requirements-ci-linux-py312.lock",
                "python -m ruff check src",
                "python -m ruff check src/product/web.py src/match_engine/action_engine.py --select C901,PLR0912,PLR0915",
                "python -m ruff check scripts/action_adoption_study.py scripts/run_formal_experiment.py",
                "python -m ruff check scripts/action_adoption_study.py scripts/run_formal_experiment.py --select C901,PLR0912,PLR0915",
                'python-version: "3.12.11"',
                "runs-on: ubuntu-24.04",
                "--deselect tests/test_world_model_semantic_event_training.py::test_trainer_persists_real_one_and_two_step_event_evidence",
                "deployment_contract_ci_verification_v1.json",
                "attestations: write",
            )
        )
        and not any(
            mutable in workflow
            for mutable in (
                "actions/checkout@v",
                "actions/setup-python@v",
                "actions/upload-artifact@v",
                "actions/attest-build-provenance@v",
                "ubuntu-latest",
                "pip install --upgrade pip",
            )
        )
        and workflow.count('python-version: "3.12.11"') == 2
        and workflow.count("runs-on: ubuntu-24.04") == 2
    )


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def verify_reproduction_release() -> dict:
    contract = _read_json(LOCK_CONTRACT)
    registry = _read_json(LICENSE_REGISTRY)
    manifest = _read_json(REPRODUCTION_MANIFEST)
    environment = _read_json(ENVIRONMENT)
    action_lock = _read_json(CI_ACTION_LOCK)
    stored_product_protocol = _read_json(PRODUCT_PROTOCOL_REPORT)
    stored_production_protocol = _read_json(PRODUCTION_PROTOCOL_REPORT)
    stored_product_value_protocol = _read_json(PRODUCT_VALUE_PROTOCOL_REPORT)
    stored_academic_replication_protocol = _read_json(
        ACADEMIC_REPLICATION_PROTOCOL_REPORT
    )
    stored_independent_protocol = _read_json(INDEPENDENT_PROTOCOL_REPORT)
    stored_paper_finalization_protocol = _read_json(
        PAPER_FINALIZATION_PROTOCOL_REPORT
    )
    stored_security_closure_protocol = _read_json(
        SECURITY_CLOSURE_PROTOCOL_REPORT
    )
    stored_evidence_kit = _read_json(EXCELLENCE_EVIDENCE_KIT_REPORT)
    excellence_scoring_contract = _read_json(EXCELLENCE_SCORING_CONTRACT)
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requirements = _parse_exact_requirements(REQUIREMENTS.read_text(encoding="utf-8"))
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
    ci_tool_input = _parse_exact_requirements(
        CI_TOOL_INPUT.read_text(encoding="utf-8")
    )
    ci_tool_packages = parse_hashed_lock(CI_TOOL_LOCK.read_text(encoding="utf-8"))
    ci_tools = {row["name"]: row for row in ci_tool_packages}
    ci_tool_contract = contract.get("ci_tooling") or {}
    expected_ci_tools = {
        _normalized_name(name): str(version)
        for name, version in (ci_tool_contract.get("direct_dependencies") or {}).items()
    }
    expected_ci_input = {
        **expected_ci_tools,
        **{
            _normalized_name(name): str(version)
            for name, version in (ci_tool_contract.get("portability_shims") or {}).items()
        },
    }
    validation = _read_json(TARGET_VALIDATION)
    sbom = _read_json(SBOM)
    expected_sbom = build_cyclonedx_sbom(
        TARGET_LOCK,
        project_name=str(pyproject["project"]["name"]),
        project_version=str(pyproject["project"]["version"]),
    )
    artifacts = manifest.get("artifacts") or {}
    commands = manifest.get("commands") or []
    ci_workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix_report = verify_reproducibility_matrix()
    local_runtime_report = verify_local_runtime()
    image_report = verify_container_images()
    data_release_report = verify_data_release()
    product_protocol = product_validation_protocol_report()
    production_protocol = production_validation_protocol_report()
    product_value_protocol = product_value_protocol_report()
    academic_replication_protocol = academic_replication_protocol_report()
    independent_protocol = independent_reproduction_protocol_report()
    paper_finalization_protocol = paper_finalization_protocol_report()
    security_closure_protocol = security_closure_protocol_report()
    evidence_kit = verify_kit()

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
            len(locked_packages)
            == contract.get("target_lock", {}).get("package_count")
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
        "ci_tooling_lock_is_exact_hashed_and_runtime_separate": (
            ci_tool_input == expected_ci_input
            and len(ci_tool_packages)
            == ci_tool_contract.get("package_count")
            == 7
            and all(row["sha256"] for row in ci_tool_packages)
            and all(
                name in ci_tools
                and _base_version(str(ci_tools[name]["version"])) == version
                for name, version in expected_ci_tools.items()
            )
            and not (set(expected_ci_tools) & set(locked))
            and ci_tool_contract.get("runtime_dependency_boundary") == "separate"
            and ci_tool_contract.get("install")
            == "python -m pip install --require-hashes -r requirements-ci-linux-py312.lock"
            and ci_tool_contract.get("source_gate") == "python -m ruff check src"
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
            and len((sbom.get("dependencies") or [])[0].get("dependsOn") or []) == 64
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
        "container_ci_is_immutable_attested_and_zero_training": _ci_workflow_is_locked(
            action_lock, ci_workflow
        ),
        "human_and_independent_validation_protocols_are_current": (
            product_protocol.get("passed") is True
            and product_protocol.get("study_executed") is False
            and product_protocol.get("participants_observed") == 0
            and stored_product_protocol.get("status") == product_protocol.get("status")
            and stored_product_protocol.get("checks") == product_protocol.get("checks")
            and stored_product_protocol.get("artifact_sha256")
            == product_protocol.get("artifact_sha256")
            and production_protocol.get("passed") is True
            and production_protocol.get("study_executed") is False
            and production_protocol.get("matches_executed") == 0
            and stored_production_protocol.get("checks")
            == production_protocol.get("checks")
            and stored_production_protocol.get("artifact_sha256")
            == production_protocol.get("artifact_sha256")
            and product_value_protocol.get("passed") is True
            and product_value_protocol.get("study_executed") is False
            and product_value_protocol.get("participants_observed") == 0
            and stored_product_value_protocol.get("checks")
            == product_value_protocol.get("checks")
            and stored_product_value_protocol.get("artifact_sha256")
            == product_value_protocol.get("artifact_sha256")
            and academic_replication_protocol.get("passed") is True
            and academic_replication_protocol.get("runs_executed") == 0
            and stored_academic_replication_protocol.get("checks")
            == academic_replication_protocol.get("checks")
            and stored_academic_replication_protocol.get("artifact_sha256")
            == academic_replication_protocol.get("artifact_sha256")
            and independent_protocol.get("passed") is True
            and independent_protocol.get("runs_executed") == 0
            and stored_independent_protocol.get("status")
            in {
                "registered_waiting_for_prerequisites",
                "ready_for_independent_reviewer",
            }
            and stored_independent_protocol.get("passed") is True
            and stored_independent_protocol.get("checks")
            == independent_protocol.get("checks")
            and stored_independent_protocol.get("artifact_sha256")
            == independent_protocol.get("artifact_sha256")
            and paper_finalization_protocol.get("passed") is True
            and stored_paper_finalization_protocol.get("checks")
            == paper_finalization_protocol.get("checks")
            and stored_paper_finalization_protocol.get("artifact_sha256")
            == paper_finalization_protocol.get("artifact_sha256")
            and security_closure_protocol.get("passed") is True
            and security_closure_protocol.get("closure_complete") is False
            and security_closure_protocol.get("known_incident_count") == 2
            and security_closure_protocol.get("revoked_incident_count") == 0
            and security_closure_protocol.get("known_incidents_complete") is False
            and security_closure_protocol.get("boundary_aware_secret_match_count")
            == 0
            and security_closure_protocol.get(
                "origin_has_embedded_credential"
            ) is False
            and stored_security_closure_protocol.get("checks")
            == security_closure_protocol.get("checks")
            and stored_security_closure_protocol.get("artifact_sha256")
            == security_closure_protocol.get("artifact_sha256")
        ),
        "excellence_scoring_contract_is_evidence_derived": all(
            validate_scoring_contract(excellence_scoring_contract).values()
        ),
        "excellence_evidence_kit_is_current_template_only_and_zero_execution": (
            evidence_kit.get("passed") is True
            and evidence_kit.get("status") == "passed_template_only_kit"
            and all(evidence_kit.get("checks", {}).values())
            and evidence_kit.get("external_calls_made") is False
            and evidence_kit.get("matches_executed") == 0
            and evidence_kit.get("training_executed") is False
            and stored_evidence_kit.get("checks") == evidence_kit.get("checks")
            and stored_evidence_kit.get("artifact_sha256")
            == evidence_kit.get("artifact_sha256")
            and stored_evidence_kit.get("archive_sha256")
            == evidence_kit.get("archive_sha256")
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
            and environment.get("status") == "supported_profile_matrix_runtime_verified"
            and manifest.get("environment", {}).get("full_transitive_hash_lock") is True
            and manifest.get("environment", {}).get("cross_platform_lock_matrix")
            is True
            and manifest.get("environment", {}).get("container_build_verified") is False
        ),
        "release_artifacts_are_registered": all(
            artifacts.get(key) == value
            for key, value in {
                "dependency_lock_contract": "data/evaluation/dependency_lock_contract_v1.json",
                "data_license_registry": "data/evaluation/data_license_registry_v1.json",
                "release_verifier": "scripts/verify_reproduction_release.py",
                "target_runtime_lock": "requirements-linux-py312.lock",
                "target_lock_input": "requirements-linux-py312.in",
                "ci_tool_lock": "requirements-ci-linux-py312.lock",
                "ci_tool_lock_input": "requirements-ci-linux-py312.in",
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
                "deployment_contract_verifier": "scripts/verify_deployment_contract.py",
                "container_ci_workflow": ".github/workflows/ci.yml",
                "ci_action_lock": "data/evaluation/ci_action_lock_v1.json",
                "product_validation_protocol": "data/evaluation/product_validation_protocol_v1.json",
                "product_validation_protocol_verification": "data/evaluation/product_validation_protocol_verification_v1.json",
                "product_validation_guide": "docs/PRODUCT_VALIDATION_STUDY.md",
                "product_validation_analyzer": "scripts/product_validation_study.py",
                "human_study_integrity": "src/product/human_study.py",
                "production_validation_protocol": "data/evaluation/production_validation_protocol_v1.json",
                "production_validation_protocol_verification": "data/evaluation/production_validation_protocol_verification_v1.json",
                "production_validation_guide": "docs/PRODUCTION_VALIDATION.md",
                "production_validation_runner": "scripts/run_production_validation.py",
                "product_value_validation_protocol": "data/evaluation/product_value_validation_protocol_v1.json",
                "product_value_validation_protocol_verification": "data/evaluation/product_value_validation_protocol_verification_v1.json",
                "product_value_validation_guide": "docs/PRODUCT_VALUE_STUDY.md",
                "product_value_validation_analyzer": "scripts/product_value_study.py",
                "product_value_case_pack_manifest": "data/evaluation/product_value_case_packs_v1.json",
                "product_value_scoring_seal": "data/evaluation/product_value_scoring_seal_v1.json",
                "product_value_study_delivery": "src/product/study_delivery.py",
                "product_value_participant_session_runner": "src/product/study_session.py",
                "academic_replication_protocol": "data/evaluation/academic_action_replication_protocol_v2.json",
                "academic_replication_protocol_verification": "data/evaluation/academic_action_replication_protocol_verification_v2.json",
                "academic_replication_guide": "docs/ACADEMIC_REPLICATION_STUDY.md",
                "academic_replication_runner": "scripts/academic_replication_study.py",
                "independent_reproduction_protocol": "data/evaluation/independent_action_reproduction_protocol_v2.json",
                "independent_reproduction_protocol_verification": "data/evaluation/independent_action_reproduction_protocol_verification_v2.json",
                "independent_reproduction_guide": "docs/INDEPENDENT_REPRODUCTION.md",
                "independent_reproduction_verifier": "scripts/verify_independent_reproduction.py",
                "paper_finalization_protocol": "data/evaluation/action_paper_finalization_protocol_v2.json",
                "paper_finalization_protocol_verification": "data/evaluation/action_paper_finalization_protocol_verification_v2.json",
                "paper_finalization_guide": "docs/PAPER_FINALIZATION.md",
                "paper_finalization_verifier": "scripts/finalize_paper.py",
                "excellence_scoring_contract": "data/evaluation/excellence_scoring_contract_v1.json",
                "excellence_score_verification": "data/evaluation/excellence_score_verification_v1.json",
                "excellence_score_module": "src/product/excellence.py",
                "excellence_score_verifier": "scripts/verify_excellence_score.py",
                "excellence_completion_plan": "src/product/completion_plan.py",
                "excellence_evidence_kit_builder": "scripts/build_excellence_evidence_kit.py",
                "excellence_evidence_kit_verification": "data/evaluation/excellence_evidence_kit_verification_v1.json",
                "excellence_evidence_kit_guide": "docs/EXCELLENCE_EVIDENCE_KIT.md",
                "product_control_plane": "src/product/control_plane.py",
                "product_cli": "src/cli.py",
                "product_web": "src/product/web.py",
                "product_telemetry": "src/product/telemetry.py",
                "security_closure_protocol": "data/evaluation/security_closure_protocol_v2.json",
                "security_closure_protocol_verification": "data/evaluation/security_closure_protocol_verification_v2.json",
                "security_closure_guide": "docs/SECURITY_CREDENTIAL_CLOSURE.md",
                "security_closure_verifier": "scripts/verify_security_closure.py",
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
            and row.get("command") == "python scripts/verify_reproduction_release.py"
            and row.get("effect") == "read_only"
            and row.get("runs_simulation") is False
            and row.get("explicit_authority_required") is False
            for row in commands
        ),
        "source_health_gate_is_read_only": any(
            row.get("id") == "source_health"
            and row.get("command") == "python -m ruff check src"
            and row.get("effect") == "read_only"
            and row.get("runs_simulation") is False
            and row.get("explicit_authority_required") is False
            for row in commands
        ),
        "release_manifest_has_completed_action_outcome_result": (
            manifest.get("stage") == "completed_action_outcome_results_package"
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
        "scripts/verify_container_runtime.py",
        "scripts/verify_deployment_contract.py",
        ".github/workflows/ci.yml",
        "data/evaluation/ci_action_lock_v1.json",
        "data/evaluation/product_validation_protocol_v1.json",
        "docs/PRODUCT_VALIDATION_STUDY.md",
        "scripts/product_validation_study.py",
        "src/product/human_study.py",
        "data/evaluation/production_validation_protocol_v1.json",
        "docs/PRODUCTION_VALIDATION.md",
        "scripts/run_production_validation.py",
        "data/evaluation/product_value_validation_protocol_v1.json",
        "docs/PRODUCT_VALUE_STUDY.md",
        "scripts/product_value_study.py",
        "data/evaluation/product_value_case_packs_v1.json",
        "data/evaluation/product_value_scoring_seal_v1.json",
        "src/product/study_delivery.py",
        "src/product/study_session.py",
        "data/evaluation/academic_action_replication_protocol_v2.json",
        "docs/ACADEMIC_REPLICATION_STUDY.md",
        "scripts/academic_replication_study.py",
        "data/evaluation/independent_action_reproduction_protocol_v2.json",
        "docs/INDEPENDENT_REPRODUCTION.md",
        "scripts/verify_independent_reproduction.py",
        "data/evaluation/action_paper_finalization_protocol_v2.json",
        "docs/PAPER_FINALIZATION.md",
        "scripts/finalize_paper.py",
        "data/evaluation/excellence_scoring_contract_v1.json",
        "src/product/excellence.py",
        "scripts/verify_excellence_score.py",
        "src/product/completion_plan.py",
        "scripts/build_excellence_evidence_kit.py",
        "data/evaluation/excellence_evidence_kit_verification_v1.json",
        "docs/EXCELLENCE_EVIDENCE_KIT.md",
        "src/product/control_plane.py",
        "src/cli.py",
        "src/product/web.py",
        "src/product/telemetry.py",
        "data/evaluation/security_closure_protocol_v2.json",
        "docs/SECURITY_CREDENTIAL_CLOSURE.md",
        "scripts/verify_security_closure.py",
        "scripts/verify_paper_package.py",
        "scripts/verify_reproduction_release.py",
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
        "product_validation_protocol_ready": product_protocol.get("passed") is True,
        "production_validation_protocol_ready": (
            production_protocol.get("passed") is True
        ),
        "product_value_validation_protocol_ready": (
            product_value_protocol.get("passed") is True
        ),
        "academic_replication_protocol_ready": (
            academic_replication_protocol.get("passed") is True
        ),
        "independent_reproduction_protocol_ready": (
            independent_protocol.get("passed") is True
        ),
        "paper_finalization_protocol_ready": (
            paper_finalization_protocol.get("passed") is True
        ),
        "security_closure_protocol_ready": (
            security_closure_protocol.get("passed") is True
        ),
        "evidence_derived_scoring_ready": all(
            validate_scoring_contract(excellence_scoring_contract).values()
        ),
        "excellence_evidence_kit_ready": evidence_kit.get("passed") is True,
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
            "the action-policy outcome study is complete, but the separate human, production, academic, and independent validation programs remain unperformed",
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
