import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_reproduction_release import (
    CI_ACTION_LOCK,
    LICENSE_REGISTRY,
    _ci_workflow_is_locked,
    _license_checks,
    _parse_exact_requirements,
    verify_reproduction_release,
)

CI_WORKFLOW = Path(".github/workflows/ci.yml")


def test_code_only_release_contract_is_honest_and_zero_compute():
    report = verify_reproduction_release()
    assert report["passed"]
    assert report["status"] == "passed_code_contract"
    assert report["release_ready"] is False
    assert (
        report["dependency_lock_level"]
        == "supported_profile_transitive_hash_lock_matrix"
    )
    assert report["known_external_source_count"] == 5
    assert report["archive_eligible_external_source_count"] == 1
    assert all(report["checks"].values())
    assert report["readiness"]["full_transitive_hash_lock"] is True
    assert report["readiness"]["runtime_direct_dependencies_match"] is True
    assert report["readiness"]["cross_platform_lock_matrix"] is True
    assert report["readiness"]["cyclonedx_sbom_current"] is True
    assert report["readiness"]["container_images_digest_pinned"] is True
    assert report["readiness"]["container_build_verified"] is False
    assert report["readiness"]["external_data_archive_approved"] is True
    assert report["readiness"]["paper_finalization_protocol_ready"] is True
    assert report["readiness"]["security_closure_protocol_ready"] is True
    assert report["readiness"]["evidence_derived_scoring_ready"] is True
    assert report["readiness"]["excellence_evidence_kit_ready"] is True
    assert report["checks"][
        "excellence_evidence_kit_is_current_template_only_and_zero_execution"
    ] is True
    assert "src/product/completion_plan.py" in report["artifact_sha256"]
    assert "src/product/control_plane.py" in report["artifact_sha256"]
    assert "src/cli.py" in report["artifact_sha256"]
    assert "src/product/web.py" in report["artifact_sha256"]
    assert "src/product/telemetry.py" in report["artifact_sha256"]
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["formal_experiment_executed"] is False


def test_release_verifier_is_a_standalone_cli():
    completed = subprocess.run(
        [sys.executable, "scripts/verify_reproduction_release.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["release_ready"] is False


def test_exact_requirement_parser_rejects_drift_and_duplicates():
    assert _parse_exact_requirements("numpy==2.4.3") == {"numpy": "2.4.3"}
    with pytest.raises(ValueError, match="not an exact direct pin"):
        _parse_exact_requirements("numpy>=2")
    duplicate = """python_dotenv==1.2.2
python-dotenv==1.2.2"""
    with pytest.raises(ValueError, match="duplicate"):
        _parse_exact_requirements(duplicate)


def test_license_gate_rejects_eligible_unapproved_source():
    registry = json.loads(LICENSE_REGISTRY.read_text(encoding="utf-8"))
    assert all(_license_checks(registry).values())
    tampered = copy.deepcopy(registry)
    tampered["sources"][0]["archive_eligible"] = True
    assert not _license_checks(tampered)[
        "source_specific_license_decisions_are_bounded"
    ]


def test_license_gate_rejects_missing_provider_coverage():
    registry = json.loads(LICENSE_REGISTRY.read_text(encoding="utf-8"))
    registry["sources"].pop()
    assert not _license_checks(registry)["known_provider_coverage_is_exact"]


def test_ci_gate_rejects_action_commit_drift():
    action_lock = json.loads(CI_ACTION_LOCK.read_text(encoding="utf-8"))
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert _ci_workflow_is_locked(action_lock, workflow)
    tampered = copy.deepcopy(action_lock)
    tampered["actions"][0]["resolved_commit"] = "0" * 40
    assert not _ci_workflow_is_locked(tampered, workflow)
