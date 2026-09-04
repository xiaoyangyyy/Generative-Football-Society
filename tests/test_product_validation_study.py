import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from scripts.product_validation_study import (
    OBSERVER_ATTESTATION,
    PROTOCOL_PATH,
    REVIEWER_ATTESTATION,
    _read_json,
    _validate_record,
    analyze,
    protocol_report,
)
from src.product.human_study import register_participant


ROLES = ("football_analyst", "research_engineer", "product_operator")
TASK_IDS = (
    "create_or_load_studio",
    "interpret_release_readiness",
    "run_stable_match",
    "review_auditable_report",
    "backup_verify_and_recover",
)


def _archive_file(root, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    (root / digest).write_bytes(payload)
    return digest


def _record(index: int, registration: dict, archive, *, complete: bool = True) -> dict:
    started = datetime.fromisoformat(registration["registered_at"]) + timedelta(seconds=1)
    return {
        "schema_version": 1,
        "protocol_id": "gfs-studio-target-user-validation-v1",
        "registration_id": registration["registration_id"],
        "participant_id": registration["participant_id"],
        "target_role": registration["target_role"],
        "consent": True,
        "session_started_at": started.isoformat(),
        "session_ended_at": (started + timedelta(hours=1)).isoformat(),
        "moderator_id": registration["moderator_id"],
        "tasks": [
            {
                "task_id": task_id,
                "completed": complete or task_index > 0,
                "critical_error": False,
                "duration_seconds": 300,
                "assistance_count": 0,
                "evidence_sha256": (
                    _archive_file(
                        archive,
                        f"task:{index}:{task_index}".encode(),
                    )
                    if complete or task_index > 0
                    else None
                ),
            }
            for task_index, task_id in enumerate(TASK_IDS)
        ],
        "sus_responses": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
        "qualitative_tags": ["trust"],
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": OBSERVER_ATTESTATION,
    }


def _external_review(archive) -> dict:
    common = {
        "independent_of_implementation": True,
        "organization": "Independent Review Lab",
        "signed_at": "2026-08-11T09:00:00Z",
        "attestation": REVIEWER_ATTESTATION,
        "passed": True,
        "unresolved_critical_findings": 0,
    }
    return {
        "schema_version": 1,
        "protocol_id": "gfs-studio-target-user-validation-v1",
        "accessibility": {
            **common,
            "reviewer_id": "R-ACCESS0001",
            "standard": "WCAG-2.2-AA",
            "artifact_sha256": {
                "review.pdf": _archive_file(archive, b"accessibility-review")
            },
        },
        "security": {
            **common,
            "reviewer_id": "R-SECURE0001",
            "method": "independent_application_security_review",
            "artifact_sha256": {
                "review.pdf": _archive_file(archive, b"security-review")
            },
        },
        "issue_resolution": {
            "findings_total": 3,
            "unresolved_critical_count": 0,
            "risk_accepted_critical_count": 0,
            "all_critical_findings_closed": True,
        },
    }


def _write_inputs(tmp_path, *, complete=lambda index: True, review_mutator=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry_path = tmp_path / "registry.json"
    records_path = tmp_path / "records.jsonl"
    archive = tmp_path / "evidence"
    archive.mkdir()
    registrations = []
    for index in range(10):
        result = register_participant(
            project_root=tmp_path,
            protocol_path=PROTOCOL_PATH,
            registry_path=registry_path,
            records_path=records_path,
            participant_id=f"P-{index:012d}",
            target_role=ROLES[index % len(ROLES)],
            moderator_id="M-00000001",
            consent_recorded=True,
        )
        registrations.append(result["registration"])
    records = [
        _record(index, registration, archive, complete=complete(index))
        for index, registration in enumerate(registrations)
    ]
    records_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    review = _external_review(archive)
    if review_mutator is not None:
        review_mutator(review)
    review_path = tmp_path / "external-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return records_path, review_path, registry_path, archive, records


def test_product_validation_protocol_is_frozen_zero_execution():
    report = protocol_report()
    assert report["passed"] is True
    assert report["status"] == "passed_preregistered_not_executed"
    assert report["result_available"] is False
    assert all(report["checks"].values())
    assert report["participants_observed"] == 0
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False


def test_analyzer_passes_only_registered_byte_verified_evidence(tmp_path):
    records, review, registry, archive, _ = _write_inputs(tmp_path)
    decision = analyze(records, review, registry, archive)
    assert decision["passed"] is True
    assert decision["participants_valid"] == 10
    assert decision["metrics"]["participant_core_workflow_success_rate"] == 1.0
    assert decision["metrics"]["mean_sus"] == 100.0
    assert all(decision["gates"].values())
    assert all(decision["external_review_checks"].values())
    assert decision["registry_checks"]["all_registrations_analyzed"] is True
    assert decision["evidence_archive"]["verified_artifacts"] == 52


def test_analyzer_preserves_below_threshold_result(tmp_path):
    records, review, registry, archive, _ = _write_inputs(
        tmp_path,
        complete=lambda index: index >= 2,
    )
    decision = analyze(records, review, registry, archive)
    assert decision["passed"] is False
    assert decision["metrics"]["participant_core_workflow_success_rate"] == 0.8
    assert decision["gates"]["core_workflow_success_threshold"] is False


def test_record_validation_rejects_missing_evidence_and_posthoc_exclusion():
    protocol = _read_json(PROTOCOL_PATH)
    record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "registration_id": "REG-" + "A" * 24,
        "participant_id": "P-000000000001",
        "target_role": "football_analyst",
        "consent": True,
        "session_started_at": "2026-08-10T09:00:00+08:00",
        "session_ended_at": "2026-08-10T10:00:00+08:00",
        "moderator_id": "M-00000001",
        "tasks": [
            {
                "task_id": task_id,
                "completed": True,
                "critical_error": False,
                "duration_seconds": 300,
                "assistance_count": 0,
                "evidence_sha256": "a" * 64,
            }
            for task_id in TASK_IDS
        ],
        "sus_responses": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
        "qualitative_tags": ["trust"],
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": OBSERVER_ATTESTATION,
    }
    record["tasks"][0]["evidence_sha256"] = None
    with pytest.raises(ValueError, match="lacks evidence"):
        _validate_record(record, protocol)
    excluded = dict(record)
    excluded["tasks"] = []
    excluded["sus_responses"] = []
    excluded["qualitative_tags"] = []
    excluded["excluded"] = True
    excluded["exclusion_reason"] = "participant_failed_tasks"
    with pytest.raises(ValueError, match="invalid exclusion"):
        _validate_record(excluded, protocol)
    private = dict(record)
    private["raw_notes"] = "must never enter the analysis record"
    with pytest.raises(ValueError, match="privacy allowlist"):
        _validate_record(private, protocol)


def test_external_review_must_be_distinct_and_independent(tmp_path):
    def mutate(review):
        review["security"]["reviewer_id"] = review["accessibility"]["reviewer_id"]

    records, review, registry, archive, _ = _write_inputs(
        tmp_path,
        review_mutator=mutate,
    )
    decision = analyze(records, review, registry, archive)
    assert decision["passed"] is False
    assert decision["external_review_checks"][
        "reviewers_are_distinct_and_independent"
    ] is False


def test_analyzer_rejects_missing_evidence_and_registry_tampering(tmp_path):
    records, review, registry, archive, rows = _write_inputs(tmp_path)
    missing_digest = rows[0]["tasks"][0]["evidence_sha256"]
    (archive / missing_digest).unlink()
    with pytest.raises(ValueError, match="missing content-addressed evidence"):
        analyze(records, review, registry, archive)

    records, review, registry, archive, rows = _write_inputs(tmp_path / "second")
    rows[0]["target_role"] = "product_operator"
    records.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="registered role mismatch"):
        analyze(records, review, registry, archive)


def test_protocol_cli_is_read_only_and_analysis_requires_explicit_inputs():
    completed = subprocess.run(
        [sys.executable, "scripts/product_validation_study.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["passed"] is True
    rejected = subprocess.run(
        [sys.executable, "scripts/product_validation_study.py", "--records", "x"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "--register or --analyze" in rejected.stderr
