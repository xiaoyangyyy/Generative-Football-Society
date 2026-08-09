import json
import subprocess
import sys

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


def _record(index: int, *, complete: bool = True) -> dict:
    roles = ("football_analyst", "research_engineer", "product_operator")
    task_ids = (
        "create_or_load_studio",
        "interpret_release_readiness",
        "run_stable_match",
        "review_auditable_report",
        "backup_verify_and_recover",
    )
    return {
        "schema_version": 1,
        "protocol_id": "gfs-studio-target-user-validation-v1",
        "participant_id": f"P-{index:012d}",
        "target_role": roles[index % len(roles)],
        "consent": True,
        "session_started_at": "2026-08-10T09:00:00+08:00",
        "session_ended_at": "2026-08-10T10:00:00+08:00",
        "moderator_id": "M-00000001",
        "tasks": [
            {
                "task_id": task_id,
                "completed": complete or task_index > 0,
                "critical_error": False,
                "duration_seconds": 300,
                "assistance_count": 0,
                "evidence_sha256": (
                    f"{index + task_index + 1:064x}"
                    if complete or task_index > 0
                    else None
                ),
            }
            for task_index, task_id in enumerate(task_ids)
        ],
        "sus_responses": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
        "qualitative_tags": ["trust"],
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": OBSERVER_ATTESTATION,
    }


def _external_review() -> dict:
    common = {
        "independent_of_implementation": True,
        "organization": "Independent Review Lab",
        "signed_at": "2026-08-11T09:00:00Z",
        "attestation": REVIEWER_ATTESTATION,
        "artifact_sha256": {"review.pdf": "a" * 64},
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
        },
        "security": {
            **common,
            "reviewer_id": "R-SECURE0001",
            "method": "independent_application_security_review",
        },
        "issue_resolution": {
            "findings_total": 3,
            "unresolved_critical_count": 0,
            "risk_accepted_critical_count": 0,
            "all_critical_findings_closed": True,
        },
    }


def _write_inputs(tmp_path, records, review=None):
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    review_path = tmp_path / "external-review.json"
    review_path.write_text(
        json.dumps(review or _external_review()),
        encoding="utf-8",
    )
    return records_path, review_path


def test_product_validation_protocol_is_frozen_zero_execution():
    report = protocol_report()
    assert report["passed"] is True
    assert report["status"] == "passed_preregistered_not_executed"
    assert report["result_available"] is False
    assert all(report["checks"].values())
    assert report["participants_observed"] == 0
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False


def test_analyzer_passes_only_complete_threshold_and_external_review(tmp_path):
    records_path, review_path = _write_inputs(
        tmp_path,
        [_record(index) for index in range(10)],
    )
    decision = analyze(records_path, review_path)
    assert decision["passed"] is True
    assert decision["participants_valid"] == 10
    assert decision["metrics"]["participant_core_workflow_success_rate"] == 1.0
    assert decision["metrics"]["mean_sus"] == 100.0
    assert all(decision["gates"].values())
    assert all(decision["external_review_checks"].values())
    assert decision["participants_observed_by_analyzer"] == 0
    assert decision["matches_executed_by_analyzer"] == 0


def test_analyzer_fails_closed_below_workflow_threshold(tmp_path):
    records = [_record(index, complete=index >= 2) for index in range(10)]
    records_path, review_path = _write_inputs(tmp_path, records)
    decision = analyze(records_path, review_path)
    assert decision["passed"] is False
    assert decision["metrics"]["participant_core_workflow_success_rate"] == 0.8
    assert decision["gates"]["core_workflow_success_threshold"] is False


def test_record_validation_rejects_missing_evidence_and_posthoc_exclusion():
    protocol = _read_json(PROTOCOL_PATH)
    record = _record(1)
    record["tasks"][0]["evidence_sha256"] = None
    with pytest.raises(ValueError, match="lacks evidence"):
        _validate_record(record, protocol)
    excluded = _record(2)
    excluded["excluded"] = True
    excluded["exclusion_reason"] = "participant_failed_tasks"
    with pytest.raises(ValueError, match="invalid exclusion"):
        _validate_record(excluded, protocol)
    private = _record(3)
    private["raw_notes"] = "must never enter the analysis record"
    with pytest.raises(ValueError, match="privacy allowlist"):
        _validate_record(private, protocol)


def test_external_review_must_be_distinct_and_independent(tmp_path):
    review = _external_review()
    review["security"]["reviewer_id"] = review["accessibility"]["reviewer_id"]
    records_path, review_path = _write_inputs(
        tmp_path,
        [_record(index) for index in range(10)],
        review,
    )
    decision = analyze(records_path, review_path)
    assert decision["passed"] is False
    assert (
        decision["external_review_checks"]["reviewers_are_distinct_and_independent"]
        is False
    )


def test_protocol_cli_is_read_only_and_analysis_requires_explicit_flag():
    completed = subprocess.run(
        [sys.executable, "scripts/product_validation_study.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["passed"] is True
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/product_validation_study.py",
            "--records",
            "records.jsonl",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "--analyze" in rejected.stderr
