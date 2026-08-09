import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.product_value_study import (
    OBSERVER_ATTESTATION,
    analyze,
    protocol_report,
)


def _record(index: int, *, gfs_seconds: float = 400.0) -> dict:
    sequence = "AB" if index < 12 else "BA"
    role = ("football_analyst", "research_engineer", "product_operator")[index % 3]
    plan = (
        [("gfs_studio", "case_pack_alpha", gfs_seconds), ("manual_baseline", "case_pack_beta", 800.0)]
        if sequence == "AB"
        else [("manual_baseline", "case_pack_alpha", 800.0), ("gfs_studio", "case_pack_beta", gfs_seconds)]
    )
    started = datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc) + timedelta(hours=index)
    return {
        "schema_version": 1,
        "protocol_id": "gfs-studio-comparative-value-validation-v1",
        "participant_id": f"participant-{index:08d}",
        "target_role": role,
        "consent": True,
        "moderator_id": "moderator-12345678",
        "sequence": sequence,
        "session_started_at": started.isoformat(),
        "session_ended_at": (started + timedelta(seconds=1800)).isoformat(),
        "conditions": [
            {
                "condition": condition,
                "case_pack": case,
                "completed": True,
                "correct": True,
                "critical_error": False,
                "duration_seconds": seconds,
                "evidence_sha256": f"{index + offset + 1:064x}",
            }
            for offset, (condition, case, seconds) in enumerate(plan)
        ],
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": OBSERVER_ATTESTATION,
    }


def _write_records(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_product_value_protocol_is_preregistered_and_unexecuted():
    report = protocol_report()
    assert report["passed"] is True
    assert report["study_executed"] is False
    assert report["participants_observed"] == 0
    assert report["matches_executed"] == 0
    assert all(report["checks"].values())


def test_product_value_analysis_passes_only_strict_comparative_evidence(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, [_record(index) for index in range(24)])
    result = analyze(records)
    assert result["passed"] is True
    assert result["minimum_sample_met"] is True
    assert result["primary_value_threshold_met"] is True
    assert result["role_counts"] == {
        "football_analyst": 8,
        "product_operator": 8,
        "research_engineer": 8,
    }
    assert result["sequence_counts"] == {"AB": 12, "BA": 12}
    assert result["metrics"][
        "paired_penalized_time_ratio_bootstrap_95"
    ] == [0.5, 0.5]
    assert result["participants_observed_by_analyzer"] == 0
    assert result["matches_executed_by_analyzer"] == 0


def test_product_value_analysis_preserves_negative_result(tmp_path):
    records = tmp_path / "negative.jsonl"
    _write_records(records, [_record(index, gfs_seconds=850.0) for index in range(24)])
    result = analyze(records)
    assert result["passed"] is False
    assert result["minimum_sample_met"] is True
    assert result["primary_value_threshold_met"] is False
    assert result["gates"]["primary_value_threshold"] is False


def test_product_value_records_fail_closed_on_unknown_or_duplicate_identity(tmp_path):
    records = [_record(index) for index in range(24)]
    records[0]["free_text"] = "must not be retained"
    path = tmp_path / "unknown.jsonl"
    _write_records(path, records)
    with pytest.raises(ValueError, match="privacy allowlist"):
        analyze(path)

    records = [_record(index) for index in range(24)]
    records[1]["participant_id"] = records[0]["participant_id"]
    path = tmp_path / "duplicate.jsonl"
    _write_records(path, records)
    with pytest.raises(ValueError, match="duplicate participant_id"):
        analyze(path)
