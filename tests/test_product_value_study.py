import hashlib
import json
from datetime import datetime, timedelta

import pytest

from scripts.product_value_study import (
    CASE_PACK_PATH,
    OBSERVER_ATTESTATION,
    PROTOCOL_PATH,
    analyze,
    protocol_report,
)
from src.product.human_study import register_participant


ROLES = ("football_analyst", "research_engineer", "product_operator")
CORRECT = {
    "case_pack_alpha": {
        "recommended_intervention": "wide_overload",
        "supported_claim": "best_frozen_simulator_branch",
        "next_evidence_action": "preregistered_paired_validation",
    },
    "case_pack_beta": {
        "recommended_intervention": "controlled_possession",
        "supported_claim": "best_frozen_simulator_branch",
        "next_evidence_action": "preregistered_paired_validation",
    },
}


def _condition_plan(sequence: str):
    if sequence == "AB":
        return [("gfs_studio", "case_pack_alpha"), ("manual_baseline", "case_pack_beta")]
    return [("manual_baseline", "case_pack_alpha"), ("gfs_studio", "case_pack_beta")]


def _receipt(archive, registration, condition, case_pack, submitted_at, answers):
    value = {
        "schema_version": 1,
        "protocol_id": "gfs-studio-comparative-value-validation-v1",
        "registration_id": registration["registration_id"],
        "participant_id": registration["participant_id"],
        "condition": condition,
        "case_pack": case_pack,
        "answers": answers,
        "submitted_at": submitted_at,
    }
    payload = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    (archive / digest).write_bytes(payload)
    return digest


def _record(index, registration, archive, *, gfs_seconds=400.0):
    started = datetime.fromisoformat(registration["registered_at"]) + timedelta(seconds=1)
    conditions = []
    for offset, (condition, case_pack) in enumerate(
        _condition_plan(registration["sequence"])
    ):
        seconds = gfs_seconds if condition == "gfs_studio" else 800.0
        submitted = (started + timedelta(seconds=600 + offset * 600)).isoformat()
        conditions.append(
            {
                "condition": condition,
                "case_pack": case_pack,
                "completed": True,
                "correct": True,
                "critical_error": False,
                "duration_seconds": seconds,
                "evidence_sha256": _receipt(
                    archive,
                    registration,
                    condition,
                    case_pack,
                    submitted,
                    CORRECT[case_pack],
                ),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": "gfs-studio-comparative-value-validation-v1",
        "registration_id": registration["registration_id"],
        "participant_id": registration["participant_id"],
        "target_role": registration["target_role"],
        "consent": True,
        "moderator_id": registration["moderator_id"],
        "sequence": registration["sequence"],
        "session_started_at": started.isoformat(),
        "session_ended_at": (started + timedelta(seconds=1800)).isoformat(),
        "conditions": conditions,
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": OBSERVER_ATTESTATION,
    }


def _write_study(tmp_path, *, gfs_seconds=400.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "registry.json"
    records = tmp_path / "records.jsonl"
    archive = tmp_path / "evidence"
    archive.mkdir()
    registrations = []
    for index in range(24):
        result = register_participant(
            project_root=tmp_path,
            protocol_path=PROTOCOL_PATH,
            registry_path=registry,
            records_path=records,
            participant_id=f"participant-{index:08d}",
            target_role=ROLES[index % len(ROLES)],
            moderator_id="moderator-12345678",
            consent_recorded=True,
            case_pack_manifest_path=CASE_PACK_PATH,
        )
        registrations.append(result["registration"])
    rows = [
        _record(index, registration, archive, gfs_seconds=gfs_seconds)
        for index, registration in enumerate(registrations)
    ]
    records.write_text(
        "".join(json.dumps(record) + "\n" for record in rows),
        encoding="utf-8",
    )
    return records, registry, archive, rows, registrations


def test_product_value_protocol_and_case_packs_are_frozen_and_unexecuted():
    report = protocol_report()
    assert report["passed"] is True
    assert report["study_executed"] is False
    assert report["participants_observed"] == 0
    assert report["matches_executed"] == 0
    assert all(report["checks"].values())


def test_product_value_analysis_passes_registered_rescored_evidence(tmp_path):
    records, registry, archive, _, _ = _write_study(tmp_path)
    result = analyze(records, registry, archive)
    assert result["passed"] is True
    assert result["minimum_sample_met"] is True
    assert result["primary_value_threshold_met"] is True
    assert result["role_counts"] == {
        "football_analyst": 8,
        "product_operator": 8,
        "research_engineer": 8,
    }
    assert result["sequence_counts"] == {"AB": 12, "BA": 12}
    assert set(result["role_sequence_counts"].values()) == {4}
    assert result["metrics"][
        "paired_penalized_time_ratio_bootstrap_95"
    ] == [0.5, 0.5]
    assert result["evidence_archive"]["verified_artifacts"] == 48
    assert result["registry_checks"]["all_registrations_analyzed"] is True


def test_product_value_analysis_preserves_negative_result(tmp_path):
    records, registry, archive, _, _ = _write_study(
        tmp_path,
        gfs_seconds=850.0,
    )
    result = analyze(records, registry, archive)
    assert result["passed"] is False
    assert result["minimum_sample_met"] is True
    assert result["primary_value_threshold_met"] is False
    assert result["gates"]["primary_value_threshold"] is False


def test_product_value_rejects_unknown_fields_and_registry_drift(tmp_path):
    records, registry, archive, rows, _ = _write_study(tmp_path)
    rows[0]["free_text"] = "must not be retained"
    records.write_text(
        "".join(json.dumps(record) + "\n" for record in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="privacy allowlist"):
        analyze(records, registry, archive)

    records, registry, archive, rows, _ = _write_study(tmp_path / "drift")
    rows[0]["sequence"] = "BA" if rows[0]["sequence"] == "AB" else "AB"
    records.write_text(
        "".join(json.dumps(record) + "\n" for record in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="condition order|registered sequence"):
        analyze(records, registry, archive)


def test_product_value_rejects_fake_digest_and_false_correctness(tmp_path):
    records, registry, archive, rows, _ = _write_study(tmp_path)
    digest = rows[0]["conditions"][0]["evidence_sha256"]
    (archive / digest).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content hash mismatch"):
        analyze(records, registry, archive)

    records, registry, archive, rows, _ = _write_study(tmp_path / "correctness")
    rows[0]["conditions"][0]["correct"] = False
    records.write_text(
        "".join(json.dumps(record) + "\n" for record in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared correctness disagrees"):
        analyze(records, registry, archive)


def test_role_stratified_blocks_are_balanced_and_registration_is_idempotent(tmp_path):
    _, registry, _, _, registrations = _write_study(tmp_path)
    by_role = {role: [] for role in ROLES}
    for registration in registrations:
        by_role[registration["target_role"]].append(registration["sequence"])
    assert all(values.count("AB") == values.count("BA") == 4 for values in by_role.values())
    result = register_participant(
        project_root=tmp_path,
        protocol_path=PROTOCOL_PATH,
        registry_path=registry,
        records_path=tmp_path / "empty-records.jsonl",
        participant_id=registrations[0]["participant_id"],
        target_role=registrations[0]["target_role"],
        moderator_id=registrations[0]["moderator_id"],
        consent_recorded=True,
        case_pack_manifest_path=CASE_PACK_PATH,
    )
    assert result["created"] is False
    assert result["registration"] == registrations[0]
