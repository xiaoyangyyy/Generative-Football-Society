import json
from datetime import datetime, timezone

from scripts import run_production_validation as production
from src.product.tasks import ProductTaskQueue


def test_production_protocol_freezes_exactly_100_offline_matches():
    report = production.protocol_report()
    protocol = production._read_json(production.PROTOCOL_PATH)
    schedule = production.build_schedule(protocol)
    assert report["passed"] is True
    assert report["study_executed"] is False
    assert report["matches_executed"] == 0
    assert len(schedule) == 100
    assert [row["seed"] for row in schedule] == list(range(91000, 91100))
    assert len({row["idempotency_key"] for row in schedule}) == 100
    assert all(report["checks"].values())


def _complete_synthetic_evidence(tmp_path):
    protocol = production._read_json(production.PROTOCOL_PATH)
    queue = ProductTaskQueue(tmp_path)
    matches = []
    runs = []
    for row in production.build_schedule(protocol):
        task, created = queue.submit_match(
            row["home"], row["away"], fast=True,
            idempotency_key=row["idempotency_key"],
        )
        assert created
        queue.claim_next("worker-validation")
        match_id = f"{row['index'] + 1:04d}-synthetic"
        relative = f"outputs/studio/validation/matches/{match_id}.json"
        report_path = tmp_path / relative
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({
                "match_id": match_id,
                "fixture": {
                    "home": row["home"], "away": row["away"],
                    "seed": row["seed"], "fast": True,
                },
                "integrity": {"accepted": True},
            }),
            encoding="utf-8",
        )
        queue.complete(
            task["task_id"], "worker-validation",
            {"match_id": match_id, "report": relative},
        )
        matches.append({"match_id": match_id})
        runs.append({"match_id": match_id, "successful_provider_calls": 0})
    session_path = tmp_path / "data/persistence/product_session.json"
    session_path.write_text(
        json.dumps({"schema_version": 1, "mode": "stable", "matches": matches, "runs": runs}),
        encoding="utf-8",
    )
    progress = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "identity": production._identity(production.PROTOCOL_PATH),
        "recovered_running_tasks": 1,
        "invocations": [
            {"deployment_instance_id": "instance-first"},
            {"deployment_instance_id": "instance-second"},
        ],
    }
    output_root = tmp_path / "data/evaluation/production_validation_v1"
    output_root.mkdir(parents=True)
    (output_root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    attestation = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "operator_id": "operator-12345678",
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "attestation": production.ATTESTATION,
        "volume_id": "volume-validation",
        "container_image_digest": "sha256:" + "a" * 64,
        "deployment_instance_ids": ["instance-first", "instance-second"],
        "forced_termination_observed": True,
        "same_volume_persisted": True,
        "raw_host_identifiers_included": False,
    }
    (output_root / "deployment_attestation.json").write_text(
        json.dumps(attestation), encoding="utf-8",
    )
    (output_root / "studio-backup.zip").write_bytes(b"test-addressed-placeholder")
    recovery = {
        "backup_valid": True,
        "session_sha256_match": True,
        "match_inventory_match": True,
    }
    return protocol, progress, attestation, recovery


def test_production_decision_requires_zero_loss_duplicate_and_real_restart(tmp_path):
    protocol, progress, attestation, recovery = _complete_synthetic_evidence(tmp_path)
    result = production.analyze_completed_run(
        tmp_path, protocol, progress, attestation, recovery,
    )
    assert result["passed"] is True
    assert result["matches_executed"] == 100
    assert result["zero_lost_transactions"] is True
    assert result["zero_duplicate_transactions"] is True
    assert result["deployment_volume_recovery_drill_passed"] is True
    assert result["successful_provider_calls"] == 0
    assert result["provider_calls_made"] is False

    progress["recovered_running_tasks"] = 0
    result = production.analyze_completed_run(
        tmp_path, protocol, progress, attestation, recovery,
    )
    assert result["passed"] is False
    assert result["deployment_volume_recovery_drill_passed"] is False


def test_production_attestation_rejects_unknown_privacy_fields(tmp_path):
    protocol, progress, attestation, recovery = _complete_synthetic_evidence(tmp_path)
    attestation["host_name"] = "private-host"
    result = production.analyze_completed_run(
        tmp_path, protocol, progress, attestation, recovery,
    )
    assert result["passed"] is False
    assert result["attestation_checks"]["exact_privacy_schema"] is False


def test_production_decision_rejects_fixture_or_seed_drift(tmp_path):
    protocol, progress, attestation, recovery = _complete_synthetic_evidence(tmp_path)
    report = tmp_path / "outputs/studio/validation/matches/0001-synthetic.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["fixture"]["seed"] += 1
    report.write_text(json.dumps(payload), encoding="utf-8")
    result = production.analyze_completed_run(
        tmp_path, protocol, progress, attestation, recovery,
    )
    assert result["passed"] is False
    assert result["gates"]["frozen_fixture_and_seed_schedule_executed"] is False
