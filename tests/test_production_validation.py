import json
from datetime import datetime, timezone

import pytest

from scripts import run_production_validation as production
from src.infrastructure import FileLease, LeaseUnavailable
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
    recovered_marker = None
    for row in production.build_schedule(protocol):
        task, created = queue.submit_match(
            row["home"], row["away"], fast=True,
            idempotency_key=row["idempotency_key"],
        )
        assert created
        claimed = queue.claim_next("worker-validation")
        if row["index"] == 0:
            recovered_marker = {
                "task_id": claimed["task_id"],
                "idempotency_sha256": ProductTaskQueue._idempotency_hash(
                    row["idempotency_key"],
                ),
                "attempt": claimed["attempt"],
                "claimed_at": claimed["started_at"],
            }
            assert queue.recover_running(reason="synthetic_restart") == 1
            queue.requeue_interrupted(
                task["task_id"], reason="production_validation_resume",
            )
            claimed = queue.claim_next("worker-validation")
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
        json.dumps({
            "schema_version": 1,
            "name": "Production validation",
            "mode": "stable",
            "seed": 91000,
            "matches": matches,
            "runs": runs,
        }),
        encoding="utf-8",
    )
    progress = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "identity": production._identity(production.PROTOCOL_PATH),
        "recovered_running_tasks": 1,
        "last_recovered_task": recovered_marker,
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
        "progress_sha256": production._payload_sha256(progress),
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

    progress["recovered_running_tasks"] = 1
    progress["last_recovered_task"]["idempotency_sha256"] = "0" * 64
    attestation["progress_sha256"] = production._payload_sha256(progress)
    result = production.analyze_completed_run(
        tmp_path, protocol, progress, attestation, recovery,
    )
    assert result["passed"] is False
    assert result["restart_evidence"]["running_task_identity_replayed"] is False
    assert result["gates"]["restart_recovered_without_new_logical_tasks"] is False


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


def test_attestation_is_written_once_and_binds_final_progress(tmp_path):
    protocol, progress, _, recovery = _complete_synthetic_evidence(tmp_path)
    attestation_path = tmp_path / protocol["outputs"]["deployment_attestation"]
    attestation_path.unlink()
    result = production.record_attestation(
        tmp_path,
        operator_id="operator-12345678",
        volume_id="volume-validation",
        container_image_digest="sha256:" + "b" * 64,
        deployment_instance_ids=["instance-first", "instance-second"],
        forced_termination_observed=True,
        same_volume_persisted=True,
    )
    assert result["recorded"] is True
    recorded = production._read_json(attestation_path)
    assert recorded["progress_sha256"] == production._payload_sha256(progress)
    assert all(
        production._validate_attestation(recorded, protocol, progress).values()
    )
    with pytest.raises(FileExistsError, match="immutable"):
        production.record_attestation(
            tmp_path,
            operator_id="operator-12345678",
            volume_id="volume-validation",
            container_image_digest="sha256:" + "b" * 64,
            deployment_instance_ids=["instance-first", "instance-second"],
            forced_termination_observed=True,
            same_volume_persisted=True,
        )

    progress["completed_tasks"] = 99
    result = production.analyze_completed_run(
        tmp_path, protocol, progress, recorded, recovery,
    )
    assert result["passed"] is False
    assert result["attestation_checks"]["final_progress_snapshot_is_bound"] is False


def test_execution_is_mutually_exclusive_with_web_and_other_validator(
    tmp_path, monkeypatch,
):
    token = "I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION"
    monkeypatch.setattr(
        production,
        "_execute_locked",
        lambda root, protocol, instance_id: {"workload_complete": True},
    )
    web_lock, validation_lock = production._lease_paths(tmp_path)
    with FileLease(web_lock):
        with pytest.raises(LeaseUnavailable):
            production.execute(tmp_path, token, "instance-first")
    with FileLease(validation_lock):
        with pytest.raises(LeaseUnavailable):
            production.execute(tmp_path, token, "instance-first")
    assert production.execute(
        tmp_path, token, "instance-first",
    )["workload_complete"] is True


def test_restore_scratch_must_be_existing_external_directory(tmp_path):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()
    assert production._validated_restore_scratch(workspace, scratch) == scratch.resolve()
    inside = workspace / "scratch"
    inside.mkdir()
    with pytest.raises(ValueError, match="outside"):
        production._validated_restore_scratch(workspace, inside)
    with pytest.raises(ValueError, match="existing"):
        production._validated_restore_scratch(workspace, tmp_path / "missing")


def test_finalization_uses_external_scratch_and_writes_one_immutable_decision(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()
    protocol, _, _, _ = _complete_synthetic_evidence(workspace)
    observed = {}

    def exercise(root, selected_protocol, *, restore_scratch=None):
        observed.update({
            "root": root,
            "protocol": selected_protocol["protocol_id"],
            "restore_scratch": restore_scratch,
        })
        return {
            "backup_valid": True,
            "session_sha256_match": True,
            "match_inventory_match": True,
        }

    monkeypatch.setattr(production, "_exercise_backup", exercise)
    monkeypatch.setattr(
        production,
        "analyze_completed_run",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "passed_production_validation",
            "passed": True,
        },
    )
    result = production.finalize(
        workspace,
        "I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION",
        restore_scratch=scratch,
    )
    assert result["passed"] is True
    assert observed == {
        "root": workspace.resolve(),
        "protocol": protocol["protocol_id"],
        "restore_scratch": scratch.resolve(),
    }
    decision_path = workspace / protocol["outputs"]["decision"]
    assert production._read_json(decision_path)["passed"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        production.finalize(
            workspace,
            "I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION",
            restore_scratch=scratch,
        )
