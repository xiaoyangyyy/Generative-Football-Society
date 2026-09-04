"""Audit or explicitly execute the frozen 100-match production validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import FileLease, file_sha256  # noqa: E402
from src.product.match_plan import MatchPlan  # noqa: E402
from src.product.recovery import ProductRecovery  # noqa: E402
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue  # noqa: E402
from src.product.workspace import ProductWorkspace, _atomic_json  # noqa: E402


PROTOCOL_PATH = ROOT / "data/evaluation/production_validation_protocol_v1.json"
OPERATOR_ID = re.compile(r"^operator-[a-z0-9]{8,40}$")
VOLUME_ID = re.compile(r"^volume-[a-z0-9][a-z0-9-]{2,50}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
INSTANCE_ID = re.compile(r"^instance-[a-z0-9][a-z0-9-]{2,50}$")
ATTESTATION = (
    "I attest that the listed deployment instances used the same named "
    "persistent volume and that one instance was terminated while a match "
    "task was running."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lease_paths(root: Path) -> tuple[Path, Path]:
    persistence = root / "data/persistence"
    return (
        persistence / "product_web.lock",
        persistence / "production_validation.lock",
    )


def build_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    execution = protocol["execution"]
    fixtures = execution["fixtures"]
    schedule = []
    for round_index in range(int(execution["rounds"])):
        for fixture_index, fixture in enumerate(fixtures):
            index = round_index * len(fixtures) + fixture_index
            schedule.append({
                "index": index,
                "home": fixture["home"],
                "away": fixture["away"],
                "seed": int(execution["base_seed"]) + index,
                "idempotency_key": f"{protocol['protocol_id']}:{index}",
            })
    return schedule


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    execution = protocol.get("execution") or {}
    restart = protocol.get("restart_drill") or {}
    recovery = protocol.get("recovery_drill") or {}
    acceptance = protocol.get("acceptance") or {}
    integrity = protocol.get("integrity") or {}
    amendments = protocol.get("preexecution_amendments") or []
    outputs = protocol.get("outputs") or {}
    current = protocol.get("current_execution") or {}
    try:
        schedule = build_schedule(protocol)
    except (KeyError, TypeError, ValueError):
        schedule = []
    return {
        "schema_and_state_are_frozen": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id") == "gfs-studio-production-validation-v1"
            and protocol.get("state") == "preregistered_not_executed"
        ),
        "schedule_is_exactly_100_unique_matches": (
            execution.get("mode") == "stable"
            and execution.get("fast") is True
            and execution.get("match_count") == 100
            and execution.get("rounds") == 10
            and len(execution.get("fixtures") or []) == 10
            and len(schedule) == 100
            and [row["index"] for row in schedule] == list(range(100))
            and len({row["idempotency_key"] for row in schedule}) == 100
            and [row["seed"] for row in schedule] == list(range(91000, 91100))
        ),
        "execution_requires_explicit_authority_and_is_offline": (
            execution.get("explicit_execution_token")
            == "I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION"
            and execution.get("provider_calls_authorized") is False
            and execution.get("training_authorized") is False
            and execution.get("formal_experiment_authorized") is False
        ),
        "real_restart_is_required": (
            restart.get("ungraceful_termination_required") is True
            and restart.get("minimum_distinct_deployment_instances") == 2
            and restart.get("same_named_volume_required") is True
            and restart.get("at_least_one_running_task_recovered") is True
            and restart.get("interrupted_task_keeps_task_id_and_idempotency_key") is True
            and restart.get("failed_tasks_are_not_silently_retried") is True
        ),
        "backup_restore_is_integrity_checked": all(
            recovery.get(key) is True for key in (
                "integrity_checked_backup_required",
                "restore_to_clean_staging_root",
                "restored_session_sha256_must_match",
                "restored_match_inventory_must_match",
                "deployment_operator_attestation_required",
            )
        ),
        "acceptance_is_zero_loss_zero_duplicate": (
            acceptance.get("matches_executed") == 100
            and acceptance.get("completed_logical_tasks") == 100
            and acceptance.get("failed_logical_tasks") == 0
            and acceptance.get("unique_task_ids") == 100
            and acceptance.get("unique_match_ids") == 100
            and acceptance.get("successful_provider_calls") == 0
            and all(acceptance.get(key) is True for key in (
                "zero_lost_transactions",
                "zero_duplicate_transactions",
                "deployment_volume_recovery_drill_passed",
                "all_report_integrity_accepted",
            ))
        ),
        "audit_trail_and_negative_results_are_preserved": (
            integrity.get("fresh_dedicated_workspace_required") is True
            and integrity.get("protocol_and_runner_hashes_bound_at_first_invocation") is True
            and integrity.get("atomic_progress_journal") is True
            and integrity.get("all_attempts_retained") is True
            and integrity.get("negative_or_incomplete_result_preserved") is True
            and integrity.get("optional_stopping") is False
            and integrity.get("unknown_attestation_fields_forbidden") is True
            and integrity.get("raw_host_identifiers_forbidden") is True
            and integrity.get(
                "workload_attestation_and_finalization_are_separate"
            ) is True
            and integrity.get("attestation_binds_final_progress_sha256") is True
            and integrity.get("writers_share_product_web_lease") is True
            and integrity.get("restore_scratch_outside_workspace_required") is True
            and integrity.get(
                "running_task_observation_persisted_before_match"
            ) is True
        ),
        "preexecution_deployment_amendment_is_explicit": (
            len(amendments) == 1
            and amendments[0].get("amendment_id")
            == "deployable-production-transaction-v1"
            and amendments[0].get("matches_executed_before_amendment") == 0
            and amendments[0].get("results_available_before_amendment") is False
        ),
        "outputs_are_confined": (
            set(outputs) == {"progress", "deployment_attestation", "backup", "decision"}
            and all(
                isinstance(value, str)
                and value.startswith("data/evaluation/production_validation_v1/")
                and ".." not in Path(value).parts
                for value in outputs.values()
            )
        ),
        "execution_remains_zero": (
            current.get("invocations") == 0
            and current.get("matches_executed") == 0
            and current.get("restart_observed") is False
            and current.get("training_executed") is False
            and current.get("provider_calls_made") is False
            and current.get("results_available") is False
        ),
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    return {
        "schema_version": 1,
        "verification": "gfs_production_validation_preregistration",
        "generated_at": _now(),
        "status": "passed_preregistered_not_executed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "execution_authorized": False,
        "study_executed": False,
        "matches_executed": 0,
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/production_validation_protocol_v1.json": file_sha256(protocol_path),
            "scripts/run_production_validation.py": file_sha256(Path(__file__)),
            "docs/PRODUCTION_VALIDATION.md": file_sha256(ROOT / "docs/PRODUCTION_VALIDATION.md"),
        },
        "external_calls_made": False,
        "training_executed": False,
        "provider_calls_made": False,
    }


def _identity(protocol_path: Path) -> dict[str, str]:
    return {
        "protocol": file_sha256(protocol_path),
        "runner": file_sha256(Path(__file__)),
    }


def _validate_attestation(
    attestation: dict[str, Any], protocol: dict[str, Any], progress: dict[str, Any],
) -> dict[str, bool]:
    allowed = {
        "schema_version", "protocol_id", "operator_id", "signed_at",
        "attestation", "volume_id", "container_image_digest",
        "deployment_instance_ids", "forced_termination_observed",
        "same_volume_persisted", "raw_host_identifiers_included",
        "progress_sha256",
    }
    try:
        signed = datetime.fromisoformat(str(attestation.get("signed_at")))
        signed_ok = signed.tzinfo is not None
    except ValueError:
        signed_ok = False
    instances = attestation.get("deployment_instance_ids") or []
    progress_instances = [row.get("deployment_instance_id") for row in progress.get("invocations") or []]
    return {
        "exact_privacy_schema": set(attestation) == allowed,
        "protocol_and_signature_valid": (
            attestation.get("schema_version") == 1
            and attestation.get("protocol_id") == protocol["protocol_id"]
            and OPERATOR_ID.fullmatch(str(attestation.get("operator_id") or "")) is not None
            and signed_ok
            and attestation.get("attestation") == ATTESTATION
        ),
        "named_volume_and_image_are_identified": (
            VOLUME_ID.fullmatch(str(attestation.get("volume_id") or "")) is not None
            and DIGEST.fullmatch(str(attestation.get("container_image_digest") or "")) is not None
        ),
        "distinct_instances_match_progress": (
            isinstance(instances, list)
            and len(instances) >= 2
            and len(instances) == len(set(instances))
            and all(INSTANCE_ID.fullmatch(str(value or "")) for value in instances)
            and set(instances).issubset(set(progress_instances))
        ),
        "final_progress_snapshot_is_bound": (
            re.fullmatch(
                r"[0-9a-f]{64}", str(attestation.get("progress_sha256") or ""),
            ) is not None
            and attestation.get("progress_sha256") == _payload_sha256(progress)
        ),
        "real_restart_and_volume_persistence_attested": (
            attestation.get("forced_termination_observed") is True
            and attestation.get("same_volume_persisted") is True
            and int(progress.get("recovered_running_tasks") or 0) >= 1
        ),
        "raw_host_identifiers_are_excluded": (
            attestation.get("raw_host_identifiers_included") is False
        ),
    }


def _inventory(session: dict[str, Any]) -> list[str]:
    return sorted(str(row.get("match_id") or "") for row in session.get("matches") or [])


def analyze_completed_run(
    root: Path,
    protocol: dict[str, Any],
    progress: dict[str, Any],
    attestation: dict[str, Any],
    recovery_result: dict[str, Any],
) -> dict[str, Any]:
    queue = ProductTaskQueue(root)
    tasks = list(reversed(queue.list_tasks(limit=500)))
    session = _read_json(root / "data/persistence/product_session.json")
    completed = [row for row in tasks if row.get("state") == "completed"]
    failed = [row for row in tasks if row.get("state") == "failed"]
    task_ids = [str(row.get("task_id") or "") for row in completed]
    match_ids = [str((row.get("result") or {}).get("match_id") or "") for row in completed]
    tasks_by_id = {str(row.get("task_id") or ""): row for row in tasks}
    expected_idempotency_by_task_id = {
        str(task.get("task_id") or ""): hashlib.sha256(
            expected["idempotency_key"].encode("utf-8"),
        ).hexdigest()
        for task, expected in zip(tasks, build_schedule(protocol))
    }
    recovered_marker = progress.get("last_recovered_task")
    recovered_task = (
        tasks_by_id.get(str(recovered_marker.get("task_id") or ""))
        if isinstance(recovered_marker, dict)
        else None
    )
    recovered_identity_replayed = (
        isinstance(recovered_marker, dict)
        and isinstance(recovered_task, dict)
        and recovered_task.get("state") == "completed"
        and expected_idempotency_by_task_id.get(
            str(recovered_marker.get("task_id") or ""),
        ) == recovered_marker.get("idempotency_sha256")
        and isinstance(recovered_marker.get("attempt"), int)
        and not isinstance(recovered_marker.get("attempt"), bool)
        and int(recovered_task.get("attempt") or 0)
        == int(recovered_marker["attempt"]) + 1
        and isinstance(recovered_marker.get("claimed_at"), str)
        and bool(recovered_marker["claimed_at"])
    )
    reports = []
    for task in completed:
        relative = str((task.get("result") or {}).get("report") or "")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("task report escapes validation workspace") from exc
        reports.append(_read_json(candidate))
    schedule = build_schedule(protocol)
    workload_matches = len(tasks) == len(schedule) and all(
        task.get("request") == {
            "home": expected["home"],
            "away": expected["away"],
            "fast": True,
            "plan": MatchPlan().as_dict(),
            "seed_override": None,
        }
        and report.get("fixture") == {
            "home": expected["home"],
            "away": expected["away"],
            "seed": expected["seed"],
            "fast": True,
        }
        for task, report, expected in zip(tasks, reports, schedule)
    )
    attestation_checks = _validate_attestation(attestation, protocol, progress)
    session_ids = _inventory(session)
    provider_calls = sum(
        int(row.get("successful_provider_calls") or 0)
        for row in session.get("runs") or []
    )
    gates = {
        "exactly_100_completed_logical_tasks": len(completed) == 100 and len(tasks) == 100,
        "frozen_fixture_and_seed_schedule_executed": workload_matches,
        "no_failed_or_active_tasks": len(failed) == 0 and all(row.get("state") == "completed" for row in tasks),
        "unique_task_and_match_ids": len(set(task_ids)) == 100 and len(set(match_ids)) == 100,
        "session_contains_exact_completed_inventory": len(session_ids) == 100 and sorted(match_ids) == session_ids,
        "all_report_integrity_accepted": len(reports) == 100 and all((row.get("integrity") or {}).get("accepted") is True for row in reports),
        "stable_mode_zero_provider_calls": session.get("mode") == "stable" and provider_calls == 0,
        "restart_recovered_without_new_logical_tasks": (
            int(progress.get("recovered_running_tasks") or 0) >= 1
            and recovered_identity_replayed
        ),
        "deployment_attestation_passes": all(attestation_checks.values()),
        "backup_restore_matches_source": all(recovery_result.get(key) is True for key in ("backup_valid", "session_sha256_match", "match_inventory_match")),
        "identity_remained_frozen": progress.get("identity") == _identity(PROTOCOL_PATH),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "generated_at": _now(),
        "status": "passed_production_validation" if passed else "failed_production_validation",
        "passed": passed,
        "matches_executed": len(completed),
        "completed_logical_tasks": len(completed),
        "failed_logical_tasks": len(failed),
        "unique_task_ids": len(set(task_ids)),
        "unique_match_ids": len(set(match_ids)),
        "successful_provider_calls": provider_calls,
        "zero_lost_transactions": gates["exactly_100_completed_logical_tasks"] and gates["session_contains_exact_completed_inventory"],
        "zero_duplicate_transactions": gates["unique_task_and_match_ids"] and len(tasks) == 100,
        "deployment_volume_recovery_drill_passed": gates["restart_recovered_without_new_logical_tasks"] and gates["deployment_attestation_passes"] and gates["backup_restore_matches_source"],
        "gates": gates,
        "attestation_checks": attestation_checks,
        "recovery": recovery_result,
        "restart_evidence": {
            "running_task_identity_replayed": recovered_identity_replayed,
            "recovered_task_id": (
                recovered_marker.get("task_id")
                if isinstance(recovered_marker, dict)
                else None
            ),
        },
        "artifact_sha256": {
            "protocol": file_sha256(PROTOCOL_PATH),
            "runner": file_sha256(Path(__file__)),
            "progress": file_sha256(root / protocol["outputs"]["progress"]),
            "deployment_attestation": file_sha256(root / protocol["outputs"]["deployment_attestation"]),
            "backup": file_sha256(root / protocol["outputs"]["backup"]),
        },
        "training_executed": False,
        "provider_calls_made": provider_calls > 0,
        "limitations": [
            "This validates the frozen stable fast-match workload, not every possible fixture or hardware profile.",
            "The operator attestation is integrity-addressed but does not cryptographically prove the operator's real-world identity.",
        ],
    }


def _validated_restore_scratch(
    root: Path, restore_scratch: Path | None,
) -> Path | None:
    if restore_scratch is None:
        return None
    candidate = Path(restore_scratch)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("restore scratch must be an existing non-symlink directory")
    resolved = candidate.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("restore scratch must be outside the validation workspace")
    return resolved


def _exercise_backup(
    root: Path,
    protocol: dict[str, Any],
    *,
    restore_scratch: Path | None = None,
) -> dict[str, Any]:
    restore_parent = _validated_restore_scratch(root, restore_scratch)
    backup_path = root / protocol["outputs"]["backup"]
    recovery = ProductRecovery(root)
    started = time.monotonic()
    created = recovery.create_backup(backup_path, overwrite=True)
    verified = ProductRecovery.verify_backup(backup_path)
    source_session = root / "data/persistence/product_session.json"
    source_hash = file_sha256(source_session)
    source_inventory = _inventory(_read_json(source_session))
    with tempfile.TemporaryDirectory(
        prefix="gfs-production-restore-", dir=restore_parent,
    ) as temporary:
        staged_root = Path(temporary)
        restored = ProductRecovery(staged_root).restore_backup(backup_path, replace=False)
        restored_session = staged_root / "data/persistence/product_session.json"
        restored_hash = file_sha256(restored_session)
        restored_inventory = _inventory(_read_json(restored_session))
    return {
        "backup_valid": created.get("valid") is True and verified.get("valid") is True and restored.get("restored") is True,
        "session_sha256_match": source_hash == restored_hash,
        "match_inventory_match": source_inventory == restored_inventory,
        "file_count": int(verified.get("file_count") or 0),
        "source_session_sha256": source_hash,
        "restored_session_sha256": restored_hash,
        "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        "staging_root": "explicit_external_scratch" if restore_parent else "system_temp",
    }


def _validated_protocol_and_authority(authorization: str) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    if not all(validate_protocol(protocol).values()):
        raise RuntimeError("production validation protocol is invalid or has drifted")
    if authorization != protocol["execution"]["explicit_execution_token"]:
        raise PermissionError("exact explicit execution token is required")
    return protocol


def _validated_workspace(root: Path, protocol: dict[str, Any]) -> ProductWorkspace:
    workspace = ProductWorkspace.load(root)
    if (
        workspace.config.mode != "stable"
        or workspace.config.seed != protocol["execution"]["base_seed"]
    ):
        raise RuntimeError("validation requires a stable Studio seeded at 91000")
    return workspace


def _load_progress(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    progress_path = root / protocol["outputs"]["progress"]
    if not progress_path.is_file():
        raise FileNotFoundError("production validation progress does not exist")
    progress = _read_json(progress_path)
    if (
        progress.get("schema_version") != 1
        or progress.get("protocol_id") != protocol["protocol_id"]
        or progress.get("identity") != _identity(PROTOCOL_PATH)
    ):
        raise RuntimeError("existing production validation progress identity mismatch")
    return progress


def _execute_locked(
    root: Path, protocol: dict[str, Any], instance_id: str,
) -> dict[str, Any]:
    if INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError("deployment instance id must be pseudonymous")
    workspace = _validated_workspace(root, protocol)
    decision_path = root / protocol["outputs"]["decision"]
    if decision_path.exists():
        raise FileExistsError("production validation decision is immutable")
    progress_path = root / protocol["outputs"]["progress"]
    identity = _identity(PROTOCOL_PATH)
    if progress_path.is_file():
        progress = _load_progress(root, protocol)
    else:
        session = workspace._session()
        if session.get("matches") or session.get("runs") or ProductTaskQueue(root).list_tasks(limit=500):
            raise RuntimeError("first invocation requires a fresh dedicated Studio workspace")
        progress = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "identity": identity,
            "started_at": _now(),
            "invocations": [],
            "recovered_running_tasks": 0,
        }
    progress["invocations"].append({
        "invocation_id": f"run-{uuid.uuid4().hex}",
        "deployment_instance_id": instance_id,
        "started_at": _now(),
    })
    queue = ProductTaskQueue(root)
    recovered = queue.recover_running(reason="production_validation_process_restart")
    progress["recovered_running_tasks"] += recovered
    if recovered:
        progress["last_recovered_task"] = progress.get("active_task")
    progress["active_task"] = None
    progress["updated_at"] = _now()
    _atomic_json(progress_path, progress)
    for task in queue.list_tasks(limit=500):
        if task.get("state") == "interrupted":
            queue.requeue_interrupted(task["task_id"], reason="production_validation_resume")
    scheduled_idempotency_by_task_id = {}
    for row in build_schedule(protocol):
        task, _ = queue.submit_match(
            row["home"], row["away"], fast=True,
            idempotency_key=row["idempotency_key"],
        )
        scheduled_idempotency_by_task_id[task["task_id"]] = hashlib.sha256(
            row["idempotency_key"].encode("utf-8"),
        ).hexdigest()
    progress["updated_at"] = _now()
    _atomic_json(progress_path, progress)
    def on_task_claimed(task: dict[str, Any]) -> None:
        progress["active_task"] = {
            "task_id": task["task_id"],
            "idempotency_sha256": scheduled_idempotency_by_task_id[
                task["task_id"]
            ],
            "attempt": int(task.get("attempt") or 0),
            "claimed_at": _now(),
        }
        progress["updated_at"] = _now()
        _atomic_json(progress_path, progress)
        print(json.dumps({
            "event": "production_validation_task_running",
            **progress["active_task"],
        }), flush=True)

    worker = BackgroundMatchWorker(queue, on_task_claimed=on_task_claimed)
    while worker.run_once():
        progress["active_task"] = None
        states = [row["state"] for row in queue.list_tasks(limit=500)]
        progress["completed_tasks"] = states.count("completed")
        progress["failed_tasks"] = states.count("failed")
        progress["updated_at"] = _now()
        _atomic_json(progress_path, progress)
        if "failed" in states:
            break
    states = [row["state"] for row in queue.list_tasks(limit=500)]
    progress["completed_tasks"] = states.count("completed")
    progress["failed_tasks"] = states.count("failed")
    progress["active_task"] = None
    progress["updated_at"] = _now()
    _atomic_json(progress_path, progress)
    workload_complete = (
        len(states) == 100
        and states.count("completed") == 100
        and states.count("failed") == 0
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": (
            "workload_complete_awaiting_attestation"
            if workload_complete
            else "workload_incomplete_preserved"
        ),
        "workload_complete": workload_complete,
        "matches_executed": states.count("completed"),
        "failed_logical_tasks": states.count("failed"),
        "recovered_running_tasks": int(progress["recovered_running_tasks"]),
        "decision_written": False,
        "next_action": (
            "record_deployment_attestation_then_finalize"
            if workload_complete
            else "inspect_failed_or_incomplete_tasks_without_optional_retry"
        ),
        "training_executed": False,
        "provider_calls_made": False,
    }


def execute(root: Path, authorization: str, instance_id: str) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol = _validated_protocol_and_authority(authorization)
    web_lock, validation_lock = _lease_paths(root)
    with FileLease(web_lock, timeout=0.0), FileLease(validation_lock, timeout=0.0):
        return _execute_locked(root, protocol, instance_id)


def _record_attestation_locked(
    root: Path,
    protocol: dict[str, Any],
    *,
    operator_id: str,
    volume_id: str,
    container_image_digest: str,
    deployment_instance_ids: list[str],
    forced_termination_observed: bool,
    same_volume_persisted: bool,
) -> dict[str, Any]:
    _validated_workspace(root, protocol)
    progress = _load_progress(root, protocol)
    attestation_path = root / protocol["outputs"]["deployment_attestation"]
    if attestation_path.exists():
        raise FileExistsError("deployment attestation is immutable once recorded")
    if not forced_termination_observed or not same_volume_persisted:
        raise PermissionError("both real deployment observations must be confirmed")
    attestation = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "operator_id": operator_id,
        "signed_at": _now(),
        "attestation": ATTESTATION,
        "volume_id": volume_id,
        "container_image_digest": container_image_digest,
        "deployment_instance_ids": deployment_instance_ids,
        "forced_termination_observed": True,
        "same_volume_persisted": True,
        "raw_host_identifiers_included": False,
        "progress_sha256": _payload_sha256(progress),
    }
    checks = _validate_attestation(attestation, protocol, progress)
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise ValueError(f"deployment attestation is invalid: {', '.join(failed)}")
    _atomic_json(attestation_path, attestation)
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "deployment_attestation_recorded",
        "recorded": True,
        "progress_sha256": attestation["progress_sha256"],
        "checks": checks,
        "next_action": "finalize_production_validation",
    }


def record_attestation(
    root: Path,
    *,
    operator_id: str,
    volume_id: str,
    container_image_digest: str,
    deployment_instance_ids: list[str],
    forced_termination_observed: bool,
    same_volume_persisted: bool,
) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol = _read_json(PROTOCOL_PATH)
    if not all(validate_protocol(protocol).values()):
        raise RuntimeError("production validation protocol is invalid or has drifted")
    web_lock, validation_lock = _lease_paths(root)
    with FileLease(web_lock, timeout=0.0), FileLease(validation_lock, timeout=0.0):
        return _record_attestation_locked(
            root,
            protocol,
            operator_id=operator_id,
            volume_id=volume_id,
            container_image_digest=container_image_digest,
            deployment_instance_ids=deployment_instance_ids,
            forced_termination_observed=forced_termination_observed,
            same_volume_persisted=same_volume_persisted,
        )


def _finalize_locked(
    root: Path,
    protocol: dict[str, Any],
    *,
    restore_scratch: Path | None,
) -> dict[str, Any]:
    _validated_workspace(root, protocol)
    progress = _load_progress(root, protocol)
    decision_path = root / protocol["outputs"]["decision"]
    if decision_path.exists():
        raise FileExistsError("production validation decision is immutable")
    attestation_path = root / protocol["outputs"]["deployment_attestation"]
    if not attestation_path.is_file():
        raise FileNotFoundError(
            "deployment attestation is required before a decision can be produced",
        )
    recovery_result = _exercise_backup(
        root, protocol, restore_scratch=restore_scratch,
    )
    decision = analyze_completed_run(
        root, protocol, progress, _read_json(attestation_path), recovery_result,
    )
    _atomic_json(decision_path, decision)
    return decision


def finalize(
    root: Path,
    authorization: str,
    *,
    restore_scratch: Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol = _validated_protocol_and_authority(authorization)
    validated_scratch = _validated_restore_scratch(root, restore_scratch)
    web_lock, validation_lock = _lease_paths(root)
    with FileLease(web_lock, timeout=0.0), FileLease(validation_lock, timeout=0.0):
        return _finalize_locked(
            root, protocol, restore_scratch=validated_scratch,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--execute", action="store_true")
    operation.add_argument("--record-attestation", action="store_true")
    operation.add_argument("--finalize", action="store_true")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--authorization", default="")
    parser.add_argument("--deployment-instance-id", default="")
    parser.add_argument("--operator-id", default="")
    parser.add_argument("--volume-id", default="")
    parser.add_argument("--container-image-digest", default="")
    parser.add_argument(
        "--attested-deployment-instance-id", action="append", default=[],
    )
    parser.add_argument("--confirm-forced-termination", action="store_true")
    parser.add_argument("--confirm-same-volume", action="store_true")
    parser.add_argument("--restore-scratch", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not (args.execute or args.record_attestation or args.finalize):
        if any((
            args.authorization,
            args.deployment_instance_id,
            args.operator_id,
            args.volume_id,
            args.container_image_digest,
            args.attested_deployment_instance_id,
            args.confirm_forced_termination,
            args.confirm_same_volume,
            args.restore_scratch,
        )):
            parser.error("operation arguments require an explicit write operation")
        report = protocol_report()
        if args.out:
            _atomic_json(args.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    if args.execute:
        if any((
            args.operator_id,
            args.volume_id,
            args.container_image_digest,
            args.attested_deployment_instance_id,
            args.confirm_forced_termination,
            args.confirm_same_volume,
            args.restore_scratch,
        )):
            parser.error("attestation and restore arguments do not belong to --execute")
        result = execute(
            args.workspace.resolve(), args.authorization,
            args.deployment_instance_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["workload_complete"] else 1
    if args.record_attestation:
        if args.authorization or args.deployment_instance_id or args.restore_scratch:
            parser.error("execution and restore arguments do not belong to --record-attestation")
        result = record_attestation(
            args.workspace.resolve(),
            operator_id=args.operator_id,
            volume_id=args.volume_id,
            container_image_digest=args.container_image_digest,
            deployment_instance_ids=args.attested_deployment_instance_id,
            forced_termination_observed=args.confirm_forced_termination,
            same_volume_persisted=args.confirm_same_volume,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if any((
        args.deployment_instance_id,
        args.operator_id,
        args.volume_id,
        args.container_image_digest,
        args.attested_deployment_instance_id,
        args.confirm_forced_termination,
        args.confirm_same_volume,
    )):
        parser.error("execution and attestation arguments do not belong to --finalize")
    result = finalize(
        args.workspace.resolve(), args.authorization,
        restore_scratch=args.restore_scratch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
