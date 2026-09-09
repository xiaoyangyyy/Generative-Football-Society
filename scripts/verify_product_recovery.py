#!/usr/bin/env python3
"""Verify Studio backup and restore behavior in an isolated temporary workspace."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402
from src.product.recovery import ProductRecovery  # noqa: E402
from src.product.tasks import ProductTaskQueue  # noqa: E402


CODE_IDENTITY_FILES = (
    "src/infrastructure/atomic_io.py",
    "src/product/workspace.py",
    "src/product/workspace_session.py",
    "src/product/workspace_evidence.py",
    "src/product/manager_world_navigator.py",
    "src/product/manager_world_player_changes.py",
    "src/product/manager_world_society_changes.py",
    "src/product/manager_world_dossier.py",
    "src/product/manager_world_story.py",
    "src/product/recovery.py",
    "src/product/web.py",
    "src/cli.py",
    "scripts/verify_product_recovery.py",
)


def _fixture(root: Path) -> tuple[Path, ...]:
    report = root / "outputs/studio/recovery/matches/0001.json"
    dashboard = report.with_suffix(".html")
    ball_log = root / "outputs/ball_log/0001.jsonl"
    cognitive = root / "data/persistence/cognitive_log/studio_0001.json"
    artifacts = {
        "ball_log": ball_log.relative_to(root).as_posix(),
        "cognitive_log": cognitive.relative_to(root).as_posix(),
    }
    contents = {
        report: json.dumps({"artifacts": artifacts}),
        dashboard: "<html><body>recovery verification</body></html>",
        ball_log: '{"event":"kickoff"}\n',
        cognitive: '{"successful_calls":1}\n',
    }
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    session = root / "data/persistence/product_session.json"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text(json.dumps({
        "schema_version": 1,
        "product": "Generative Football Society Studio",
        "name": "Recovery Verification",
        "slug": "recovery",
        "mode": "cognitive",
        "seed": 42,
        "matches": [{
            "match_id": "0001",
            "report": report.relative_to(root).as_posix(),
            "dashboard": dashboard.relative_to(root).as_posix(),
        }],
    }), encoding="utf-8")
    return session, report, dashboard, ball_log, cognitive


def _crash_restore(root: Path, bundle: Path, crash_after: int) -> int:
    code = """
import os
import sys
from pathlib import Path
import src.product.recovery as recovery_module

root = Path(sys.argv[1])
bundle = Path(sys.argv[2])
crash_after = int(sys.argv[3])
real_replace = recovery_module._durable_replace
switched = 0

def crash_after_staged_replace(source, target):
    global switched
    real_replace(source, target)
    parts = Path(source).parts
    if "product_restore_transaction" in parts and "staged" in parts:
        switched += 1
        if switched == crash_after:
            os._exit(91)

recovery_module._durable_replace = crash_after_staged_replace
recovery_module.ProductRecovery(root).restore_backup(bundle, replace=True)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root), str(bundle), str(crash_after)],
        cwd=ROOT,
        timeout=30,
        check=False,
    )
    return completed.returncode


def _prepare_fault_case(root: Path):
    paths = _fixture(root)
    recovery = ProductRecovery(root)
    bundle = root / "backup.zip"
    recovery.create_backup(bundle)
    backup_state = {path: path.read_bytes() for path in paths}
    for path in paths:
        path.write_bytes(("current-" + path.name).encode("utf-8"))
    queue = ProductTaskQueue(root)
    queue.submit_match("France", "Spain", fast=True)
    claimed = queue.claim_next("finished-worker")
    queue.complete(claimed["task_id"], "finished-worker", {"match_id": "old"})
    current_state = {path: path.read_bytes() for path in (*paths, queue.path)}
    return recovery, bundle, paths, backup_state, current_state


def verify_product_recovery() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="gfs-recovery-") as temporary:
        root = Path(temporary)
        paths = _fixture(root)
        original = {path: path.read_bytes() for path in paths}
        bundle = root / "backups/studio.zip"
        recovery = ProductRecovery(root)
        created = recovery.create_backup(bundle)
        checks["backup_created"] = bundle.is_file() and created.get("valid") is True
        checks["all_referenced_files_included"] = created.get("file_count") == len(paths)
        checks["final_bundle_path_reported"] = created.get("bundle") == str(bundle.resolve())

        verified = ProductRecovery.verify_backup(bundle)
        checks["integrity_verification_passed"] = (
            verified.get("valid") is True
            and verified.get("studio", {}).get("name") == "Recovery Verification"
        )

        paths[0].write_text(paths[0].read_text(encoding="utf-8").replace(
            "Recovery Verification", "Current Session",
        ), encoding="utf-8")
        try:
            recovery.restore_backup(bundle)
        except FileExistsError:
            checks["implicit_replacement_rejected"] = True
        else:
            checks["implicit_replacement_rejected"] = False

        queue = ProductTaskQueue(root)
        old_task, _ = queue.submit_match("France", "Spain", fast=True)
        claimed = queue.claim_next("completed-before-restore")
        queue.complete(claimed["task_id"], "completed-before-restore", {
            "match_id": "old-session-match",
        })
        restored = recovery.restore_backup(bundle, replace=True)
        checks["explicit_restore_roundtrip"] = (
            {path: path.read_bytes() for path in paths} == original
        )
        checks["cross_session_task_history_reset"] = (
            restored.get("task_history_reset") is True
            and queue.list_tasks() == []
            and old_task["task_id"] not in json.dumps(queue.list_tasks())
        )

        tampered = root / "backups/tampered.zip"
        tampered.write_bytes(bundle.read_bytes())
        with zipfile.ZipFile(tampered, "a") as archive:
            archive.writestr("outputs/studio/untracked.html", "untracked")
        try:
            ProductRecovery.verify_backup(tampered)
        except ValueError:
            checks["untracked_member_rejected"] = True
        else:
            checks["untracked_member_rejected"] = False

        rollback_root = root / "fault-rollback"
        (
            rollback_recovery,
            rollback_bundle,
            rollback_paths,
            _rollback_backup,
            rollback_current,
        ) = _prepare_fault_case(rollback_root)
        rollback_exit = _crash_restore(rollback_root, rollback_bundle, 1)
        rollback_transaction_persisted = (
            rollback_recovery.restore_transaction_root.is_dir()
        )
        rollback_result = rollback_recovery.recover_interrupted_restore()
        checks["real_process_crash_was_injected"] = rollback_exit == 91
        checks["mid_restore_transaction_was_durable"] = (
            rollback_transaction_persisted
        )
        checks["mid_restore_crash_rolled_back_exactly"] = (
            rollback_result.get("outcome")
            == "rolled_back_interrupted_restore"
            and {
                path: path.read_bytes()
                for path in (*rollback_paths, ProductTaskQueue(rollback_root).path)
            } == rollback_current
        )
        checks["rollback_recovery_is_idempotent"] = (
            rollback_recovery.recover_interrupted_restore().get("outcome")
            == "no_interrupted_restore"
        )

        commit_root = root / "fault-commit"
        (
            commit_recovery,
            commit_bundle,
            commit_paths,
            commit_backup,
            _commit_current,
        ) = _prepare_fault_case(commit_root)
        commit_exit = _crash_restore(
            commit_root, commit_bundle, len(commit_paths) + 1,
        )
        commit_result = commit_recovery.recover_interrupted_restore()
        checks["post_switch_crash_was_injected"] = commit_exit == 91
        checks["fully_switched_restore_was_completed"] = (
            commit_result.get("outcome") == "completed_committed_restore"
            and {path: path.read_bytes() for path in commit_paths} == commit_backup
            and ProductTaskQueue(commit_root).list_tasks() == []
        )
        checks["restore_transaction_residue_removed"] = not any(
            path.is_dir()
            for path in (
                rollback_recovery.restore_transaction_root,
                commit_recovery.restore_transaction_root,
            )
        )

        atomic_queue = ProductTaskQueue(root / "atomic-replace")
        atomic_queue.submit_match("France", "Spain", fast=True)
        atomic_before = atomic_queue.path.read_bytes()
        import src.product.workspace as workspace_module
        real_atomic_replace = workspace_module.os.replace

        def fail_atomic_replace(_source, _target):
            raise OSError("injected atomic replacement failure")

        workspace_module.os.replace = fail_atomic_replace
        try:
            try:
                atomic_queue.submit_match("Brazil", "Argentina", fast=True)
            except OSError:
                atomic_failed = True
            else:
                atomic_failed = False
        finally:
            workspace_module.os.replace = real_atomic_replace
        checks["atomic_replace_failure_preserved_document"] = (
            atomic_failed and atomic_queue.path.read_bytes() == atomic_before
        )
        checks["atomic_replace_failure_cleaned_temporary"] = not list(
            atomic_queue.path.parent.glob(".product_tasks.json-*.tmp")
        )

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_product_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_local_recovery" if passed else "failed",
        "process_crash_verified": bool(
            checks.get("real_process_crash_was_injected")
            and checks.get("mid_restore_crash_rolled_back_exactly")
            and checks.get("fully_switched_restore_was_completed")
        ),
        "passed": passed,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
        "checks": checks,
        "code_sha256": {
            relative: file_sha256(ROOT / relative)
            for relative in CODE_IDENTITY_FILES
        },
        "limitations": [
            "local process termination is not an operational deployment drill, deployment-volume test, or host-power-loss test",
            "backup integrity hashes detect corruption but do not authenticate an untrusted bundle",
            "recovery time and recovery point objectives are not yet measured on production storage",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_product_recovery()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
