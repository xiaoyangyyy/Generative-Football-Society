import hashlib
import json
import os
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from src.cli import (
    cmd_studio_backup,
    cmd_studio_recover,
    cmd_studio_restore,
    cmd_studio_verify_backup,
)
from src.infrastructure import FileLease, LeaseUnavailable
from src.product.recovery import MANIFEST_NAME, ProductRecovery
from src.product.tasks import ProductTaskQueue


def _write_workspace(root, *, name="Backup Studio"):
    report = root / "outputs/studio/backup-studio/matches/0001.json"
    dashboard = report.with_suffix(".html")
    ball_log = root / "outputs/ball_log/0001.jsonl"
    cognitive = root / "data/persistence/cognitive_log/studio_0001.json"
    for path, content in (
        (report, json.dumps({
            "artifacts": {
                "ball_log": ball_log.relative_to(root).as_posix(),
                "cognitive_log": cognitive.relative_to(root).as_posix(),
            },
        })),
        (dashboard, "<html><body>auditable report</body></html>"),
        (ball_log, '{"event":"kickoff"}\n'),
        (cognitive, json.dumps({"calls": 1})),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    session = root / "data/persistence/product_session.json"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text(json.dumps({
        "schema_version": 1,
        "product": "Generative Football Society Studio",
        "name": name,
        "slug": "backup-studio",
        "mode": "cognitive",
        "seed": 42,
        "matches": [{
            "match_id": "0001",
            "report": report.relative_to(root).as_posix(),
            "dashboard": dashboard.relative_to(root).as_posix(),
        }],
    }), encoding="utf-8")
    return session, report, dashboard, ball_log, cognitive


def _crash_restore(root: Path, bundle: Path, *, after_staged_switches: int):
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
    return subprocess.run(
        [
            sys.executable, "-c", code, str(root), str(bundle),
            str(after_staged_switches),
        ],
        cwd=Path(__file__).resolve().parents[1],
        timeout=30,
        check=False,
    )


def test_backup_verifies_and_restores_every_referenced_product_artifact(tmp_path):
    paths = _write_workspace(tmp_path)
    bundle = tmp_path / "backups/studio.zip"
    created = ProductRecovery(tmp_path).create_backup(bundle)
    assert created["valid"]
    assert created["bundle"] == str(bundle.resolve())
    assert created["file_count"] == len(paths)
    verified = ProductRecovery.verify_backup(bundle)
    assert verified["studio"]["name"] == "Backup Studio"

    expected = {path: path.read_bytes() for path in paths}
    for path in paths:
        path.unlink()
    restored = ProductRecovery(tmp_path).restore_backup(bundle)
    assert restored["restored"]
    assert {path: path.read_bytes() for path in paths} == expected
    telemetry = ProductRecovery(tmp_path).telemetry
    operations = telemetry.snapshot()["recovery"]
    assert operations["backup_created"] == 1
    assert operations["restore_completed"] == 1
    raw = telemetry.path.read_text(encoding="utf-8")
    assert "Backup Studio" not in raw
    assert str(bundle) not in raw


def test_restore_requires_explicit_replace_for_existing_session(tmp_path):
    session, *_ = _write_workspace(tmp_path)
    bundle = tmp_path / "backup.zip"
    ProductRecovery(tmp_path).create_backup(bundle)
    session.write_text(session.read_text(encoding="utf-8").replace(
        "Backup Studio", "Current Studio",
    ), encoding="utf-8")
    with pytest.raises(FileExistsError, match="--replace"):
        ProductRecovery(tmp_path).restore_backup(bundle)
    ProductRecovery(tmp_path).restore_backup(bundle, replace=True)
    assert json.loads(session.read_text(encoding="utf-8"))["name"] == "Backup Studio"


def test_restore_without_replace_preserves_orphan_product_artifact(tmp_path):
    paths = _write_workspace(tmp_path)
    bundle = tmp_path / "backup.zip"
    recovery = ProductRecovery(tmp_path)
    recovery.create_backup(bundle)
    for path in paths:
        path.unlink()
    orphan = paths[1]
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"unrelated-existing-report")

    with pytest.raises(FileExistsError, match="restore target exists"):
        recovery.restore_backup(bundle)
    assert orphan.read_bytes() == b"unrelated-existing-report"
    assert not recovery.restore_transaction_root.exists()


def test_restore_rejects_bundle_changed_after_verification(
    tmp_path, monkeypatch,
):
    paths = _write_workspace(tmp_path)
    bundle = tmp_path / "backup.zip"
    recovery = ProductRecovery(tmp_path)
    recovery.create_backup(bundle)
    before = {path: path.read_bytes() for path in paths}
    real_verify = ProductRecovery.verify_backup

    def verify_then_change(value):
        result = real_verify(value)
        with Path(value).open("ab") as handle:
            handle.write(b"changed-after-verification")
        return result

    monkeypatch.setattr(
        ProductRecovery, "verify_backup", staticmethod(verify_then_change),
    )
    with pytest.raises(RuntimeError, match="changed during verification"):
        recovery.restore_backup(bundle, replace=True)
    assert {path: path.read_bytes() for path in paths} == before
    assert not recovery.restore_transaction_root.exists()


def test_restore_target_rejects_symbolic_link_ancestors(
    tmp_path, monkeypatch,
):
    recovery = ProductRecovery(tmp_path)
    link = tmp_path / "outputs/studio/redirected"
    real_is_symlink = Path.is_symlink

    def injected_symlink(path):
        return path == link or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", injected_symlink)

    with pytest.raises(ValueError, match="must not use symbolic links"):
        recovery._safe_restore_target(
            "outputs/studio/redirected/report.json"
        )


def test_managed_backup_catalog_never_accepts_or_exposes_arbitrary_paths(tmp_path):
    session, *_ = _write_workspace(tmp_path)
    recovery = ProductRecovery(tmp_path)
    created = recovery.create_managed_backup()
    backup_id = created["backup_id"]
    assert created["operation"] == "create" and created["valid"]
    assert str(tmp_path) not in json.dumps(created)
    catalog = recovery.list_managed_backups()
    assert catalog["count"] == 1 and catalog["capacity"] == 50
    assert catalog["backups"][0]["backup_id"] == backup_id
    assert str(tmp_path) not in json.dumps(catalog)
    verified = recovery.verify_managed_backup(backup_id)
    assert verified["operation"] == "verify" and verified["valid"]

    session.write_text(session.read_text(encoding="utf-8").replace(
        "Backup Studio", "Changed Studio",
    ), encoding="utf-8")
    restored = recovery.restore_managed_backup(backup_id, replace=True)
    assert restored["restored"] and restored["task_history_reset"]
    assert json.loads(session.read_text(encoding="utf-8"))["name"] == "Backup Studio"
    for value in ("../escape", "backup.zip", "backup-20260809t000000z-deadbeeg"):
        with pytest.raises((ValueError, FileNotFoundError)):
            recovery.verify_managed_backup(value)


def test_managed_backup_capacity_is_bounded(tmp_path, monkeypatch):
    _write_workspace(tmp_path)
    monkeypatch.setattr("src.product.recovery.MAX_MANAGED_BACKUPS", 2)
    recovery = ProductRecovery(tmp_path)
    recovery.create_managed_backup()
    recovery.create_managed_backup()
    with pytest.raises(RuntimeError, match="capacity"):
        recovery.create_managed_backup()


def test_managed_backup_directory_identity_cannot_change(tmp_path, monkeypatch):
    _write_workspace(tmp_path)
    recovery = ProductRecovery(tmp_path)
    outside = tmp_path.parent / "outside-managed-backups"
    real_resolve = Path.resolve

    def redirected_resolve(path, *args, **kwargs):
        if path == recovery._managed_dir_input:
            return outside
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    with pytest.raises(ValueError, match="escapes"):
        recovery.list_managed_backups()


def test_restore_rolls_back_all_files_when_final_session_switch_fails(
    tmp_path, monkeypatch,
):
    paths = _write_workspace(tmp_path, name="Backup Studio")
    bundle = tmp_path / "backup.zip"
    ProductRecovery(tmp_path).create_backup(bundle)
    for path in paths:
        path.write_text("current-" + path.name, encoding="utf-8")
    before = {path: path.read_bytes() for path in paths}
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_match("France", "Spain", fast=True)
    claimed = queue.claim_next("finished-worker")
    queue.complete(claimed["task_id"], "finished-worker", {"match_id": "old"})
    task_queue_before = queue.path.read_bytes()
    real_replace = os.replace

    def fail_session_switch(source, target):
        if "staged" in Path(source).parts and str(target).endswith(
            "product_session.json",
        ):
            raise OSError("injected final switch failure")
        return real_replace(source, target)

    monkeypatch.setattr("src.product.recovery.os.replace", fail_session_switch)
    with pytest.raises(OSError, match="injected"):
        ProductRecovery(tmp_path).restore_backup(bundle, replace=True)
    assert {path: path.read_bytes() for path in paths} == before
    assert queue.path.read_bytes() == task_queue_before
    assert queue.get_task(task["task_id"])["state"] == "completed"
    assert ProductRecovery(tmp_path).telemetry.snapshot()["recovery"][
        "restore_completed"
    ] == 0


def test_process_crash_mid_restore_is_rolled_back_from_durable_journal(tmp_path):
    paths = _write_workspace(tmp_path)
    bundle = tmp_path / "backup.zip"
    ProductRecovery(tmp_path).create_backup(bundle)
    for path in paths:
        path.write_bytes(("current-" + path.name).encode("utf-8"))
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_match("France", "Spain", fast=True)
    claimed = queue.claim_next("finished-worker")
    queue.complete(claimed["task_id"], "finished-worker", {"match_id": "old"})
    before = {path: path.read_bytes() for path in (*paths, queue.path)}

    crashed = _crash_restore(tmp_path, bundle, after_staged_switches=1)
    assert crashed.returncode == 91
    recovery = ProductRecovery(tmp_path)
    assert recovery.restore_transaction_root.is_dir()
    recovered = recovery.recover_interrupted_restore()
    assert recovered["outcome"] == "rolled_back_interrupted_restore"
    assert {path: path.read_bytes() for path in (*paths, queue.path)} == before
    assert not recovery.restore_transaction_root.exists()
    assert recovery.recover_interrupted_restore() == {
        "recovered": False, "outcome": "no_interrupted_restore",
    }


def test_process_crash_after_all_switches_completes_committed_restore(tmp_path):
    paths = _write_workspace(tmp_path)
    bundle = tmp_path / "backup.zip"
    ProductRecovery(tmp_path).create_backup(bundle)
    backup_state = {path: path.read_bytes() for path in paths}
    for path in paths:
        path.write_bytes(("current-" + path.name).encode("utf-8"))
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_match("France", "Spain", fast=True)
    claimed = queue.claim_next("finished-worker")
    queue.complete(claimed["task_id"], "finished-worker", {"match_id": "old"})

    crashed = _crash_restore(
        tmp_path, bundle, after_staged_switches=len(paths) + 1,
    )
    assert crashed.returncode == 91
    recovery = ProductRecovery(tmp_path)
    recovered = recovery.recover_interrupted_restore()
    assert recovered["outcome"] == "completed_committed_restore"
    assert {path: path.read_bytes() for path in paths} == backup_state
    assert ProductTaskQueue(tmp_path).list_tasks() == []
    assert not recovery.restore_transaction_root.exists()


def test_atomic_queue_replace_failure_preserves_previous_document(
    tmp_path, monkeypatch,
):
    queue = ProductTaskQueue(tmp_path)
    queue.submit_match("France", "Spain", fast=True)
    before = queue.path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("injected atomic replacement failure")

    monkeypatch.setattr("src.product.workspace.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replacement"):
        queue.submit_match("Brazil", "Argentina", fast=True)
    assert queue.path.read_bytes() == before
    assert not list(queue.path.parent.glob(".product_tasks.json-*.tmp"))


def test_unverifiable_interrupted_restore_fails_closed_without_deletion(tmp_path):
    recovery = ProductRecovery(tmp_path)
    journal = recovery.restore_transaction_root / "journal.json"
    journal.parent.mkdir(parents=True)
    journal.write_text('{"schema_version": 1, "files": []}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="journal is invalid"):
        recovery.recover_interrupted_restore()
    assert journal.is_file()


def test_verifier_rejects_untracked_and_traversal_members(tmp_path):
    _write_workspace(tmp_path)
    bundle = tmp_path / "backup.zip"
    ProductRecovery(tmp_path).create_backup(bundle)
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("outputs/studio/untracked.html", "surprise")
    with pytest.raises(ValueError, match="untracked"):
        ProductRecovery.verify_backup(bundle)

    malicious = tmp_path / "malicious.zip"
    content = b"escape"
    manifest = {
        "schema_version": 1,
        "file_count": 1,
        "files": [{
            "path": "../escape.txt", "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }],
    }
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../escape.txt", content)
        archive.writestr(MANIFEST_NAME, json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        ProductRecovery.verify_backup(malicious)


def test_verifier_rejects_semantically_invalid_studio_session(tmp_path):
    bundle = tmp_path / "invalid-session.zip"
    session = json.dumps({
        "schema_version": 1, "name": "Invalid", "slug": "invalid",
        "mode": "unsupported", "seed": 42, "matches": [],
    }).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "studio": {
            "name": "Invalid", "slug": "invalid", "mode": "unsupported", "seed": 42,
        },
        "file_count": 1,
        "files": [{
            "path": "data/persistence/product_session.json",
            "size": len(session), "sha256": hashlib.sha256(session).hexdigest(),
        }],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("data/persistence/product_session.json", session)
        archive.writestr(MANIFEST_NAME, json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration"):
        ProductRecovery.verify_backup(bundle)


def test_backup_refuses_to_race_an_active_studio_transaction(tmp_path):
    _write_workspace(tmp_path)
    recovery = ProductRecovery(tmp_path)
    with FileLease(recovery.lease_path):
        with pytest.raises(LeaseUnavailable):
            recovery.create_backup(tmp_path / "busy.zip")


def test_restore_rejects_active_tasks_and_resets_terminal_history(tmp_path):
    _write_workspace(tmp_path)
    recovery = ProductRecovery(tmp_path)
    bundle = tmp_path / "backup.zip"
    recovery.create_backup(bundle)
    queue = ProductTaskQueue(tmp_path)
    queued, _ = queue.submit_match("Brazil", "Argentina", fast=True)
    with pytest.raises(RuntimeError, match="queued or running"):
        recovery.restore_backup(bundle, replace=True)
    claimed = queue.claim_next("worker")
    assert claimed["task_id"] == queued["task_id"]
    queue.complete(claimed["task_id"], "worker", {"match_id": "old"})
    result = recovery.restore_backup(bundle, replace=True)
    assert result["task_history_reset"]
    assert queue.list_tasks() == []


def test_recovery_cli_roundtrip(tmp_path, capsys):
    _write_workspace(tmp_path)
    bundle = tmp_path / "cli.zip"
    assert cmd_studio_backup(Namespace(
        base_dir=str(tmp_path), out=str(bundle), overwrite=False,
    )) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert cmd_studio_verify_backup(Namespace(bundle=str(bundle))) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert cmd_studio_restore(Namespace(
        base_dir=str(tmp_path), bundle=str(bundle), replace=True,
    )) == 0
    assert json.loads(capsys.readouterr().out)["restored"]
    assert cmd_studio_recover(Namespace(base_dir=str(tmp_path))) == 0
    recovered = json.loads(capsys.readouterr().out)
    assert recovered["outcome"] == "no_interrupted_restore"
