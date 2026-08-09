import hashlib
import json
import os
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from src.cli import cmd_studio_backup, cmd_studio_restore, cmd_studio_verify_backup
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
