import json
import zipfile
from pathlib import Path

import pytest

from gfs_studio.api import backup_create, backup_restore, backup_verify, web_gateway
from gfs_studio.plugins.product.recovery.service import BackupService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_backup_verify_restore_and_explicit_replace(tmp_path):
    service = BackupService(PROJECT_ROOT)
    created = service.create()
    backup_id = created["backup_id"]
    assert len(backup_id) == 24
    verified = service.verify(backup_id)
    assert verified["verified"] is True
    assert verified["files"] > 0
    destination = tmp_path / "restored"
    result = service.restore(backup_id, destination)
    assert result["restored"] is True
    assert (destination / "evidence" / "manifest.toml").is_file()
    with pytest.raises(FileExistsError):
        service.restore(backup_id, destination)
    replaced = service.restore(backup_id, destination, replace=True)
    assert replaced["replace"] is True


def test_backup_api_matches_service(tmp_path):
    created = backup_create(PROJECT_ROOT)
    assert backup_verify(PROJECT_ROOT, created["backup_id"])["verified"] is True
    destination = tmp_path / "api-restored"
    result = backup_restore(PROJECT_ROOT, created["backup_id"], destination)
    assert result["restored"] is True


def test_backup_rejects_path_traversal(tmp_path):
    backup_id = "a" * 24
    path = tmp_path / f"{backup_id}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("backup.json", json.dumps({"files": []}))
        archive.writestr("../escape", b"bad")
    service = BackupService(tmp_path)
    service.backup_dir = tmp_path
    with pytest.raises(ValueError, match="unsafe"):
        service.verify(backup_id)


def test_backup_http_create_verify_restore(tmp_path):
    gateway = web_gateway(PROJECT_ROOT)
    created_response = gateway.handle("POST", "/api/backups", host="127.0.0.1")
    assert created_response.status == 200
    created = json.loads(created_response.body)
    verified_response = gateway.handle(
        "POST",
        "/api/backups/verify",
        host="127.0.0.1",
        body=json.dumps({"backup_id": created["backup_id"]}).encode(),
    )
    assert verified_response.status == 200
    destination = tmp_path / "http-restored"
    restored_response = gateway.handle(
        "POST",
        "/api/backups/restore",
        host="127.0.0.1",
        body=json.dumps({
            "backup_id": created["backup_id"],
            "destination": str(destination),
            "replace": False,
        }).encode(),
    )
    assert restored_response.status == 200
    assert json.loads(restored_response.body)["restored"] is True
    gateway.close()
