from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
_BACKUP_ID = re.compile(r"^[0-9a-f]{24}$")


class BackupService:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.backup_dir = self.root / "backups"

    def create(self) -> dict:
        files = {}
        for path in _allowlisted_files(self):
            relative = path.relative_to(self.root).as_posix()
            files[relative] = path.read_bytes()
        manifest = {
            "schema_version": 1,
            "files": [
                {"path": name, "sha256": _hash(content)}
                for name, content in sorted(files.items())
            ],
        }
        files["backup.json"] = json.dumps(
            manifest, ensure_ascii=False, indent=2,
        ).encode("utf-8")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
        encoded = buffer.getvalue()
        backup_id = _hash(encoded)[:24]
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        path = self.backup_dir / f"{backup_id}.zip"
        if not path.exists():
            path.write_bytes(encoded)
        return {"backup_id": backup_id, "path": str(path), "files": len(files) - 1}

    def _path(self, backup_id: str) -> Path:
        if not _BACKUP_ID.fullmatch(backup_id):
            raise ValueError("invalid backup id")
        path = self.backup_dir / f"{backup_id}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"backup not found: {backup_id}")
        return path

    def verify(self, backup_id: str) -> dict:
        path = self._path(backup_id)
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("backup ZIP is corrupt")
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)):
                raise ValueError("backup contains duplicate entries")
            if "backup.json" not in names:
                raise ValueError("backup manifest is missing")
            total = 0
            for info in archive.infolist():
                _safe_entry(info.filename)
                if info.flag_bits & 0x1:
                    raise ValueError("encrypted backups are not supported")
                if info.file_size > MAX_FILE_BYTES:
                    raise ValueError("backup entry is oversized")
                total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("backup is oversized")
            manifest = json.loads(archive.read("backup.json"))
            expected = {
                row["path"]: row["sha256"] for row in manifest.get("files", [])
            }
            if set(expected) != set(names) - {"backup.json"}:
                raise ValueError("backup file set does not match manifest")
            for name, digest in expected.items():
                if _hash(archive.read(name)) != digest:
                    raise ValueError(f"backup entry identity mismatch: {name}")
        return {"backup_id": backup_id, "files": len(expected), "verified": True}

    def restore(self, backup_id: str, destination: str | Path, *, replace: bool = False) -> dict:
        verified = self.verify(backup_id)
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self._path(backup_id)) as archive:
            targets = []
            for info in archive.infolist():
                if info.filename == "backup.json":
                    continue
                target = (destination / info.filename).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as error:
                    raise ValueError("backup entry escapes restore root") from error
                if target.is_symlink():
                    raise ValueError("restore target cannot be a symlink")
                if target.exists() and not replace:
                    raise FileExistsError(f"restore target exists: {target}")
                targets.append((info, target))
            for info, target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f".{target.name}.restore-tmp")
                temporary.write_bytes(archive.read(info))
                os.replace(temporary, target)
        return {**verified, "restored": True, "replace": replace}


class ProductRecoveryPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.product.recovery",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.product",
        provides=("product-recovery",),
        requires=("gfs.studio.product.studio",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = project_root
        self.service = BackupService(project_root)

    def activate(self, context: PluginContext) -> None:
        context.provide("product-recovery", self.service)

    def deactivate(self) -> None:
        pass


def _allowlisted_files(self):
    for name in ("data/manifest.toml", "evidence/manifest.toml"):
        path = self.root / name
        if path.is_file():
            yield path
    for root in (self.root / "evidence" / "frozen", self.root / "data" / "world-model"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                yield path


def _safe_entry(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("backup entry path is unsafe")


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
