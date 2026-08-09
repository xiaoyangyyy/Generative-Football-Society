"""Integrity-checked backup and transactional restore for one Studio session."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from src.infrastructure import FileLease, file_sha256
from src.product.workspace import StudioConfig


RECOVERY_SCHEMA_VERSION = 1
MANIFEST_NAME = "gfs-backup-manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 5 * 1024 * 1024 * 1024
ALLOWED_PATHS = ("data/persistence/product_session.json",)
ALLOWED_PREFIXES = (
    "data/persistence/cognitive_log/",
    "outputs/studio/",
    "outputs/ball_log/",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allowed_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        bool(relative)
        and chr(92) not in relative
        and not path.is_absolute()
        and ".." not in path.parts
        and (relative in ALLOWED_PATHS or relative.startswith(ALLOWED_PREFIXES))
    )


def _stream_hash(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


class ProductRecovery:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.session_path = self.root / ALLOWED_PATHS[0]
        self.lease_path = self.root / "data/persistence/product_session.lock"

    def _relative_file(self, value: Any, *, required: bool) -> str | None:
        if value is None or not str(value).strip():
            if required:
                raise ValueError("backup references an empty required artifact path")
            return None
        raw = Path(str(value))
        candidate = (raw if raw.is_absolute() else self.root / raw).resolve()
        try:
            relative = candidate.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("backup artifact escapes the workspace") from exc
        if not _allowed_path(relative):
            raise ValueError(f"backup artifact is outside the recoverable boundary: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"backup artifact is missing: {relative}")
        return relative

    def _collect_files(self) -> tuple[dict[str, Any], list[str]]:
        if not self.session_path.is_file():
            raise FileNotFoundError("studio not initialized; no session is available to back up")
        session = json.loads(self.session_path.read_text(encoding="utf-8"))
        if session.get("schema_version") != 1:
            raise ValueError("unsupported Studio session schema")
        files = {ALLOWED_PATHS[0]}
        for match in session.get("matches") or []:
            report_relative = self._relative_file(match.get("report"), required=True)
            dashboard_relative = self._relative_file(match.get("dashboard"), required=True)
            files.update((report_relative, dashboard_relative))
            report = json.loads((self.root / report_relative).read_text(encoding="utf-8"))
            for value in (report.get("artifacts") or {}).values():
                relative = self._relative_file(value, required=False)
                if relative:
                    files.add(relative)
        return session, sorted(files)

    def create_backup(
        self, output: str | Path, *, overwrite: bool = False,
    ) -> dict[str, Any]:
        output_path = Path(output).resolve()
        if output_path.suffix.lower() != ".zip":
            raise ValueError("backup output must use the .zip extension")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"backup already exists: {output_path}")
        with FileLease(self.lease_path, timeout=0.0):
            session, relative_files = self._collect_files()
            records = []
            for relative in relative_files:
                path = self.root / relative
                records.append({
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                })
            manifest = {
                "schema_version": RECOVERY_SCHEMA_VERSION,
                "product": "Generative Football Society Studio backup",
                "created_at": _now(),
                "studio": {
                    "name": session.get("name"),
                    "slug": session.get("slug"),
                    "mode": session.get("mode"),
                    "seed": session.get("seed"),
                },
                "file_count": len(records),
                "files": records,
            }
            fd, temporary_name = tempfile.mkstemp(
                dir=output_path.parent, prefix=f".{output_path.name}-", suffix=".tmp",
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                with zipfile.ZipFile(
                    temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                ) as archive:
                    for record in records:
                        archive.write(self.root / record["path"], arcname=record["path"])
                    archive.writestr(
                        MANIFEST_NAME,
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    )
                verification = self.verify_backup(temporary)
                if output_path.exists() and not overwrite:
                    raise FileExistsError(f"backup already exists: {output_path}")
                os.replace(temporary, output_path)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            **verification,
            "bundle": str(output_path),
            "size": output_path.stat().st_size,
        }

    @staticmethod
    def verify_backup(bundle: str | Path) -> dict[str, Any]:
        bundle_path = Path(bundle).resolve()
        if not bundle_path.is_file() or not zipfile.is_zipfile(bundle_path):
            raise ValueError("backup is not a readable ZIP archive")
        with zipfile.ZipFile(bundle_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("backup contains duplicate archive members")
            if MANIFEST_NAME not in names:
                raise ValueError("backup manifest is missing")
            manifest_info = archive.getinfo(MANIFEST_NAME)
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError("backup manifest is too large")
            try:
                manifest = json.loads(archive.read(MANIFEST_NAME))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("backup manifest is invalid JSON") from exc
            if manifest.get("schema_version") != RECOVERY_SCHEMA_VERSION:
                raise ValueError("unsupported backup schema")
            records = manifest.get("files")
            if not isinstance(records, list) or manifest.get("file_count") != len(records):
                raise ValueError("backup file inventory is inconsistent")
            expected_names = {MANIFEST_NAME}
            total = 0
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("backup file record is invalid")
                relative = str(record.get("path") or "")
                if not _allowed_path(relative) or relative in expected_names:
                    raise ValueError(f"unsafe or duplicate backup path: {relative}")
                expected_names.add(relative)
                try:
                    info = archive.getinfo(relative)
                    expected_size = int(record["size"])
                    expected_hash = str(record["sha256"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid backup record: {relative}") from exc
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise ValueError(f"backup member must not be a symbolic link: {relative}")
                if expected_size < 0 or expected_size > MAX_MEMBER_BYTES:
                    raise ValueError(f"backup member size is out of bounds: {relative}")
                if info.file_size != expected_size:
                    raise ValueError(f"backup member size mismatch: {relative}")
                total += expected_size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("backup expands beyond the total size limit")
                with archive.open(info, "r") as handle:
                    observed_hash = _stream_hash(handle)
                if observed_hash != expected_hash:
                    raise ValueError(f"backup member hash mismatch: {relative}")
            if set(names) != expected_names:
                raise ValueError("backup contains untracked archive members")
            if ALLOWED_PATHS[0] not in expected_names:
                raise ValueError("backup does not contain a Studio session")
            session_info = archive.getinfo(ALLOWED_PATHS[0])
            if session_info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError("backup Studio session is too large")
            try:
                session = json.loads(archive.read(ALLOWED_PATHS[0]))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("backup Studio session is invalid JSON") from exc
            if not isinstance(session, dict) or session.get("schema_version") != 1:
                raise ValueError("backup Studio session schema is invalid")
            try:
                config = StudioConfig(
                    name=str(session["name"]), mode=str(session["mode"]),
                    seed=int(session["seed"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("backup Studio configuration is invalid") from exc
            studio = manifest.get("studio") or {}
            if (
                studio.get("name") != config.name
                or studio.get("mode") != config.mode
                or studio.get("seed") != config.seed
                or studio.get("slug") != session.get("slug")
            ):
                raise ValueError("backup manifest does not match the Studio session")
            matches = session.get("matches") or []
            if not isinstance(matches, list):
                raise ValueError("backup Studio match journal is invalid")
            for match in matches:
                if not isinstance(match, dict):
                    raise ValueError("backup Studio match record is invalid")
                report = str(match.get("report") or "")
                dashboard = str(match.get("dashboard") or "")
                if (
                    report not in expected_names or not report.endswith(".json")
                    or dashboard not in expected_names or not dashboard.endswith(".html")
                ):
                    raise ValueError("backup Studio match artifacts are incomplete")
        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "valid": True,
            "bundle": str(bundle_path),
            "file_count": len(records),
            "expanded_size": total,
            "studio": studio,
        }

    def restore_backup(
        self, bundle: str | Path, *, replace: bool = False,
    ) -> dict[str, Any]:
        verification = self.verify_backup(bundle)
        bundle_path = Path(bundle).resolve()
        with FileLease(self.lease_path, timeout=0.0):
            if self.session_path.exists() and not replace:
                raise FileExistsError(
                    "a Studio session already exists; pass --replace for explicit restore",
                )
            with tempfile.TemporaryDirectory(
                dir=self.root, prefix=".gfs-restore-",
            ) as temporary_name:
                temporary = Path(temporary_name)
                staged_root = temporary / "staged"
                rollback_root = temporary / "rollback"
                with zipfile.ZipFile(bundle_path, "r") as archive:
                    manifest = json.loads(archive.read(MANIFEST_NAME))
                    records = manifest["files"]
                    for record in records:
                        relative = str(record["path"])
                        staged = staged_root / Path(relative)
                        staged.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(relative, "r") as source, staged.open("wb") as target:
                            while chunk := source.read(1024 * 1024):
                                target.write(chunk)
                            target.flush()
                            os.fsync(target.fileno())

                ordered = sorted(
                    (str(record["path"]) for record in records),
                    key=lambda value: value == ALLOWED_PATHS[0],
                )
                applied: list[tuple[Path, Path | None]] = []
                try:
                    for relative in ordered:
                        target = self.root / Path(relative)
                        try:
                            target.resolve(strict=False).relative_to(self.root)
                        except ValueError as exc:
                            raise ValueError("restore target escapes the workspace") from exc
                        rollback = None
                        if target.exists():
                            if not replace:
                                raise FileExistsError(f"restore target exists: {relative}")
                            rollback = rollback_root / Path(relative)
                            rollback.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(target, rollback)
                        applied.append((target, rollback))
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(staged_root / Path(relative), target)
                except BaseException:
                    for target, rollback in reversed(applied):
                        target.unlink(missing_ok=True)
                        if rollback is not None and rollback.exists():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(rollback, target)
                    raise
        return {
            **verification,
            "restored": True,
            "replace": bool(replace),
            "restored_at": _now(),
        }
