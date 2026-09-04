"""Integrity-checked backup and transactional restore for one Studio session."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from src.infrastructure import FileLease, file_sha256, fsync_directory
from src.product.tasks import TASK_SCHEMA_VERSION, TASK_STATES
from src.product.telemetry import ProductTelemetry
from src.product.workspace import StudioConfig, _atomic_json


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
TASK_QUEUE_RELATIVE = "data/persistence/product_tasks.json"
ACTIVE_TASK_STATES = {"queued", "running"}
MANAGED_BACKUP_ID = re.compile(r"^backup-[0-9]{8}t[0-9]{6}z-[a-f0-9]{8}$")
MAX_MANAGED_BACKUPS = 50
RESTORE_TRANSACTION_RELATIVE = "data/persistence/product_restore_transaction"
RESTORE_JOURNAL_NAME = "journal.json"
RESTORE_TRANSACTION_SCHEMA_VERSION = 1


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


def _durable_replace(source: str | Path, target: str | Path) -> None:
    source_path, target_path = Path(source), Path(target)
    os.replace(source_path, target_path)
    fsync_directory(target_path.parent)
    if source_path.parent != target_path.parent:
        fsync_directory(source_path.parent)


class ProductRecovery:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.session_path = self.root / ALLOWED_PATHS[0]
        self.lease_path = self.root / "data/persistence/product_session.lock"
        self.task_lease_path = self.root / "data/persistence/product_tasks.lock"
        self.telemetry = ProductTelemetry(self.root)
        self._managed_dir_input = self.root / "backups/studio"
        self.managed_dir = self._managed_dir_input.resolve()
        try:
            self.managed_dir.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("managed backup directory escapes the workspace") from exc
        self.managed_lease_path = self.managed_dir / ".managed-backups.lock"
        self.restore_transaction_root = (
            self.root / RESTORE_TRANSACTION_RELATIVE
        )

    def _safe_restore_target(self, relative: str) -> Path:
        if relative != TASK_QUEUE_RELATIVE and not _allowed_path(relative):
            raise ValueError("restore transaction contains an unsafe target")
        target = self.root / Path(relative)
        try:
            target.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise ValueError("restore transaction target escapes the workspace") from exc
        current = target
        while current != self.root:
            if current.is_symlink():
                raise ValueError(
                    "restore transaction targets must not use symbolic links"
                )
            current = current.parent
        return target

    @staticmethod
    def _entry_exists(path: Path) -> bool:
        return path.exists() or path.is_symlink()

    def _transaction_artifact(self, area: str, relative: str) -> Path:
        if area not in {"staged", "rollback"}:
            raise ValueError("unsupported restore transaction area")
        base = self.restore_transaction_root / area
        candidate = base / Path(relative)
        try:
            candidate.resolve(strict=False).relative_to(base.resolve(strict=False))
        except ValueError as exc:
            raise RuntimeError("restore transaction artifact escapes its area") from exc
        current = candidate
        while current != self.restore_transaction_root:
            if current.is_symlink():
                raise RuntimeError(
                    "restore transaction artifacts must not use symbolic links"
                )
            current = current.parent
        return candidate

    def _remove_restore_transaction_unlocked(self) -> None:
        transaction_root = self.restore_transaction_root
        expected_parent = (self.root / "data/persistence").resolve()
        if transaction_root.parent.resolve() != expected_parent:
            raise RuntimeError("restore transaction directory is outside persistence")
        if transaction_root.is_symlink():
            raise RuntimeError("restore transaction directory must not be a symlink")
        if transaction_root.exists():
            discarded = transaction_root.parent / (
                ".product-restore-discarded-" + uuid.uuid4().hex
            )
            _durable_replace(transaction_root, discarded)
            try:
                shutil.rmtree(discarded)
            except OSError:
                # The authoritative transaction is already atomically inactive.
                # A later recovery pass can safely retry this bounded cleanup.
                return
            fsync_directory(transaction_root.parent)

    def _cleanup_inactive_restore_directories_unlocked(self) -> None:
        parent = self.restore_transaction_root.parent
        if not parent.is_dir():
            return
        for pattern in (
            ".product-restore-preparing-*",
            ".product-restore-discarded-*",
        ):
            for candidate in parent.glob(pattern):
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                try:
                    candidate.resolve().relative_to(parent.resolve())
                except ValueError:
                    continue
                try:
                    shutil.rmtree(candidate)
                except OSError:
                    continue
        fsync_directory(parent)

    def _load_restore_journal_unlocked(self) -> dict[str, Any] | None:
        transaction_root = self.restore_transaction_root
        if transaction_root.is_symlink():
            raise RuntimeError("invalid interrupted restore transaction")
        if not transaction_root.exists():
            return None
        if not transaction_root.is_dir():
            raise RuntimeError("invalid interrupted restore transaction")
        journal_path = transaction_root / RESTORE_JOURNAL_NAME
        if journal_path.is_symlink() or not journal_path.is_file():
            raise RuntimeError("interrupted restore journal is unreadable")
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("interrupted restore journal is unreadable") from exc
        records = journal.get("files")
        if (
            journal.get("schema_version") != RESTORE_TRANSACTION_SCHEMA_VERSION
            or journal.get("state") not in {"prepared", "committed"}
            or not isinstance(journal.get("transaction_id"), str)
            or not re.fullmatch(r"[a-f0-9]{32}", journal["transaction_id"])
            or not isinstance(records, list)
            or not records
        ):
            raise RuntimeError("interrupted restore journal is invalid")
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError("interrupted restore record is invalid")
            relative = str(record.get("path") or "")
            new_sha = str(record.get("new_sha256") or "")
            old_sha = record.get("old_sha256")
            had_original = record.get("had_original")
            if (
                relative in seen
                or not re.fullmatch(r"[a-f0-9]{64}", new_sha)
                or not isinstance(had_original, bool)
                or (
                    had_original
                    and not re.fullmatch(r"[a-f0-9]{64}", str(old_sha or ""))
                )
                or (not had_original and old_sha is not None)
            ):
                raise RuntimeError("interrupted restore record is invalid")
            self._safe_restore_target(relative)
            seen.add(relative)
        ordered_paths = [str(record["path"]) for record in records]
        if ordered_paths[-2:] != [TASK_QUEUE_RELATIVE, ALLOWED_PATHS[0]]:
            raise RuntimeError("interrupted restore record order is invalid")
        return journal

    @staticmethod
    def _matches(path: Path, expected_sha256: str) -> bool:
        return (
            path.is_file()
            and not path.is_symlink()
            and file_sha256(path) == expected_sha256
        )

    def _recover_interrupted_restore_unlocked(
        self, *, force_rollback: bool = False,
    ) -> dict[str, Any]:
        self._cleanup_inactive_restore_directories_unlocked()
        journal = self._load_restore_journal_unlocked()
        if journal is None:
            return {
                "recovered": False,
                "outcome": "no_interrupted_restore",
            }
        records = journal["files"]
        all_new = all(
            self._matches(
                self._safe_restore_target(str(record["path"])),
                str(record["new_sha256"]),
            )
            for record in records
        )
        if all_new and not force_rollback:
            self._remove_restore_transaction_unlocked()
            return {
                "recovered": True,
                "outcome": "completed_committed_restore",
                "transaction_id": journal["transaction_id"],
            }

        # Validate every rollback source and target before changing any path.
        for record in records:
            relative = str(record["path"])
            target = self._safe_restore_target(relative)
            rollback = self._transaction_artifact("rollback", relative)
            if record["had_original"]:
                old_sha = str(record["old_sha256"])
                rollback_is_old = self._matches(rollback, old_sha)
                target_is_old = self._matches(target, old_sha)
                if not rollback_is_old and not target_is_old:
                    raise RuntimeError(
                        "interrupted restore cannot prove its original state"
                    )
            else:
                if self._entry_exists(rollback):
                    raise RuntimeError(
                        "interrupted restore has an unexpected rollback artifact"
                    )
                if self._entry_exists(target) and not self._matches(
                    target, str(record["new_sha256"]),
                ):
                    raise RuntimeError(
                        "interrupted restore target has unexpected content"
                    )

        for record in reversed(records):
            relative = str(record["path"])
            target = self._safe_restore_target(relative)
            rollback = self._transaction_artifact("rollback", relative)
            if record["had_original"]:
                old_sha = str(record["old_sha256"])
                if self._matches(rollback, old_sha):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _durable_replace(rollback, target)
            elif self._entry_exists(target):
                target.unlink()
                fsync_directory(target.parent)

        for record in records:
            target = self._safe_restore_target(str(record["path"]))
            if record["had_original"]:
                if not self._matches(target, str(record["old_sha256"])):
                    raise RuntimeError("restore rollback verification failed")
            elif self._entry_exists(target):
                raise RuntimeError("restore rollback left a new artifact")
        self._remove_restore_transaction_unlocked()
        return {
            "recovered": True,
            "outcome": "rolled_back_interrupted_restore",
            "transaction_id": journal["transaction_id"],
        }

    def recover_interrupted_restore(self) -> dict[str, Any]:
        """Resolve an activated multi-file restore after an ungraceful stop."""

        with (
            FileLease(self.lease_path, timeout=0.0),
            FileLease(self.task_lease_path, timeout=0.0),
        ):
            return self._recover_interrupted_restore_unlocked()

    def _assert_managed_dir(self) -> None:
        observed = self._managed_dir_input.resolve()
        try:
            observed.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("managed backup directory escapes the workspace") from exc
        if observed != self.managed_dir:
            raise ValueError("managed backup directory identity changed")

    def _managed_path(self, backup_id: str, *, must_exist: bool) -> Path:
        if not isinstance(backup_id, str) or not MANAGED_BACKUP_ID.fullmatch(backup_id):
            raise ValueError("invalid managed backup id")
        candidate = self.managed_dir / f"{backup_id}.zip"
        if must_exist:
            if candidate.is_symlink() or not candidate.is_file():
                raise FileNotFoundError("managed backup not found")
            try:
                candidate.resolve().relative_to(self.managed_dir)
            except ValueError as exc:
                raise ValueError("managed backup escapes its directory") from exc
        return candidate

    @staticmethod
    def _managed_entry(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "backup_id": path.stem,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc,
            ).isoformat(),
        }

    def _list_managed_unlocked(self) -> list[dict[str, Any]]:
        if not self.managed_dir.is_dir():
            return []
        paths = [
            path for path in self.managed_dir.glob("backup-*.zip")
            if MANAGED_BACKUP_ID.fullmatch(path.stem)
            and path.is_file() and not path.is_symlink()
        ]
        paths.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
        return [self._managed_entry(path) for path in paths]

    def list_managed_backups(self) -> dict[str, Any]:
        self._assert_managed_dir()
        with FileLease(self.managed_lease_path, timeout=2.0):
            backups = self._list_managed_unlocked()
        return {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "backups": backups,
            "count": len(backups),
            "capacity": MAX_MANAGED_BACKUPS,
        }

    @staticmethod
    def _public_managed_result(
        backup_id: str, result: dict[str, Any], *, operation: str,
    ) -> dict[str, Any]:
        public = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "operation": operation,
            "backup_id": backup_id,
            "valid": bool(result.get("valid")),
            "file_count": int(result.get("file_count", 0)),
            "expanded_size": int(result.get("expanded_size", 0)),
            "studio": result.get("studio"),
        }
        if operation == "create":
            public["size_bytes"] = int(result.get("size", 0))
        if operation == "restore":
            public.update({
                "restored": bool(result.get("restored")),
                "replace": bool(result.get("replace")),
                "task_history_reset": bool(result.get("task_history_reset")),
                "restored_at": result.get("restored_at"),
            })
        return public

    def create_managed_backup(self) -> dict[str, Any]:
        self.managed_dir.mkdir(parents=True, exist_ok=True)
        self._assert_managed_dir()
        with FileLease(self.managed_lease_path, timeout=2.0):
            if len(self._list_managed_unlocked()) >= MAX_MANAGED_BACKUPS:
                raise RuntimeError("managed backup capacity reached")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
            for _attempt in range(10):
                backup_id = f"backup-{timestamp}-{secrets.token_hex(4)}"
                output = self._managed_path(backup_id, must_exist=False)
                if not output.exists():
                    break
            else:
                raise RuntimeError("unable to allocate a managed backup id")
            result = self.create_backup(output)
        return self._public_managed_result(backup_id, result, operation="create")

    def verify_managed_backup(self, backup_id: str) -> dict[str, Any]:
        self._assert_managed_dir()
        with FileLease(self.managed_lease_path, timeout=2.0):
            result = self.verify_backup(self._managed_path(backup_id, must_exist=True))
        return self._public_managed_result(backup_id, result, operation="verify")

    def restore_managed_backup(
        self, backup_id: str, *, replace: bool,
    ) -> dict[str, Any]:
        self._assert_managed_dir()
        with FileLease(self.managed_lease_path, timeout=2.0):
            result = self.restore_backup(
                self._managed_path(backup_id, must_exist=True), replace=replace,
            )
        return self._public_managed_result(backup_id, result, operation="restore")

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
        with (
            FileLease(self.lease_path, timeout=0.0),
            FileLease(self.task_lease_path, timeout=0.0),
        ):
            self._recover_interrupted_restore_unlocked()
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
                _durable_replace(temporary, output_path)
            finally:
                temporary.unlink(missing_ok=True)
        result = {
            **verification,
            "bundle": str(output_path),
            "size": output_path.stat().st_size,
        }
        self.telemetry.try_record(
            "backup_created", file_count=int(result["file_count"]),
            size_bytes=int(result["size"]),
        )
        return result

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
        bundle_path = Path(bundle).resolve()
        with (
            FileLease(self.lease_path, timeout=0.0),
            FileLease(self.task_lease_path, timeout=0.0),
        ):
            self._recover_interrupted_restore_unlocked()
            bundle_sha256 = file_sha256(bundle_path)
            verification = self.verify_backup(bundle_path)
            if file_sha256(bundle_path) != bundle_sha256:
                raise RuntimeError("backup changed during verification")
            if self.session_path.exists() and not replace:
                raise FileExistsError(
                    "a Studio session already exists; pass --replace for explicit restore",
                )
            transaction_parent = self.restore_transaction_root.parent
            transaction_parent.mkdir(parents=True, exist_ok=True)
            preparing = Path(tempfile.mkdtemp(
                dir=transaction_parent, prefix=".product-restore-preparing-",
            ))
            try:
                staged_root = preparing / "staged"
                task_path = self.root / TASK_QUEUE_RELATIVE
                if task_path.is_file():
                    try:
                        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError("existing product task queue is invalid") from exc
                    tasks = task_payload.get("tasks")
                    if (
                        task_payload.get("schema_version") != TASK_SCHEMA_VERSION
                        or not isinstance(tasks, list)
                        or any(
                            not isinstance(task, dict)
                            or task.get("state") not in TASK_STATES
                            for task in tasks
                        )
                    ):
                        raise ValueError("existing product task queue is invalid")
                    if any(task.get("state") in ACTIVE_TASK_STATES for task in tasks):
                        raise RuntimeError(
                            "cannot restore while product tasks are queued or running",
                        )
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
                        if (
                            staged.stat().st_size != int(record["size"])
                            or file_sha256(staged) != str(record["sha256"])
                        ):
                            raise RuntimeError(
                                "backup changed during staged extraction"
                            )

                staged_tasks = staged_root / TASK_QUEUE_RELATIVE
                staged_tasks.parent.mkdir(parents=True, exist_ok=True)
                with staged_tasks.open("w", encoding="utf-8") as handle:
                    json.dump({
                        "schema_version": TASK_SCHEMA_VERSION,
                        "updated_at": _now(), "tasks": [],
                    }, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                fsync_directory(staged_tasks.parent)

                ordered_artifacts = sorted(
                    (str(record["path"]) for record in records),
                    key=lambda value: value == ALLOWED_PATHS[0],
                )
                ordered = [
                    value for value in ordered_artifacts
                    if value != ALLOWED_PATHS[0]
                ] + [TASK_QUEUE_RELATIVE, ALLOWED_PATHS[0]]
                journal_records = []
                for relative in ordered:
                    target = self._safe_restore_target(relative)
                    staged = staged_root / Path(relative)
                    if not staged.is_file() or staged.is_symlink():
                        raise RuntimeError("prepared restore artifact is invalid")
                    target_exists = self._entry_exists(target)
                    if target_exists and (
                        target.is_symlink() or not target.is_file()
                    ):
                        raise RuntimeError("restore target must be a regular file")
                    if (
                        target_exists
                        and not replace
                        and relative != TASK_QUEUE_RELATIVE
                    ):
                        raise FileExistsError(
                            f"restore target exists: {relative}"
                        )
                    journal_records.append({
                        "path": relative,
                        "new_sha256": file_sha256(staged),
                        "had_original": target_exists,
                        "old_sha256": (
                            file_sha256(target) if target_exists else None
                        ),
                    })
                journal = {
                    "schema_version": RESTORE_TRANSACTION_SCHEMA_VERSION,
                    "transaction_id": uuid.uuid4().hex,
                    "state": "prepared",
                    "created_at": _now(),
                    "bundle_sha256": bundle_sha256,
                    "replace": bool(replace),
                    "files": journal_records,
                }
                if file_sha256(bundle_path) != bundle_sha256:
                    raise RuntimeError("backup changed during staged extraction")
                _atomic_json(preparing / RESTORE_JOURNAL_NAME, journal)
                _durable_replace(preparing, self.restore_transaction_root)
                preparing = None
                try:
                    for record in journal_records:
                        relative = str(record["path"])
                        target = self._safe_restore_target(relative)
                        rollback = self._transaction_artifact(
                            "rollback", relative,
                        )
                        if self._entry_exists(target):
                            rollback.parent.mkdir(parents=True, exist_ok=True)
                            _durable_replace(target, rollback)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        _durable_replace(
                            self._transaction_artifact("staged", relative),
                            target,
                        )
                    journal["state"] = "committed"
                    _atomic_json(
                        self.restore_transaction_root / RESTORE_JOURNAL_NAME,
                        journal,
                    )
                    completion = self._recover_interrupted_restore_unlocked()
                    if completion.get("outcome") != "completed_committed_restore":
                        raise RuntimeError("restore commit verification failed")
                except BaseException:
                    try:
                        self._recover_interrupted_restore_unlocked(
                            force_rollback=True,
                        )
                    except BaseException as rollback_error:
                        raise RuntimeError(
                            "restore failed and rollback could not be proven"
                        ) from rollback_error
                    raise
            finally:
                if preparing is not None and preparing.exists():
                    shutil.rmtree(preparing)
                    fsync_directory(transaction_parent)
        result = {
            **verification,
            "restored": True,
            "replace": bool(replace),
            "task_history_reset": True,
            "restored_at": _now(),
        }
        self.telemetry.try_record(
            "restore_completed", file_count=int(result["file_count"]),
            replace=bool(replace), task_history_reset=True,
        )
        return result
