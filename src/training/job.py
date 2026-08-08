"""Atomic progress, structured logs, resume, and cooperative stop for training."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import socket
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.infrastructure import FileLease, LeaseUnavailable


SCHEMA_VERSION = 1
TERMINAL_STATES = {"completed", "stopped", "failed", "interrupted"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class TrainingJob:
    """Own one long-running training lifecycle under an isolated run directory."""

    def __init__(self, run_dir: str | Path, config: Mapping[str, Any]) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.log_path = self.run_dir / "epochs.jsonl"
        self.progress_path = self.run_dir / "latest.pt"
        self.stop_path = self.run_dir / "STOP"
        self.lease_path = self.run_dir / ".run.lock"
        self._lease = FileLease(self.lease_path)
        self.config = dict(config)
        self.config_fingerprint = _fingerprint(self.config)
        self._terminal = False
        self._started = False
        self.attempt_id: str | None = None
        atexit.register(self._mark_interrupted_at_exit)

    def _status(self, state: str, **fields: Any) -> dict[str, Any]:
        previous: dict[str, Any] = {}
        if self.status_path.is_file():
            previous = json.loads(self.status_path.read_text(encoding="utf-8"))
        payload = {
            **previous,
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": _now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
            **fields,
        }
        payload.setdefault("started_at", _now())
        _atomic_json(self.status_path, payload)
        self._terminal = state in TERMINAL_STATES
        return payload

    def _require_active(self) -> None:
        if not self._started or self._terminal or not self._lease.held:
            raise RuntimeError("training job is not in an active running state")

    def start(self, *, total_epochs: int, resume: bool) -> None:
        prior: dict[str, Any] = {}
        if self.status_path.is_file():
            prior = json.loads(self.status_path.read_text(encoding="utf-8"))
            if prior.get("config_fingerprint") != self.config_fingerprint:
                raise ValueError("training run config does not match existing run directory")
            if not resume:
                raise FileExistsError("training run already exists; pass --resume or use a new --run-dir")
            if prior.get("state") == "completed":
                raise ValueError("completed training runs cannot be resumed")
        elif resume:
            raise FileNotFoundError("resume requested but training status is missing")
        if resume and not self.progress_path.is_file():
            raise FileNotFoundError("resume requested but latest.pt is missing")
        try:
            self._lease.acquire()
        except LeaseUnavailable as exc:
            raise RuntimeError("training run is already active in another process") from exc
        try:
            if resume:
                self.stop_path.unlink(missing_ok=True)
            self.attempt_id = uuid.uuid4().hex
            attempts = list(prior.get("attempts") or [])
            attempts.append({
                "attempt_id": self.attempt_id,
                "started_at": _now(),
                "resumed": bool(resume),
                "resume_from_epoch": int(prior.get("last_checkpoint_epoch", 0)),
            })
            self._status(
                "running", total_epochs=int(total_epochs), resumed=bool(resume),
                current_attempt_id=self.attempt_id, attempts=attempts,
            )
            self._started = True
        except BaseException:
            self._lease.release()
            raise

    def record_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        self._require_active()
        record = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": _now(),
            "attempt_id": self.attempt_id,
            "epoch": int(epoch),
            **dict(metrics),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._status("running", last_completed_epoch=int(epoch), last_metrics=dict(metrics))

    def save_progress(self, payload: Mapping[str, Any]) -> None:
        self._require_active()
        import torch

        fd, temporary = tempfile.mkstemp(
            dir=self.run_dir, prefix=".latest-", suffix=".pt.tmp",
        )
        os.close(fd)
        try:
            torch.save(dict(payload), temporary)
            os.replace(temporary, self.progress_path)
            self._status(
                "running",
                last_checkpoint_epoch=int(payload["epoch"]),
                resumable_from_epoch=int(payload["epoch"]),
            )
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def load_progress(self) -> dict[str, Any]:
        self._require_active()
        if not self.progress_path.is_file():
            raise FileNotFoundError("resume requested but latest.pt is missing")
        import torch

        payload = torch.load(self.progress_path, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError("training progress config fingerprint mismatch")
        return dict(payload)

    def stop_requested(self) -> bool:
        return self.stop_path.is_file()

    def mark_stopped(self, *, epoch: int) -> None:
        self._require_active()
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        if int(status.get("last_checkpoint_epoch", -1)) != int(epoch):
            raise RuntimeError("cannot stop safely before checkpointing the current epoch")
        self._status("stopped", last_completed_epoch=int(epoch), reason="cooperative_stop")
        self._lease.release()

    def complete(self, *, artifact: str, epoch: int) -> None:
        self._require_active()
        self._status("completed", artifact=artifact, last_completed_epoch=int(epoch))
        self._lease.release()

    def fail(self, error: BaseException) -> None:
        self._require_active()
        self._status("failed", error_type=type(error).__name__, error=str(error))
        self._lease.release()

    def _mark_interrupted_at_exit(self) -> None:
        if self._started and not self._terminal and self.status_path.is_file():
            try:
                status = json.loads(self.status_path.read_text(encoding="utf-8"))
                if status.get("state") == "running":
                    self._status("interrupted", reason="process_exit_before_terminal_state")
            except Exception:
                pass
            finally:
                self._lease.release()
