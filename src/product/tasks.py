"""Persistent background task queue for the Studio Web adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.infrastructure import FileLease
from src.product.telemetry import ProductTelemetry
from src.product.workspace import ProductWorkspace, _atomic_json, _now


TASK_SCHEMA_VERSION = 1
TASK_STATES = {"queued", "running", "completed", "failed", "interrupted"}
TERMINAL_TASK_STATES = {"completed", "failed", "interrupted"}
MAX_RETAINED_TASKS = 1000
PRUNE_TO_TASKS = 900
LOGGER = logging.getLogger(__name__)


class TaskConflict(RuntimeError):
    """An idempotency key was reused for a different immutable request."""


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in task.items()
        if key not in {"idempotency_hash", "worker_id"}
    }


class ProductTaskQueue:
    def __init__(
        self, root: str | Path, *, telemetry: ProductTelemetry | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.path = self.root / "data/persistence/product_tasks.json"
        self.lease_path = self.root / "data/persistence/product_tasks.lock"
        self.telemetry = telemetry or ProductTelemetry(self.root)

    @staticmethod
    def _duration_ms(started_at: Any) -> float:
        try:
            started = datetime.fromisoformat(str(started_at))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - started).total_seconds() * 1000)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"schema_version": TASK_SCHEMA_VERSION, "updated_at": _now(), "tasks": []}

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != TASK_SCHEMA_VERSION:
            raise ValueError("unsupported product task queue schema")
        tasks = payload.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("invalid product task queue")
        if any(
            not isinstance(task, dict) or task.get("state") not in TASK_STATES
            for task in tasks
        ):
            raise ValueError("invalid product task record")
        task_ids = [str(task.get("task_id") or "") for task in tasks]
        if any(not task_id or not task_id.isalnum() for task_id in task_ids):
            raise ValueError("invalid product task id in queue")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate product task id in queue")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now()
        _atomic_json(self.path, payload)

    @staticmethod
    def _idempotency_hash(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        if len(normalized) > 200 or any(ord(char) < 33 for char in normalized):
            raise ValueError("invalid idempotency key")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def submit_match(
        self, home: str, away: str, *, fast: bool,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], bool]:
        if (
            not isinstance(home, str) or not home.strip() or len(home.strip()) > 80
            or any(ord(char) < 32 for char in home)
        ):
            raise ValueError("invalid home team")
        if (
            not isinstance(away, str) or not away.strip() or len(away.strip()) > 80
            or any(ord(char) < 32 for char in away)
        ):
            raise ValueError("invalid away team")
        home, away = home.strip(), away.strip()
        if home.casefold() == away.casefold():
            raise ValueError("home and away teams must differ")
        if not isinstance(fast, bool):
            raise ValueError("fast must be boolean")
        idempotency_hash = self._idempotency_hash(idempotency_key)
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            if idempotency_hash:
                for task in payload["tasks"]:
                    if task.get("idempotency_hash") == idempotency_hash:
                        expected = {"home": home, "away": away, "fast": fast}
                        if task.get("request") != expected:
                            raise TaskConflict(
                                "idempotency key was already used for a different match",
                            )
                        public = _public_task(task)
                        self.telemetry.try_record(
                            "task_submitted", task_id=public["task_id"],
                            created=False, fast=fast,
                        )
                        return public, False
            if len(payload["tasks"]) >= MAX_RETAINED_TASKS:
                removable = sum(
                    task.get("state") in TERMINAL_TASK_STATES
                    for task in payload["tasks"]
                )
                remove_count = min(
                    removable, len(payload["tasks"]) - PRUNE_TO_TASKS,
                )
                retained = []
                for task in payload["tasks"]:
                    if remove_count and task.get("state") in TERMINAL_TASK_STATES:
                        remove_count -= 1
                        continue
                    retained.append(task)
                payload["tasks"] = retained
            if len(payload["tasks"]) >= MAX_RETAINED_TASKS:
                raise RuntimeError("product task queue is full")
            task = {
                "task_id": uuid.uuid4().hex,
                "kind": "match",
                "state": "queued",
                "created_at": _now(),
                "updated_at": _now(),
                "attempt": 0,
                "request": {"home": home, "away": away, "fast": bool(fast)},
                "idempotency_hash": idempotency_hash,
            }
            payload["tasks"].append(task)
            self._write(payload)
            public = _public_task(task)
            self.telemetry.try_record(
                "task_submitted", task_id=public["task_id"], created=True, fast=fast,
            )
            return public, True

    def list_tasks(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("task limit must be between 1 and 500")
        with FileLease(self.lease_path, timeout=5.0):
            tasks = self._load()["tasks"]
        return [_public_task(task) for task in reversed(tasks[-limit:])]

    def get_task(self, task_id: str) -> dict[str, Any]:
        if not task_id or len(task_id) > 64 or not task_id.isalnum():
            raise ValueError("invalid product task id")
        with FileLease(self.lease_path, timeout=5.0):
            for task in self._load()["tasks"]:
                if task.get("task_id") == task_id:
                    return _public_task(task)
        raise FileNotFoundError(f"product task not found: {task_id}")

    def recover_running(self, *, reason: str = "web_worker_restarted") -> int:
        recovered = 0
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            for task in payload["tasks"]:
                if task.get("state") == "running":
                    task.update({
                        "state": "interrupted", "finished_at": _now(),
                        "updated_at": _now(), "reason": reason,
                    })
                    recovered += 1
            if recovered:
                self._write(payload)
        if recovered:
            self.telemetry.try_record(
                "task_recovered", count=recovered, reason=reason,
            )
        return recovered

    def claim_next(self, worker_id: str) -> dict[str, Any] | None:
        if not worker_id.strip():
            raise ValueError("worker id must not be empty")
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            for task in payload["tasks"]:
                if task.get("state") != "queued":
                    continue
                task.update({
                    "state": "running", "started_at": _now(),
                    "updated_at": _now(), "worker_id": worker_id,
                    "attempt": int(task.get("attempt", 0)) + 1,
                })
                self._write(payload)
                self.telemetry.try_record(
                    "task_claimed", task_id=task["task_id"], attempt=task["attempt"],
                )
                return dict(task)
        return None

    def _finish(
        self, task_id: str, worker_id: str, *, state: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        if state not in {"completed", "failed"}:
            raise ValueError("unsupported terminal task state")
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            for task in payload["tasks"]:
                if task.get("task_id") != task_id:
                    continue
                if task.get("state") != "running" or task.get("worker_id") != worker_id:
                    raise RuntimeError("product task is not owned by this worker")
                task.update({
                    "state": state, "finished_at": _now(), "updated_at": _now(),
                    **values,
                })
                self._write(payload)
                telemetry_fields = {
                    "task_id": task["task_id"],
                    "attempt": int(task.get("attempt", 0)),
                    "duration_ms": self._duration_ms(task.get("started_at")),
                }
                if state == "failed":
                    telemetry_fields["error_type"] = str(
                        (values.get("error") or {}).get("type") or "UnknownError"
                    )[:100]
                self.telemetry.try_record(
                    f"task_{state}", **telemetry_fields,
                )
                return _public_task(task)
        raise FileNotFoundError(f"product task not found: {task_id}")

    def complete(self, task_id: str, worker_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return self._finish(
            task_id, worker_id, state="completed", values={"result": result},
        )

    def fail(self, task_id: str, worker_id: str, error_type: str) -> dict[str, Any]:
        return self._finish(task_id, worker_id, state="failed", values={
            "error": {
                "code": "match_execution_failed",
                "type": str(error_type)[:100],
                "message": "Match execution failed; inspect local logs with the task ID",
            },
        })


class BackgroundMatchWorker:
    def __init__(
        self, queue: ProductTaskQueue, *, poll_interval: float = 0.2,
        workspace_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.queue = queue
        self.worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"
        self.poll_interval = max(0.05, float(poll_interval))
        self.workspace_loader = workspace_loader or ProductWorkspace.load
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _relative(root: Path, value: Any) -> str:
        candidate = Path(str(value)).resolve()
        return candidate.relative_to(root).as_posix()

    def run_once(self) -> bool:
        task = self.queue.claim_next(self.worker_id)
        if task is None:
            return False
        try:
            request = task["request"]
            report = self.workspace_loader(self.queue.root).run_match(
                str(request["home"]), str(request["away"]), fast=bool(request["fast"]),
            )
            result = {
                "match_id": report["match_id"],
                "fixture": report["fixture"],
                "result": report["result"],
                "integrity": report["integrity"],
                "report": self._relative(self.queue.root, report["report_path"]),
                "dashboard": self._relative(self.queue.root, report["dashboard_path"]),
            }
            self.queue.complete(task["task_id"], self.worker_id, result)
        except Exception as exc:
            try:
                self.queue.fail(task["task_id"], self.worker_id, type(exc).__name__)
            except Exception:
                LOGGER.exception(
                    "Unable to persist failed product task task_id=%s",
                    task.get("task_id"),
                )
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:
                LOGGER.exception("Background match worker loop failed")
                worked = False
            if not worked:
                self._stop.wait(self.poll_interval)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            raise RuntimeError("background match worker is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="gfs-match-worker", daemon=True,
        )
        self._thread.start()
        self.queue.telemetry.try_record("worker_lifecycle", state="started")

    def stop(self, *, timeout: float = 5.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=max(0.0, timeout))
        stopped = not thread.is_alive()
        self.queue.telemetry.try_record(
            "worker_lifecycle", state="stopped" if stopped else "stop_timeout",
        )
        return stopped
