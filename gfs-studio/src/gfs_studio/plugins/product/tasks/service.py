from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.identity import content_identity
from gfs_core.kernel.workspace import WorkspaceRepository

TERMINAL_STATES = {"review", "failed", "interrupted", "cancelled"}


class TaskCancelled(RuntimeError):
    pass


class TaskService:
    def __init__(self, project_root, workflows) -> None:
        self.workspace = WorkspaceRepository(project_root)
        self.workflows = workflows
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._running_ids = set()
        self._closed = False

    def submit(self, idempotency_key: str, payload: dict) -> dict:
        key = _key(idempotency_key)
        task_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        fingerprint = _payload_hash(payload)
        with self._lock, self.workspace.lease("tasks.lock", timeout=30.0):
            if self._closed:
                raise RuntimeError("task service is closed")
            existing = self.get(task_id)
            if existing is not None:
                if existing.get("payload_sha256", _payload_hash(existing["payload"])) != fingerprint:
                    raise ValueError("idempotency key conflicts with a different payload")
                return existing
            task = {
                "task_id": task_id,
                "idempotency_key": key,
                "kind": "micro_match",
                "payload": dict(payload),
                "payload_sha256": fingerprint,
                "state": "running",
                "cancel_requested": False,
                "result": None,
                "error_type": None,
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._save(task)
            self._start(task)
            return task

    def get(self, task_id: str) -> dict | None:
        tasks = self._read()
        return next((task for task in tasks if task["task_id"] == task_id), None)

    def list(self) -> list[dict]:
        return sorted(self._read(), key=lambda task: task["created_at"])

    def wait(self, task_id: str, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                task = self.get(task_id)
                if task is None:
                    raise KeyError(task_id)
                if task["state"] in TERMINAL_STATES:
                    return task
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(task_id)
                self._condition.wait(min(0.1, remaining))

    def recover(self) -> int:
        with self._lock, self.workspace.lease("tasks.lock", timeout=30.0):
            tasks = self._read()
            recovered = 0
            for task in tasks:
                if task["state"] == "running" and task["task_id"] not in self._running_ids:
                    task["state"] = "interrupted"
                    task["updated_at"] = _now()
                    recovered += 1
            if recovered:
                self.workspace.write("tasks.json", {"tasks": tasks})
                self._condition.notify_all()
            return recovered

    def resume(self, task_id: str | None = None) -> int:
        with self._lock, self.workspace.lease("tasks.lock", timeout=30.0):
            if self._closed:
                raise RuntimeError("task service is closed")
            tasks = self._read()
            selected = [
                task for task in tasks
                if task["state"] == "interrupted"
                and (task_id is None or task["task_id"] == task_id)
            ]
            for task in selected:
                task["state"] = "running"
                task["cancel_requested"] = False
                task["updated_at"] = _now()
            if selected:
                self.workspace.write("tasks.json", {"tasks": tasks})
            for task in selected:
                self._start(task)
            return len(selected)

    def cancel(self, task_id: str) -> dict:
        with self._condition, self.workspace.lease("tasks.lock", timeout=30.0):
            tasks = self._read()
            for task in tasks:
                if task["task_id"] != task_id:
                    continue
                if task["state"] not in TERMINAL_STATES:
                    task["cancel_requested"] = True
                    task["updated_at"] = _now()
                    self.workspace.write("tasks.json", {"tasks": tasks})
                    self._condition.notify_all()
                return task
            raise KeyError(task_id)

    def close(self, *, cancel: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = list(self._running_ids)
        if cancel:
            for task_id in active:
                try:
                    self.cancel(task_id)
                except KeyError:
                    pass
        self._executor.shutdown(wait=True)

    def _start(self, task: dict) -> None:
        self._running_ids.add(task["task_id"])
        try:
            self._executor.submit(self._execute, task)
        except RuntimeError:
            self._running_ids.discard(task["task_id"])
            raise

    def _execute(self, task: dict) -> None:
        try:
            payload = task["payload"]
            if task.get("cancel_requested"):
                raise TaskCancelled(task["task_id"])
            result = self.workflows.micro_match(
                payload["home"], payload["away"], int(payload.get("seed", 42)),
                before_tick=lambda: self._raise_if_cancelled(task["task_id"]),
            )
            self._finish(task["task_id"], "review", result=result)
        except TaskCancelled as error:
            self._finish(task["task_id"], "cancelled", error=error)
        except Exception as error:
            self._finish(task["task_id"], "failed", error=error)

    def _raise_if_cancelled(self, task_id: str) -> None:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.get("cancel_requested"):
            raise TaskCancelled(task_id)

    def _finish(self, task_id: str, state: str, *, result=None, error=None) -> None:
        with self._condition, self.workspace.lease("tasks.lock", timeout=30.0):
            tasks = self._read()
            for task in tasks:
                if task["task_id"] == task_id:
                    task["state"] = state
                    task["result"] = result
                    task["error_type"] = None if error is None else type(error).__name__
                    task["updated_at"] = _now()
                    break
            else:
                raise KeyError(f"task disappeared while executing: {task_id}")
            self._running_ids.discard(task_id)
            self.workspace.write("tasks.json", {"tasks": tasks})
            self._condition.notify_all()

    def _save(self, task: dict) -> None:
        tasks = self._read()
        tasks.append(task)
        self.workspace.write("tasks.json", {"tasks": tasks})

    def _read(self) -> list[dict]:
        try:
            payload = self.workspace.read("tasks.json")
        except FileNotFoundError:
            return []
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError("task registry must contain an array")
        return tasks


class TaskPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.product.tasks",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.product",
        provides=("product-tasks",),
        requires=("gfs.studio.product.studio",),
    )

    def __init__(self, workflows, *, cancel_on_close: bool = True) -> None:
        self.service = TaskService(workflows.project_root, workflows)
        self.cancel_on_close = cancel_on_close

    def activate(self, context: PluginContext) -> None:
        self.service.recover()
        context.provide("product-tasks", self.service)

    def deactivate(self) -> None:
        self.service.close(cancel=self.cancel_on_close)


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 180:
        raise ValueError("idempotency key is invalid")
    return value.strip()


def _payload_hash(payload: dict) -> str:
    return content_identity({"kind": "micro_match", "payload": payload})


def _now() -> str:
    return datetime.now(UTC).isoformat()
