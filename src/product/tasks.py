"""Persistent background task queue for the Studio Web adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import socket
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.infrastructure import FileLease
from src.product.telemetry import ProductTelemetry
from src.product.match_plan import MatchPlan, PairedMatchPlan, WorldModelForkPlan
from src.product.tactical_study import TacticalStudyPlan
from src.product.workspace import ProductWorkspace, _atomic_json, _now


TASK_SCHEMA_VERSION = 1
TASK_STATES = {"queued", "running", "completed", "failed", "interrupted"}
TASK_KINDS = {
    "match", "paired_match", "world_model_fork", "tactical_study",
    "season_matchday",
}
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


def _world_model_propagation_digest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "available": False,
            "status": "comparison_propagation_unavailable",
        }
    propagation = (
        payload.get("policy_propagation")
        if isinstance(payload, dict) else None
    )
    if not isinstance(propagation, dict) or not propagation.get("available"):
        return {
            "available": False,
            "status": "legacy_comparison_without_propagation",
        }
    summary = propagation.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}

    def bounded_count(key: str) -> int:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        number = float(value)
        return (
            max(0, min(100_000, int(number)))
            if math.isfinite(number) else 0
        )

    return {
        "available": True,
        "status": str(propagation.get("status") or "unknown")[:80],
        "changed_decisions": bounded_count("valid_changed_decisions"),
        "directly_observed_changes": bounded_count(
            "directly_observed_changes"
        ),
        "locally_attributable_changes": bounded_count(
            "locally_attributable_changes"
        ),
        "replay_windows_available": bool(
            summary.get("replay_windows_available") is True
        ),
        "downstream_causal_attribution_authorized": False,
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
            or task.get("kind") not in TASK_KINDS
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
        plan: MatchPlan | dict[str, Any] | None = None,
        seed_override: int | None = None,
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
        normalized_plan = (
            plan if isinstance(plan, MatchPlan) else MatchPlan.from_payload(plan)
        )
        if (
            seed_override is not None
            and (
                isinstance(seed_override, bool)
                or not isinstance(seed_override, int)
                or not 0 <= seed_override <= 2**31 - 1
            )
        ):
            raise ValueError("seed_override must be a 32-bit non-negative integer")
        normalized_request = {
            "home": home, "away": away, "fast": fast,
            "plan": normalized_plan.as_dict(),
            "seed_override": seed_override,
        }
        idempotency_hash = self._idempotency_hash(idempotency_key)
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            if idempotency_hash:
                for task in payload["tasks"]:
                    if task.get("idempotency_hash") == idempotency_hash:
                        if task.get("request") != normalized_request:
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
                "request": normalized_request,
                "idempotency_hash": idempotency_hash,
            }
            payload["tasks"].append(task)
            self._write(payload)
            public = _public_task(task)
            self.telemetry.try_record(
                "task_submitted", task_id=public["task_id"], created=True, fast=fast,
            )
            return public, True

    def submit_season_matchday(
        self, *, season_id: str, matchday: int, season_revision: int,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], bool]:
        if not isinstance(season_id, str) or not season_id.strip():
            raise ValueError("invalid season_id")
        if isinstance(matchday, bool) or not isinstance(matchday, int) or matchday < 1:
            raise ValueError("invalid matchday")
        if isinstance(season_revision, bool) or not isinstance(season_revision, int) or season_revision < 0:
            raise ValueError("invalid season_revision")
        request = {
            "season_id": season_id.strip(), "matchday": matchday,
            "season_revision": season_revision,
        }
        idempotency_hash = self._idempotency_hash(idempotency_key)
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            if idempotency_hash:
                for task in payload["tasks"]:
                    if task.get("idempotency_hash") != idempotency_hash:
                        continue
                    if task.get("kind") != "season_matchday" or task.get("request") != request:
                        raise TaskConflict(
                            "idempotency key was already used for a different matchday"
                        )
                    return _public_task(task), False
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
                "kind": "season_matchday", "state": "queued",
                "created_at": _now(), "updated_at": _now(), "attempt": 0,
                "request": request, "idempotency_hash": idempotency_hash,
            }
            payload["tasks"].append(task)
            self._write(payload)
            public = _public_task(task)
            self.telemetry.try_record(
                "task_submitted", task_id=public["task_id"], created=True,
            )
            return public, True

    def submit_tactical_study(
        self, plan: TacticalStudyPlan | dict[str, Any], *,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], bool]:
        normalized = (
            plan if isinstance(plan, TacticalStudyPlan)
            else TacticalStudyPlan.from_payload(plan)
        )
        request = {"plan": normalized.as_dict()}
        idempotency_hash = self._idempotency_hash(idempotency_key)
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            if idempotency_hash:
                for task in payload["tasks"]:
                    if task.get("idempotency_hash") != idempotency_hash:
                        continue
                    if task.get("kind") != "tactical_study" or task.get(
                        "request"
                    ) != request:
                        raise TaskConflict(
                            "idempotency key was already used for a different study"
                        )
                    public = _public_task(task)
                    self.telemetry.try_record(
                        "task_submitted", task_id=public["task_id"],
                        created=False, fast=normalized.fast,
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
                "kind": "tactical_study",
                "state": "queued",
                "created_at": _now(), "updated_at": _now(), "attempt": 0,
                "request": request,
                "idempotency_hash": idempotency_hash,
            }
            payload["tasks"].append(task)
            self._write(payload)
            public = _public_task(task)
            self.telemetry.try_record(
                "task_submitted", task_id=public["task_id"],
                created=True, fast=normalized.fast,
            )
            return public, True

    def submit_paired_match(
        self, home: str, away: str, *, fast: bool,
        plan: PairedMatchPlan | dict[str, Any], idempotency_key: str = "",
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
        normalized = (
            plan if isinstance(plan, PairedMatchPlan)
            else PairedMatchPlan.from_payload(plan)
        )
        request = {
            "home": home, "away": away, "fast": fast,
            "plan": normalized.as_dict(),
        }
        idempotency_hash = self._idempotency_hash(idempotency_key)
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            if idempotency_hash:
                for task in payload["tasks"]:
                    if task.get("idempotency_hash") != idempotency_hash:
                        continue
                    if task.get("kind") != "paired_match" or task.get(
                        "request"
                    ) != request:
                        raise TaskConflict(
                            "idempotency key was already used for a different pair"
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
                "task_id": uuid.uuid4().hex, "kind": "paired_match",
                "state": "queued", "created_at": _now(),
                "updated_at": _now(), "attempt": 0,
                "request": request, "idempotency_hash": idempotency_hash,
            }
            payload["tasks"].append(task)
            self._write(payload)
            public = _public_task(task)
            self.telemetry.try_record(
                "task_submitted", task_id=public["task_id"],
                created=True, fast=fast,
            )
            return public, True

    def submit_world_model_fork(
        self, home: str, away: str, *, fast: bool,
        plan: WorldModelForkPlan | dict[str, Any], idempotency_key: str = "",
    ) -> tuple[dict[str, Any], bool]:
        for side, value in (("home", home), ("away", away)):
            if (
                not isinstance(value, str) or not value.strip()
                or len(value.strip()) > 80
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError(f"invalid {side} team")
        home, away = home.strip(), away.strip()
        if home.casefold() == away.casefold():
            raise ValueError("home and away teams must differ")
        if not isinstance(fast, bool):
            raise ValueError("fast must be boolean")
        normalized = (
            plan if isinstance(plan, WorldModelForkPlan)
            else WorldModelForkPlan.from_payload(plan)
        )
        request = {
            "home": home, "away": away, "fast": fast,
            "plan": normalized.as_dict(),
        }
        idempotency_hash = self._idempotency_hash(idempotency_key)
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            if idempotency_hash:
                for task in payload["tasks"]:
                    if task.get("idempotency_hash") != idempotency_hash:
                        continue
                    if (
                        task.get("kind") != "world_model_fork"
                        or task.get("request") != request
                    ):
                        raise TaskConflict(
                            "idempotency key was already used for a different fork"
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
                "task_id": uuid.uuid4().hex, "kind": "world_model_fork",
                "state": "queued", "created_at": _now(),
                "updated_at": _now(), "attempt": 0,
                "request": request, "idempotency_hash": idempotency_hash,
            }
            payload["tasks"].append(task)
            self._write(payload)
            public = _public_task(task)
            self.telemetry.try_record(
                "task_submitted", task_id=public["task_id"],
                created=True, fast=fast,
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

    def requeue_interrupted(self, task_id: str, *, reason: str) -> dict[str, Any]:
        """Explicitly retry an interrupted task without changing its identity.

        Failed tasks are intentionally not retryable through this method: an
        operator must first resolve the failure and start a new validation run.
        Retaining the task and idempotency hash lets recovery audits prove that
        a restart did not create a second logical transaction.
        """
        if not task_id or len(task_id) > 64 or not task_id.isalnum():
            raise ValueError("invalid product task id")
        normalized_reason = str(reason).strip()
        if not normalized_reason or len(normalized_reason) > 100:
            raise ValueError("requeue reason must be between 1 and 100 characters")
        with FileLease(self.lease_path, timeout=5.0):
            payload = self._load()
            for task in payload["tasks"]:
                if task.get("task_id") != task_id:
                    continue
                if task.get("state") != "interrupted":
                    raise RuntimeError("only interrupted product tasks may be requeued")
                task.update({
                    "state": "queued",
                    "updated_at": _now(),
                    "requeue_reason": normalized_reason,
                })
                for field in ("finished_at", "reason", "worker_id"):
                    task.pop(field, None)
                self._write(payload)
                public = _public_task(task)
                self.telemetry.try_record(
                    "task_requeued",
                    task_id=task["task_id"],
                    attempt=int(task.get("attempt", 0)),
                    reason=normalized_reason,
                )
                return public
        raise FileNotFoundError(f"product task not found: {task_id}")

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
            workspace = self.workspace_loader(self.queue.root)
            if task.get("kind") == "tactical_study":
                from src.product.tactical_study import execute_tactical_study

                plan = TacticalStudyPlan.from_payload(request["plan"])
                study = execute_tactical_study(workspace, plan)
                result = {
                    "study_id": plan.study_id,
                    "status": study["status"],
                    "study_result": self._relative(
                        self.queue.root, study["result_path"],
                    ),
                    "study_dashboard": self._relative(
                        self.queue.root, study["dashboard_path"],
                    ),
                }
            elif task.get("kind") == "season_matchday":
                season = workspace.play_next_matchday(
                    expected_season_id=str(request["season_id"]),
                    expected_matchday=int(request["matchday"]),
                    expected_revision=int(request["season_revision"]),
                )
                result = {
                    "season_id": season["season_id"],
                    "completed_matchday": int(request["matchday"]),
                    "next_matchday": season["next_matchday"],
                    "progress": season["progress"],
                    "state": season["state"],
                }
            elif task.get("kind") in {"paired_match", "world_model_fork"}:
                is_world_model_fork = task.get("kind") == "world_model_fork"
                plan = (
                    WorldModelForkPlan.from_payload(request["plan"])
                    if is_world_model_fork else
                    PairedMatchPlan.from_payload(request["plan"])
                )
                plan.validate_for_mode(workspace.config.mode)
                baseline, treatment = workspace.run_paired_matches(
                    str(request["home"]), str(request["away"]),
                    fast=bool(request["fast"]),
                    baseline_plan=plan.baseline_plan(),
                    treatment_plan=plan.treatment_plan(), seed=plan.seed,
                    transaction_id=str(task["task_id"]),
                )
                if not treatment.get("comparison_dashboard_path"):
                    raise RuntimeError("paired match did not produce a comparison")
                result = {
                    "fork_id" if is_world_model_fork else "pair_id": (
                        f"{treatment['match_id']}-paired-vs-"
                        f"{baseline['match_id']}"
                    ),
                    "fixture": treatment["fixture"],
                    "baseline_match_id": baseline["match_id"],
                    "treatment_match_id": treatment["match_id"],
                    "baseline_dashboard": self._relative(
                        self.queue.root, baseline["dashboard_path"],
                    ),
                    "treatment_dashboard": self._relative(
                        self.queue.root, treatment["dashboard_path"],
                    ),
                    "comparison": self._relative(
                        self.queue.root, treatment["comparison_path"],
                    ),
                    "comparison_dashboard": self._relative(
                        self.queue.root,
                        treatment["comparison_dashboard_path"],
                    ),
                }
                if is_world_model_fork:
                    result["propagation"] = _world_model_propagation_digest(
                        treatment["comparison_path"]
                    )
            else:
                report = workspace.run_match(
                    str(request["home"]), str(request["away"]),
                    fast=bool(request["fast"]),
                    plan=MatchPlan.from_payload(request.get("plan")),
                    seed_override=request.get("seed_override"),
                )
                result = {
                    "match_id": report["match_id"],
                    "fixture": report["fixture"],
                    "result": report["result"],
                    "integrity": report["integrity"],
                    "report": self._relative(
                        self.queue.root, report["report_path"],
                    ),
                    "dashboard": self._relative(
                        self.queue.root, report["dashboard_path"],
                    ),
                }
                if report.get("comparison_dashboard_path"):
                    result["comparison"] = self._relative(
                        self.queue.root, report["comparison_path"],
                    )
                    result["comparison_dashboard"] = self._relative(
                        self.queue.root, report["comparison_dashboard_path"],
                    )
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
