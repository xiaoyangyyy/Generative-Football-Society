"""Privacy-bounded operational telemetry for the product surface."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.infrastructure import FileLease


LOGGER = logging.getLogger(__name__)
TELEMETRY_SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
MAX_RECORD_BYTES = 4096
MAX_READ_EVENTS = 10_000
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
ROUTES = {
    "/", "/healthz", "/login", "/readyz", "/api/v1/login",
    "/api/v1/logout",
    "/api/v1/excellence/evidence-kit.zip",
    "/api/v1/studio", "/api/v1/matches", "/api/v1/tasks",
    "/api/v1/recovery", "/api/v1/backups",
    "/api/v1/backups/{backup_id}/verify",
    "/api/v1/backups/{backup_id}/restore",
    "/api/v1/tasks/{task_id}", "/api/v1/operations",
    "/artifacts/{artifact}", "unmatched",
}
EVENT_FIELDS = {
    "web_request": {
        "request_id", "method", "route", "status", "duration_ms", "error_code",
    },
    "task_submitted": {"task_id", "created", "fast"},
    "task_claimed": {"task_id", "attempt"},
    "task_completed": {"task_id", "attempt", "duration_ms"},
    "task_failed": {"task_id", "attempt", "duration_ms", "error_type"},
    "task_recovered": {"count", "reason"},
    "worker_lifecycle": {"state"},
    "backup_created": {"file_count", "size_bytes"},
    "restore_completed": {"file_count", "replace", "task_history_reset"},
}
EVENT_REQUIRED_FIELDS = {
    event: fields - ({"error_code"} if event == "web_request" else set())
    for event, fields in EVENT_FIELDS.items()
}
ENUMS = {
    "method": {
        "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "OTHER",
    },
    "state": {"started", "stopped", "stop_timeout"},
}
INTEGER_FIELDS = {"status", "attempt", "count", "file_count", "size_bytes"}
BOOLEAN_FIELDS = {"created", "fast", "replace", "task_history_reset"}
FLOAT_FIELDS = {"duration_ms"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def route_template(path: str) -> str:
    """Collapse user-controlled path components before persistence."""
    if path in ROUTES:
        return path
    if path.startswith("/api/v1/tasks/"):
        return "/api/v1/tasks/{task_id}"
    if re.fullmatch(r"/api/v1/backups/[^/]+/verify", path):
        return "/api/v1/backups/{backup_id}/verify"
    if re.fullmatch(r"/api/v1/backups/[^/]+/restore", path):
        return "/api/v1/backups/{backup_id}/restore"
    if path.startswith("/artifacts/"):
        return "/artifacts/{artifact}"
    return "unmatched"


def _validate_field(name: str, value: Any) -> Any:
    if name in BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            raise ValueError(f"telemetry field {name} must be boolean")
        return value
    if name in INTEGER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"telemetry field {name} must be a non-negative integer")
        return value
    if name in FLOAT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"telemetry field {name} must be numeric")
        observed = round(float(value), 3)
        if not math.isfinite(observed) or observed < 0:
            raise ValueError(f"telemetry field {name} must be finite and non-negative")
        return observed
    if name == "route":
        if not isinstance(value, str) or value not in ROUTES:
            raise ValueError("telemetry route must be normalized")
        return value
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"telemetry field {name} must be a bounded identifier")
    if name in ENUMS and value not in ENUMS[name]:
        raise ValueError(f"unsupported telemetry {name}")
    return value


def _percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * proportion) - 1)
    return round(ordered[index], 3)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _valid_persisted_row(row: Any) -> bool:
    if not isinstance(row, dict) or row.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        return False
    event = row.get("event")
    allowed = EVENT_FIELDS.get(event)
    if allowed is None:
        return False
    base = {"schema_version", "event_id", "timestamp", "event"}
    observed_fields = set(row) - base
    if observed_fields - allowed or not EVENT_REQUIRED_FIELDS[event] <= observed_fields:
        return False
    if not isinstance(row.get("event_id"), str) or not IDENTIFIER.fullmatch(row["event_id"]):
        return False
    try:
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
    except (TypeError, ValueError):
        return False
    if timestamp.tzinfo is None:
        return False
    try:
        for name in observed_fields:
            _validate_field(name, row[name])
    except ValueError:
        return False
    return True


class ProductTelemetry:
    """Durable bounded events plus privacy-safe aggregate operations metrics."""

    def __init__(self, root: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root).resolve()
        self.path = self.root / "data/persistence/product_telemetry.jsonl"
        self.rotated_path = self.path.with_suffix(".jsonl.1")
        self.lease_path = self.root / "data/persistence/product_telemetry.lock"
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 512:
            raise ValueError("telemetry max_bytes must be at least 512")
        self.max_bytes = max_bytes
        self._failure_lock = threading.Lock()
        self._write_failures = 0

    @property
    def write_failures(self) -> int:
        with self._failure_lock:
            return self._write_failures

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        allowed = EVENT_FIELDS.get(event)
        if allowed is None:
            raise ValueError("unsupported product telemetry event")
        if set(fields) - allowed:
            raise ValueError("telemetry event contains non-allowlisted fields")
        if not EVENT_REQUIRED_FIELDS[event] <= set(fields):
            raise ValueError("telemetry event is missing required fields")
        normalized = {
            name: _validate_field(name, value)
            for name, value in sorted(fields.items()) if value is not None
        }
        row = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "timestamp": utc_now(),
            "event": event,
            **normalized,
        }
        encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > MAX_RECORD_BYTES:
            raise ValueError("telemetry record is too large")
        with FileLease(self.lease_path, timeout=2.0):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            current_size = self.path.stat().st_size if self.path.is_file() else 0
            if current_size and current_size + len(encoded) > self.max_bytes:
                self.rotated_path.unlink(missing_ok=True)
                os.replace(self.path, self.rotated_path)
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        return row

    def try_record(self, event: str, **fields: Any) -> bool:
        """Record without making a core product operation depend on telemetry I/O."""
        try:
            self.record(event, **fields)
        except Exception as exc:
            with self._failure_lock:
                self._write_failures += 1
            LOGGER.warning(
                "Product telemetry write failed event=%s error_type=%s",
                event, type(exc).__name__,
            )
            return False
        return True

    def _read_locked(self, *, limit: int) -> tuple[list[dict[str, Any]], int]:
        events: list[dict[str, Any]] = []
        corrupt = 0
        for path in (self.rotated_path, self.path):
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                for raw in handle:
                    if len(raw) > MAX_RECORD_BYTES:
                        corrupt += 1
                        continue
                    try:
                        row = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        corrupt += 1
                        continue
                    if not _valid_persisted_row(row):
                        corrupt += 1
                        continue
                    events.append(row)
        return events[-limit:], corrupt

    def read_events(self, *, limit: int = 1000) -> tuple[list[dict[str, Any]], int]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_READ_EVENTS:
            raise ValueError(f"telemetry limit must be between 1 and {MAX_READ_EVENTS}")
        with FileLease(self.lease_path, timeout=2.0):
            return self._read_locked(limit=limit)

    def snapshot(self, *, limit: int = 5000) -> dict[str, Any]:
        events, corrupt = self.read_events(limit=limit)
        requests = [row for row in events if row["event"] == "web_request"]
        durations = [float(row["duration_ms"]) for row in requests if "duration_ms" in row]
        status_classes = {key: 0 for key in ("2xx", "3xx", "4xx", "5xx")}
        routes: dict[str, int] = {}
        error_codes: dict[str, int] = {}
        for row in requests:
            status = int(row.get("status", 0))
            bucket = f"{status // 100}xx"
            if bucket in status_classes:
                status_classes[bucket] += 1
            route = str(row.get("route", "unmatched"))
            routes[route] = routes.get(route, 0) + 1
            code = row.get("error_code")
            if code:
                error_codes[str(code)] = error_codes.get(str(code), 0) + 1
        task_counts = {
            event: sum(row["event"] == event for row in events)
            for event in (
                "task_submitted", "task_claimed", "task_completed",
                "task_failed", "task_recovered",
            )
        }
        submitted = [row for row in events if row["event"] == "task_submitted"]
        task_counts["new_submissions"] = sum(row.get("created") is True for row in submitted)
        task_counts["idempotent_replays"] = sum(
            row.get("created") is False for row in submitted
        )
        task_counts["recovered_tasks"] = sum(
            int(row.get("count", 0))
            for row in events if row["event"] == "task_recovered"
        )
        recovery_counts = {
            event: sum(row["event"] == event for row in events)
            for event in ("backup_created", "restore_completed")
        }
        login_rows = [row for row in requests if row.get("route") == "/api/v1/login"]
        logout_rows = [row for row in requests if row.get("route") == "/api/v1/logout"]
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "privacy": {
                "raw_events_exposed": False,
                "request_bodies_recorded": False,
                "credentials_recorded": False,
                "network_identifiers_recorded": False,
                "user_fixture_names_recorded": False,
            },
            "retention": {
                "max_current_bytes": self.max_bytes,
                "one_rotated_segment": True,
                "events_scanned": len(events),
                "corrupt_records": corrupt,
                "write_failures_since_start": self.write_failures,
                "current_log_bytes": _file_size(self.path),
                "rotated_log_bytes": _file_size(self.rotated_path),
            },
            "requests": {
                "total": len(requests),
                "status_classes": status_classes,
                "routes": dict(sorted(routes.items())),
                "error_codes": dict(sorted(error_codes.items())),
                "latency_ms": {
                    "p50": _percentile(durations, 0.50),
                    "p95": _percentile(durations, 0.95),
                    "p99": _percentile(durations, 0.99),
                    "max": round(max(durations), 3) if durations else None,
                },
            },
            "authentication": {
                "accepted": sum(int(row.get("status", 0)) == 200 for row in login_rows),
                "rejected": sum(int(row.get("status", 0)) == 401 for row in login_rows),
                "rate_limited": sum(
                    int(row.get("status", 0)) == 429 for row in login_rows
                ),
                "logged_out": sum(
                    int(row.get("status", 0)) == 200 for row in logout_rows
                ),
            },
            "tasks": task_counts,
            "recovery": recovery_counts,
        }
