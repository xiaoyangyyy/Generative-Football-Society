import json
import threading

import pytest

from src.product.telemetry import ProductTelemetry, route_template


def _request(telemetry, *, status=200, route="/api/v1/studio", duration=1.0, code=None):
    fields = {
        "request_id": "abc123", "method": "GET", "route": route,
        "status": status, "duration_ms": duration,
    }
    if code is not None:
        fields["error_code"] = code
    telemetry.record("web_request", **fields)


def test_route_template_removes_user_controlled_components():
    assert route_template("/api/v1/tasks/private-task") == "/api/v1/tasks/{task_id}"
    assert route_template("/artifacts/outputs/private/team.html") == "/artifacts/{artifact}"
    assert route_template("/unknown/private") == "unmatched"
    assert route_template("/api/v1/studio") == "/api/v1/studio"


def test_telemetry_rejects_non_allowlisted_or_unbounded_data(tmp_path):
    telemetry = ProductTelemetry(tmp_path)
    with pytest.raises(ValueError, match="non-allowlisted"):
        telemetry.record(
            "web_request", request_id="a", method="GET", route="/",
            status=200, duration_ms=1, access_token="never-persist-this",
        )
    with pytest.raises(ValueError, match="missing required"):
        telemetry.record("web_request", request_id="a", method="GET", route="/")
    with pytest.raises(ValueError, match="normalized"):
        _request(telemetry, route="/api/v1/tasks/user-value")
    with pytest.raises(ValueError, match="bounded identifier"):
        telemetry.record(
            "task_failed", task_id="x", attempt=1, duration_ms=1,
            error_type="private path/value",
        )
    assert not telemetry.path.exists()


def test_snapshot_aggregates_without_exposing_raw_events(tmp_path):
    telemetry = ProductTelemetry(tmp_path)
    _request(telemetry, status=200, route="/api/v1/login", duration=4.0)
    _request(
        telemetry, status=401, route="/api/v1/login", duration=8.0,
        code="invalid_credentials",
    )
    _request(telemetry, status=500, duration=12.0, code="internal_error")
    telemetry.record("task_submitted", task_id="task1", created=True, fast=True)
    telemetry.record("task_claimed", task_id="task1", attempt=1)
    telemetry.record("task_failed", task_id="task1", attempt=1, duration_ms=9, error_type="RuntimeError")

    snapshot = telemetry.snapshot()
    assert snapshot["requests"]["total"] == 3
    assert snapshot["requests"]["status_classes"] == {
        "2xx": 1, "3xx": 0, "4xx": 1, "5xx": 1,
    }
    assert snapshot["requests"]["latency_ms"] == {
        "p50": 8.0, "p95": 12.0, "p99": 12.0, "max": 12.0,
    }
    assert snapshot["authentication"] == {"accepted": 1, "rejected": 1}
    assert snapshot["tasks"]["task_failed"] == 1
    assert snapshot["privacy"]["raw_events_exposed"] is False
    assert "events" not in snapshot


def test_concurrent_writers_produce_complete_json_lines(tmp_path):
    telemetry = ProductTelemetry(tmp_path)
    threads = [
        threading.Thread(
            target=_request, args=(telemetry,), kwargs={"duration": index + 0.5},
        )
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    lines = telemetry.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert all(json.loads(line)["event"] == "web_request" for line in lines)
    assert telemetry.snapshot()["retention"]["corrupt_records"] == 0


def test_bounded_rotation_and_corruption_visibility(tmp_path):
    telemetry = ProductTelemetry(tmp_path, max_bytes=512)
    for index in range(8):
        _request(telemetry, duration=index + 1)
    assert telemetry.path.stat().st_size <= 512
    assert telemetry.rotated_path.is_file()
    with telemetry.path.open("ab") as handle:
        handle.write(b"not-json\n")
        handle.write(json.dumps({
            "schema_version": 1,
            "event_id": "injected",
            "timestamp": "2026-08-09T00:00:00+00:00",
            "event": "web_request",
            "request_id": "abc", "method": "GET", "route": "/",
            "status": 200, "duration_ms": 1.0,
            "access_token": "must-be-detected-as-schema-corruption",
        }).encode() + b"\n")
    snapshot = telemetry.snapshot()
    assert snapshot["retention"]["one_rotated_segment"] is True
    assert snapshot["retention"]["corrupt_records"] == 2


def test_try_record_is_fail_open_and_observable(tmp_path, monkeypatch):
    telemetry = ProductTelemetry(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(telemetry, "record", fail)
    assert telemetry.try_record("worker_lifecycle", state="started") is False
    assert telemetry.write_failures == 1
