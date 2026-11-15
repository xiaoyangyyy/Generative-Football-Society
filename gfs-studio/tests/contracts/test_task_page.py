from pathlib import Path

from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_studio.api import web_gateway

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_task_page_lists_persistent_task_states():
    WorkspaceRepository(PROJECT_ROOT).write("tasks.json", {"tasks": [{
        "task_id": "task-1",
        "idempotency_key": "match-42",
        "kind": "micro_match",
        "payload": {},
        "state": "review",
        "result": {},
        "error_type": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }]})
    gateway = web_gateway(PROJECT_ROOT)
    response = gateway.handle("GET", "/tasks", host="127.0.0.1")
    assert response.status == 200
    assert b"match-42" in response.body
    assert b"review" in response.body
    assert b"/api/tasks/task-1" in response.body
    gateway.close()


def test_task_page_escapes_task_keys():
    WorkspaceRepository(PROJECT_ROOT).write("tasks.json", {"tasks": [{
        "task_id": "task-xss",
        "idempotency_key": "<script>alert(1)</script>",
        "kind": "micro_match",
        "payload": {},
        "state": "failed",
        "result": None,
        "error_type": "RuntimeError",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }]})
    gateway = web_gateway(PROJECT_ROOT)
    response = gateway.handle("GET", "/tasks", host="127.0.0.1")
    assert b"<script>alert(1)</script>" not in response.body
    assert b"&lt;script&gt;" in response.body
