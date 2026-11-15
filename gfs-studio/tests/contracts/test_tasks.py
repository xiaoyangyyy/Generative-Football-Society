import json
from pathlib import Path

from gfs_studio.api import (
    submit_task,
    task_get,
    task_list,
    task_wait,
    web_gateway,
)
from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_studio.plugins.product.tasks.service import TaskService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_idempotent_task_submission_and_review():
    WorkspaceRepository(PROJECT_ROOT).write("tasks.json", {"tasks": []})
    first = submit_task(
        PROJECT_ROOT,
        home="Brazil",
        away="Argentina",
        seed=42,
        idempotency_key="match-42",
    )
    duplicate = submit_task(
        PROJECT_ROOT,
        home="Brazil",
        away="Argentina",
        seed=42,
        idempotency_key="match-42",
    )
    assert duplicate["task_id"] == first["task_id"]
    result = task_wait(PROJECT_ROOT, first["task_id"], timeout=60)
    assert result["state"] == "review"
    assert result["result"]["score"] == [0, 3]
    assert len(task_list(PROJECT_ROOT)) == 1
    assert task_get(PROJECT_ROOT, first["task_id"])["state"] == "review"


def test_recovery_marks_running_tasks_interrupted(tmp_path):
    workspace = WorkspaceRepository(tmp_path)
    workspace.write("tasks.json", {"tasks": [{
        "task_id": "running-1",
        "idempotency_key": "running-1",
        "kind": "micro_match",
        "payload": {},
        "state": "running",
        "result": None,
        "error_type": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }]})
    service = TaskService(tmp_path, None)
    assert service.recover() == 1
    assert service.get("running-1")["state"] == "interrupted"
    assert service.recover() == 0
    service.close()


def test_http_task_submission_wait_and_duplicate(tmp_path):
    WorkspaceRepository(PROJECT_ROOT).write("tasks.json", {"tasks": []})
    gateway = web_gateway(PROJECT_ROOT)
    payload = json.dumps({
        "home": "Brazil",
        "away": "Argentina",
        "seed": 42,
        "idempotency_key": "http-match-42",
    }).encode()
    first_response = gateway.handle("POST", "/api/tasks", host="127.0.0.1", body=payload)
    assert first_response.status == 200
    first = json.loads(first_response.body)
    duplicate_response = gateway.handle("POST", "/api/tasks", host="127.0.0.1", body=payload)
    duplicate = json.loads(duplicate_response.body)
    assert duplicate["task_id"] == first["task_id"]
    result_response = gateway.handle(
        "POST",
        "/api/tasks/wait",
        host="127.0.0.1",
        body=json.dumps({"task_id": first["task_id"], "timeout": 60}).encode(),
    )
    result = json.loads(result_response.body)
    assert result["state"] == "review"
    assert result["result"]["score"] == [0, 3]
    get_response = gateway.handle("GET", f"/api/tasks/{first['task_id']}", host="127.0.0.1")
    assert json.loads(get_response.body)["state"] == "review"
    gateway.close()
