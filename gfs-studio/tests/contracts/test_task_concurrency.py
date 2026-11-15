import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_studio.plugins.product.tasks.service import TaskService


class ImmediateWorkflows:
    def micro_match(self, home, away, seed):
        time.sleep(0.002)
        return {"home": home, "away": away, "score": [seed % 2, 0]}


def payload(seed):
    return {"home": "Brazil", "away": "Argentina", "seed": seed}


def test_cross_service_submissions_do_not_lose_tasks(tmp_path):
    services = [TaskService(tmp_path, ImmediateWorkflows()), TaskService(tmp_path, ImmediateWorkflows())]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda index: services[index % 2].submit(f"task-{index}", payload(index)), range(20)))
        with ThreadPoolExecutor(max_workers=2) as pool:
            duplicates = list(pool.map(lambda service: service.submit("same", payload(42)), services))
        assert duplicates[0]["task_id"] == duplicates[1]["task_id"]
        for service in services:
            service._executor.shutdown(wait=True)
        assert len(services[0].list()) == 21
        with pytest.raises(ValueError, match="different payload"):
            services[0].submit("same", payload(43))
    finally:
        for service in services:
            service.close()


def test_interrupted_task_can_resume_execution(tmp_path):
    task_id = "interrupted-1"
    WorkspaceRepository(tmp_path).write("tasks.json", {"tasks": [{
        "task_id": task_id,
        "idempotency_key": "interrupted-1",
        "kind": "micro_match",
        "payload": payload(42),
        "state": "interrupted",
        "result": None,
        "error_type": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }]})
    service = TaskService(tmp_path, ImmediateWorkflows())
    try:
        assert service.resume(task_id) == 1
        assert service.resume(task_id) == 0
        result = service.wait(task_id, timeout=5)
        assert result["state"] == "review"
        assert result["result"]["score"] == [0, 0]
    finally:
        service.close()
