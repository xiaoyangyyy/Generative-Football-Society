from pathlib import Path
import time
from types import SimpleNamespace

from gfs_core.kernel.context import PluginContext
from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_studio.plugins.product.tasks.service import TaskPlugin, TaskService


class CooperativeWorkflows:
    project_root: Path = None

    def micro_match(self, home, away, seed, *, before_tick):
        for _ in range(1000):
            before_tick()
            time.sleep(0.001)
        return {"home": home, "away": away, "score": [seed, 0]}


def test_cancel_stops_at_a_cooperative_tick_boundary(tmp_path):
    workflows = CooperativeWorkflows()
    workflows.project_root = tmp_path
    service = TaskService(tmp_path, workflows)
    try:
        task = service.submit("cancel-me", {
            "home": "Brazil", "away": "Argentina", "seed": 42,
        })
        requested = service.cancel(task["task_id"])
        assert requested["cancel_requested"] is True
        result = service.wait(task["task_id"], timeout=5)
        assert result["state"] == "cancelled"
        assert result["error_type"] == "TaskCancelled"
    finally:
        service.close()


def test_plugin_activation_recovers_interrupted_process_tasks(tmp_path):
    WorkspaceRepository(tmp_path).write("tasks.json", {"tasks": [{
        "task_id": "orphaned",
        "idempotency_key": "orphaned",
        "kind": "micro_match",
        "payload": {},
        "state": "running",
        "result": None,
        "error_type": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }]})
    plugin = TaskPlugin(SimpleNamespace(project_root=tmp_path))
    context = PluginContext()
    plugin.activate(context)
    try:
        assert plugin.service.get("orphaned")["state"] == "interrupted"
    finally:
        plugin.deactivate()
