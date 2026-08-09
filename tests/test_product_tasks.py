import threading

import pytest

from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue, TaskConflict


def test_idempotent_concurrent_submission_persists_exactly_one_task(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    observed = []

    def submit():
        observed.append(queue.submit_match(
            "Brazil", "Argentina", fast=True, idempotency_key="same-request",
        ))

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert len({item[0]["task_id"] for item in observed}) == 1
    assert sum(item[1] for item in observed) == 1
    assert len(queue.list_tasks()) == 1
    assert "idempotency_hash" not in queue.list_tasks()[0]


def test_claim_and_completion_require_the_owning_worker(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    submitted, _ = queue.submit_match("Brazil", "Argentina", fast=False)
    claimed = queue.claim_next("worker-a")
    assert claimed["task_id"] == submitted["task_id"]
    assert claimed["state"] == "running"
    try:
        queue.complete(claimed["task_id"], "worker-b", {"score": "0-0"})
    except RuntimeError as exc:
        assert "not owned" in str(exc)
    else:
        raise AssertionError("non-owning worker completed a task")
    completed = queue.complete(claimed["task_id"], "worker-a", {"score": "1-0"})
    assert completed["state"] == "completed"
    assert completed["result"]["score"] == "1-0"


def test_worker_executes_domain_match_and_persists_only_relative_artifacts(
    tmp_path, monkeypatch,
):
    report = tmp_path / "outputs/studio/demo/matches/0001.json"
    dashboard = report.with_suffix(".html")

    class Workspace:
        def run_match(self, home, away, *, fast):
            return {
                "match_id": "0001",
                "fixture": {"home": home, "away": away, "fast": fast},
                "result": {"score": {"home": 2, "away": 1}},
                "integrity": {"accepted": True},
                "report_path": str(report), "dashboard_path": str(dashboard),
            }

    monkeypatch.setattr(
        "src.product.tasks.ProductWorkspace.load", lambda _root: Workspace(),
    )
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_match("Brazil", "Argentina", fast=True)
    assert BackgroundMatchWorker(queue).run_once()
    completed = queue.get_task(task["task_id"])
    assert completed["state"] == "completed"
    assert completed["result"]["report"] == "outputs/studio/demo/matches/0001.json"
    assert completed["result"]["dashboard"].endswith("/0001.html")
    assert str(tmp_path) not in str(completed)


def test_worker_failure_is_terminal_and_does_not_persist_exception_text(
    tmp_path, monkeypatch,
):
    class Workspace:
        def run_match(self, *args, **kwargs):
            raise RuntimeError("secret local path C:/private/workspace")

    monkeypatch.setattr(
        "src.product.tasks.ProductWorkspace.load", lambda _root: Workspace(),
    )
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_match("Brazil", "Argentina", fast=True)
    assert BackgroundMatchWorker(queue).run_once()
    failed = queue.get_task(task["task_id"])
    assert failed["state"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"
    assert "private" not in str(failed)
    telemetry_text = queue.telemetry.path.read_text(encoding="utf-8")
    assert "secret local path" not in telemetry_text
    assert "task_failed" in telemetry_text


def test_task_lifecycle_is_aggregated_without_fixture_names(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    task, created = queue.submit_match(
        "Private Home Name", "Private Away Name", fast=True,
        idempotency_key="private-browser-key",
    )
    duplicate, duplicate_created = queue.submit_match(
        "Private Home Name", "Private Away Name", fast=True,
        idempotency_key="private-browser-key",
    )
    assert created and not duplicate_created and duplicate["task_id"] == task["task_id"]
    claimed = queue.claim_next("worker-a")
    queue.complete(claimed["task_id"], "worker-a", {"score": "1-0"})

    metrics = queue.telemetry.snapshot()["tasks"]
    assert metrics["new_submissions"] == 1
    assert metrics["idempotent_replays"] == 1
    assert metrics["task_claimed"] == 1
    assert metrics["task_completed"] == 1
    raw = queue.telemetry.path.read_text(encoding="utf-8")
    assert "Private Home Name" not in raw
    assert "Private Away Name" not in raw
    assert "private-browser-key" not in raw


def test_startup_recovery_marks_running_tasks_interrupted_without_requeue(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    submitted, _ = queue.submit_match("Brazil", "Argentina", fast=True)
    queue.claim_next("crashed-worker")
    assert queue.recover_running() == 1
    recovered = queue.get_task(submitted["task_id"])
    assert recovered["state"] == "interrupted"
    assert recovered["reason"] == "web_worker_restarted"
    assert queue.claim_next("new-worker") is None


@pytest.mark.parametrize("home,away,fast", [
    ("", "Argentina", True),
    ("Brazil", "brazil", True),
    ("Brazil", "Argentina", "yes"),
])
def test_queue_validates_match_contract_without_relying_on_web(
    tmp_path, home, away, fast,
):
    with pytest.raises(ValueError):
        ProductTaskQueue(tmp_path).submit_match(home, away, fast=fast)


def test_queue_prunes_only_oldest_terminal_history_at_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr("src.product.tasks.MAX_RETAINED_TASKS", 3)
    monkeypatch.setattr("src.product.tasks.PRUNE_TO_TASKS", 2)
    queue = ProductTaskQueue(tmp_path)
    first_ids = []
    for index in range(3):
        task, _ = queue.submit_match(f"Home {index}", f"Away {index}", fast=True)
        first_ids.append(task["task_id"])
        claimed = queue.claim_next("worker")
        queue.complete(claimed["task_id"], "worker", {"index": index})
    newest, _ = queue.submit_match("Brazil", "Argentina", fast=True)
    ids = {task["task_id"] for task in queue.list_tasks()}
    assert len(ids) == 3
    assert first_ids[0] not in ids
    assert newest["task_id"] in ids


def test_queue_fails_closed_when_capacity_contains_only_active_tasks(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("src.product.tasks.MAX_RETAINED_TASKS", 2)
    monkeypatch.setattr("src.product.tasks.PRUNE_TO_TASKS", 1)
    queue = ProductTaskQueue(tmp_path)
    queue.submit_match("A", "B", fast=True)
    queue.submit_match("C", "D", fast=True)
    with pytest.raises(RuntimeError, match="full"):
        queue.submit_match("E", "F", fast=True)


def test_idempotency_key_reuse_with_different_request_is_a_conflict(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    queue.submit_match(
        "Brazil", "Argentina", fast=True, idempotency_key="request-1",
    )
    with pytest.raises(TaskConflict, match="different match"):
        queue.submit_match(
            "France", "Spain", fast=True, idempotency_key="request-1",
        )
