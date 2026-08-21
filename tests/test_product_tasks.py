import threading

import pytest

from src.product.match_plan import MatchPlan, PairedMatchPlan
from src.product.tactical_study import TacticalStudyPlan
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
    comparison = report.with_name("0001.comparison.json")
    comparison_dashboard = comparison.with_suffix(".html")
    observed = {}

    class Workspace:
        def run_match(self, home, away, *, fast, plan, seed_override):
            observed.update({
                "plan": plan.as_dict(), "seed_override": seed_override,
            })
            return {
                "match_id": "0001",
                "fixture": {"home": home, "away": away, "fast": fast},
                "result": {"score": {"home": 2, "away": 1}},
                "integrity": {"accepted": True},
                "report_path": str(report), "dashboard_path": str(dashboard),
                "comparison_path": str(comparison),
                "comparison_dashboard_path": str(comparison_dashboard),
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
    assert completed["result"]["comparison"].endswith(".comparison.json")
    assert completed["result"]["comparison_dashboard"].endswith(
        ".comparison.html"
    )
    assert str(tmp_path) not in str(completed)
    assert observed["plan"]["experience"] == "observational"
    assert observed["seed_override"] is None


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


def test_interrupted_task_can_be_requeued_without_duplicate_identity(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    task, created = queue.submit_match(
        "Brazil", "Argentina", fast=True, idempotency_key="soak-001",
    )
    assert created is True
    claimed = queue.claim_next("worker-a")
    assert claimed["task_id"] == task["task_id"]
    assert queue.recover_running(reason="process_restart") == 1

    retried = queue.requeue_interrupted(
        task["task_id"], reason="production_validation_resume",
    )
    assert retried["task_id"] == task["task_id"]
    assert retried["state"] == "queued"
    duplicate, created = queue.submit_match(
        "Brazil", "Argentina", fast=True, idempotency_key="soak-001",
    )
    assert created is False
    assert duplicate["task_id"] == task["task_id"]
    assert len(queue.list_tasks()) == 1


def test_requeue_rejects_non_interrupted_or_invalid_requests(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_match("Brazil", "Argentina", fast=True)
    with pytest.raises(RuntimeError, match="only interrupted"):
        queue.requeue_interrupted(task["task_id"], reason="not_interrupted")
    with pytest.raises(ValueError, match="reason"):
        queue.requeue_interrupted(task["task_id"], reason="")


def test_season_matchday_task_is_idempotent_and_worker_uses_frozen_identity(
    tmp_path, monkeypatch,
):
    observed = {}

    class Workspace:
        def play_next_matchday(self, **kwargs):
            observed.update(kwargs)
            return {
                "season_id": "season-0001", "state": "active",
                "next_matchday": 2,
                "progress": {"completed": 2, "total": 6},
            }

    monkeypatch.setattr(
        "src.product.tasks.ProductWorkspace.load", lambda _root: Workspace(),
    )
    queue = ProductTaskQueue(tmp_path)
    first, created = queue.submit_season_matchday(
        season_id="season-0001", matchday=1, season_revision=3,
        idempotency_key="season-0001:matchday:1",
    )
    repeated, repeated_created = queue.submit_season_matchday(
        season_id="season-0001", matchday=1, season_revision=3,
        idempotency_key="season-0001:matchday:1",
    )
    assert created and not repeated_created
    assert repeated["task_id"] == first["task_id"]
    assert BackgroundMatchWorker(queue).run_once()
    completed = queue.get_task(first["task_id"])
    assert completed["kind"] == "season_matchday"
    assert completed["result"]["completed_matchday"] == 1
    assert observed == {
        "expected_season_id": "season-0001", "expected_matchday": 1,
        "expected_revision": 3,
    }


def test_season_matchday_idempotency_key_cannot_change_identity(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    queue.submit_season_matchday(
        season_id="season-0001", matchday=1, season_revision=0,
        idempotency_key="same",
    )
    with pytest.raises(TaskConflict, match="different matchday"):
        queue.submit_season_matchday(
            season_id="season-0001", matchday=2, season_revision=0,
            idempotency_key="same",
        )


def test_interrupted_paired_task_requeues_same_recoverable_identity(
    tmp_path,
):
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_paired_match(
        "Brazil", "Argentina", fast=True, plan=_paired_plan(),
    )
    queue.claim_next("crashed-pair-worker")
    assert queue.recover_running(reason="process_restart") == 1
    requeued = queue.requeue_interrupted(
        task["task_id"], reason="resume_persisted_pair_transaction",
    )
    assert requeued["task_id"] == task["task_id"]
    assert requeued["state"] == "queued"


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


def test_idempotency_contract_includes_tactics_and_resolved_seed(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    baseline = MatchPlan(
        experience="tactical_lab", home_tactic="gegenpress",
    )
    task, created = queue.submit_match(
        "Brazil", "Argentina", fast=True, plan=baseline,
        seed_override=77, idempotency_key="paired-plan",
    )
    duplicate, duplicate_created = queue.submit_match(
        "Brazil", "Argentina", fast=True, plan=baseline,
        seed_override=77, idempotency_key="paired-plan",
    )
    assert created and not duplicate_created
    assert duplicate["task_id"] == task["task_id"]
    changed = MatchPlan(
        experience="tactical_lab", home_tactic="counter_attack",
    )
    with pytest.raises(TaskConflict):
        queue.submit_match(
            "Brazil", "Argentina", fast=True, plan=changed,
            seed_override=77, idempotency_key="paired-plan",
        )


def _study_plan():
    return TacticalStudyPlan(
        study_id="queue-study-v1", home="Brazil", away="Argentina",
        baseline_home_tactic="gegenpress",
        baseline_away_tactic="low_block_counter",
        treatment_home_tactic="counter_attack",
        treatment_away_tactic="low_block_counter",
        seeds=(11, 12, 13, 14),
    )


def _paired_plan(seed=77):
    return PairedMatchPlan(
        baseline_home_tactic="balanced",
        baseline_away_tactic="low_block_counter",
        treatment_home_tactic="gegenpress",
        treatment_away_tactic="low_block_counter",
        seed=seed,
    )


def test_paired_match_submission_is_normalized_and_idempotent(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    task, created = queue.submit_paired_match(
        "Brazil", "Argentina", fast=True, plan=_paired_plan(),
        idempotency_key="pair-request",
    )
    duplicate, duplicate_created = queue.submit_paired_match(
        "Brazil", "Argentina", fast=True, plan=_paired_plan().as_dict(),
        idempotency_key="pair-request",
    )
    assert created and not duplicate_created
    assert task["kind"] == "paired_match"
    assert task["request"]["plan"]["seed"] == 77
    assert duplicate["task_id"] == task["task_id"]
    with pytest.raises(TaskConflict, match="different pair"):
        queue.submit_paired_match(
            "Brazil", "Argentina", fast=True, plan=_paired_plan(seed=78),
            idempotency_key="pair-request",
        )


def test_worker_executes_one_locked_pair_and_persists_only_artifact_links(
    tmp_path, monkeypatch,
):
    matches = tmp_path / "outputs/studio/demo/matches"
    observed = {}

    class Workspace:
        config = type("Config", (), {"mode": "research"})()

        def run_paired_matches(
            self, home, away, *, fast, baseline_plan, treatment_plan, seed,
            transaction_id,
        ):
            observed.update({
                "fixture": (home, away), "fast": fast, "seed": seed,
                "baseline": baseline_plan, "treatment": treatment_plan,
                "transaction_id": transaction_id,
            })
            baseline = {
                "match_id": "0001-brazil-vs-argentina",
                "dashboard_path": str(matches / "0001.html"),
            }
            treatment = {
                "match_id": "0002-brazil-vs-argentina",
                "fixture": {"home": home, "away": away, "seed": seed},
                "dashboard_path": str(matches / "0002.html"),
                "comparison_path": str(matches / "0002-pair.comparison.json"),
                "comparison_dashboard_path": str(
                    matches / "0002-pair.comparison.html"
                ),
            }
            return baseline, treatment

    monkeypatch.setattr(
        "src.product.tasks.ProductWorkspace.load", lambda _root: Workspace(),
    )
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_paired_match(
        "Brazil", "Argentina", fast=True, plan=_paired_plan(),
    )
    assert BackgroundMatchWorker(queue).run_once()
    completed = queue.get_task(task["task_id"])
    assert completed["state"] == "completed"
    assert completed["result"]["baseline_dashboard"].endswith("/0001.html")
    assert completed["result"]["treatment_dashboard"].endswith("/0002.html")
    assert completed["result"]["comparison_dashboard"].endswith(
        ".comparison.html"
    )
    assert "analysis" not in completed["result"]
    assert str(tmp_path) not in str(completed)
    assert observed["seed"] == 77
    assert observed["transaction_id"] == task["task_id"]
    assert observed["baseline"].reuse_last_seed is False
    assert observed["treatment"].reuse_last_seed is True


def test_tactical_study_submission_is_normalized_and_idempotent(tmp_path):
    queue = ProductTaskQueue(tmp_path)
    task, created = queue.submit_tactical_study(
        _study_plan(), idempotency_key="study-request",
    )
    duplicate, duplicate_created = queue.submit_tactical_study(
        _study_plan().as_dict(), idempotency_key="study-request",
    )
    assert created and not duplicate_created
    assert task["kind"] == "tactical_study"
    assert task["request"]["plan"]["fixed_pair_budget"] == 4
    assert duplicate["task_id"] == task["task_id"]


def test_worker_executes_tactical_study_without_putting_analysis_in_task(
    tmp_path, monkeypatch,
):
    result_path = tmp_path / "outputs/studio/lab/studies/queue-study-v1/result.json"
    dashboard_path = result_path.with_name("index.html")
    monkeypatch.setattr(
        "src.product.tasks.ProductWorkspace.load", lambda _root: object(),
    )
    observed = {}

    def fake_execute(workspace, plan):
        observed.update({"workspace": workspace, "plan": plan})
        return {
            "status": "complete", "analysis": {"must_not_cross": True},
            "result_path": str(result_path),
            "dashboard_path": str(dashboard_path),
        }

    monkeypatch.setattr(
        "src.product.tactical_study.execute_tactical_study", fake_execute,
    )
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_tactical_study(_study_plan())
    assert BackgroundMatchWorker(queue).run_once()
    completed = queue.get_task(task["task_id"])
    assert completed["state"] == "completed"
    assert completed["result"]["study_result"].endswith("/result.json")
    assert completed["result"]["study_dashboard"].endswith("/index.html")
    assert "analysis" not in completed["result"]
    assert observed["plan"].study_id == "queue-study-v1"
