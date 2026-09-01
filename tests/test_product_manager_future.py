import copy
from types import SimpleNamespace

import pytest

from src.product.manager_future import (
    build_manager_future_context,
    validate_manager_future_context,
    validate_manager_future_context_shape,
)
from src.product.match_plan import WorldModelForkSetPlan
from src.product.season import (
    ManagerDecision,
    SeasonPlan,
    new_season_state,
)
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue


def _season_with_decision():
    season = new_season_state(
        SeasonPlan(
            ("A", "B", "C", "D"),
            fast=True,
            manager_team="A",
        ),
        season_id="season-0001",
        seed=7,
        created_at="2026-01-01T00:00:00Z",
    )
    fixture = next(
        row for row in season["fixtures"]
        if "A" in {row["home"], row["away"]}
    )
    fixture["manager_decision"] = ManagerDecision(
        team="A", tactic="gegenpress",
    ).as_dict()
    return season, fixture


def test_manager_future_context_reuses_exact_official_fixture_controls():
    season, fixture = _season_with_decision()

    context = build_manager_future_context(
        season, fixture_id=fixture["fixture_id"],
    )

    assert context["fixture"] == {
        key: fixture[key]
        for key in ("fixture_id", "matchday", "order", "home", "away")
    }
    assert context["match_seed"] == (
        season["seed"] + fixture["matchday"] * 100 + fixture["order"]
    )
    assert context["fast"] is True
    assert context["home_tactic"] == "gegenpress"
    assert context["away_tactic"] == "team_identity"
    assert context["tactic_source"] == (
        "manager_decision_and_opponent_preparation"
    )
    assert validate_manager_future_context_shape(context) == context
    assert validate_manager_future_context(context, season) == context


def test_manager_future_context_requires_frozen_decision_and_fails_closed():
    season, fixture = _season_with_decision()
    context = build_manager_future_context(season)

    without_decision = copy.deepcopy(season)
    without_decision["fixtures"][0]["manager_decision"] = None
    with pytest.raises(ValueError, match="freeze the manager decision"):
        build_manager_future_context(without_decision)

    stale = copy.deepcopy(season)
    stale["revision"] += 1
    with pytest.raises(ValueError, match="stale or modified"):
        validate_manager_future_context(context, stale)

    started = copy.deepcopy(season)
    started["fixtures"][0]["attempts"] = 1
    with pytest.raises(ValueError, match="execution starts"):
        build_manager_future_context(started, fixture_id=fixture["fixture_id"])

    tampered = copy.deepcopy(context)
    tampered["fixture"]["away"] = tampered["fixture"]["home"]
    with pytest.raises(ValueError, match="fixture values"):
        validate_manager_future_context_shape(tampered)

    tampered = copy.deepcopy(context)
    tampered["context_identity"] = "f" * 64
    with pytest.raises(ValueError, match="context identity"):
        validate_manager_future_context_shape(tampered)


def test_manager_future_task_rejects_control_mismatch(tmp_path):
    season, _fixture = _season_with_decision()
    context = build_manager_future_context(season)
    plan = WorldModelForkSetPlan(
        context["home_tactic"],
        context["away_tactic"],
        context["match_seed"],
        (1800, 2700),
    )
    queue = ProductTaskQueue(tmp_path)

    task, created = queue.submit_world_model_fork_set(
        context["fixture"]["home"],
        context["fixture"]["away"],
        fast=context["fast"],
        plan=plan,
        manager_context=context,
        idempotency_key="manager-future",
    )
    assert created is True
    assert task["request"]["manager_context"] == context

    with pytest.raises(ValueError, match="does not match"):
        queue.submit_world_model_fork_set(
            context["fixture"]["home"],
            context["fixture"]["away"],
            fast=context["fast"],
            plan=WorldModelForkSetPlan(
                "balanced",
                context["away_tactic"],
                context["match_seed"],
                (1800, 2700),
            ),
            manager_context=context,
        )


def test_manager_future_worker_rechecks_identity_after_execution(
    tmp_path, monkeypatch,
):
    season, _fixture = _season_with_decision()
    context = build_manager_future_context(season)
    plan = WorldModelForkSetPlan(
        context["home_tactic"],
        context["away_tactic"],
        context["match_seed"],
        (1800, 2700),
    )
    queue = ProductTaskQueue(tmp_path)
    task, _ = queue.submit_world_model_fork_set(
        context["fixture"]["home"],
        context["fixture"]["away"],
        fast=context["fast"],
        plan=plan,
        manager_context=context,
    )
    calls = []

    class Workspace:
        config = SimpleNamespace(mode="research")

        def validate_manager_future_set_context(self, candidate):
            calls.append(candidate["context_identity"])
            if len(calls) == 2:
                raise ValueError("manager future-set context is stale or modified")
            return candidate

    def execute(_workspace, _plan, **kwargs):
        assert kwargs["source_context"] == context
        return {
            "status": "complete",
            "result_path": str(tmp_path / "result.json"),
            "dashboard_path": str(tmp_path / "index.html"),
            "aggregate": {"ranking_performed": False},
            "claim_authority": {"promotion_authorized": False},
        }

    monkeypatch.setattr(
        "src.product.world_model_fork_set.execute_world_model_fork_set",
        execute,
    )
    assert BackgroundMatchWorker(
        queue, workspace_loader=lambda _root: Workspace(),
    ).run_once()
    failed = queue.get_task(task["task_id"])
    assert calls == [context["context_identity"]] * 2
    assert failed["state"] == "failed"
    assert failed.get("result") is None
    assert failed["error"]["type"] == "ValueError"
