import copy
import json
from types import SimpleNamespace

import pytest

from src.product.manager_future import (
    build_manager_future_context,
    validate_manager_future_context,
    validate_manager_future_context_shape,
)
from src.product.manager_future_review import (
    build_manager_future_review,
    validate_manager_future_review,
)
from src.product.match_plan import WorldModelForkSetPlan
from src.product.season import (
    ManagerDecision,
    SeasonPlan,
    new_season_state,
)
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue
from src.product.workspace import ProductWorkspace, StudioConfig


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


def _task_evidence(context, task_id="future123", workspace=None):
    plan = WorldModelForkSetPlan(
        context["home_tactic"],
        context["away_tactic"],
        context["match_seed"],
        (1800, 2700, 3600),
    )
    request = {
        "home": context["fixture"]["home"],
        "away": context["fixture"]["away"],
        "fast": context["fast"],
        "plan": plan.as_dict(),
        "manager_context": context,
    }
    manager_digest = {
        "season_id": context["season_id"],
        "season_revision": context["season_revision"],
        "fixture_id": context["fixture"]["fixture_id"],
        "matchday": context["fixture"]["matchday"],
        "manager_team": context["manager_team"],
        "decision_identity": context["decision_identity"],
        "context_identity": context["context_identity"],
        "claim_boundary": context["claim_boundary"],
    }
    result = {
        "fork_set_id": task_id,
        "status": "complete",
        "manager_context": manager_digest,
        "aggregate": {
            "eligible_scenarios": 3,
            "verified_anchor_scenarios": 3,
            "action_divergence_scenarios": 2,
            "local_attribution_scenarios": 1,
            "descriptive_future_difference_scenarios": 1,
            "timing_sensitivity_observed": True,
            "status_counts": {
                "local_action_divergence_with_descriptive_future_difference": 1,
                "no_realized_action_divergence": 2,
            },
            "ranking_performed": False,
            "best_branch_time": None,
        },
        "claim_authority": {
            "descriptive_simulator_timing_sensitivity": True,
            "best_time_recommendation": False,
            "match_outcome_causality": False,
            "population_inference": False,
            "real_football_causality": False,
            "promotion_authorized": False,
        },
    }
    if workspace is not None:
        artifact_root = workspace.output_root / "fork_sets" / task_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 1,
            "set_id": task_id,
            "status": "complete",
            "fixture": {
                "home": context["fixture"]["home"],
                "away": context["fixture"]["away"],
            },
            "source_context": context,
            "plan": plan.as_dict(),
            "aggregate": result["aggregate"],
            "claim_authority": result["claim_authority"],
        }
        result_path = artifact_root / "result.json"
        dashboard_path = artifact_root / "index.html"
        result_path.write_text(json.dumps(artifact), encoding="utf-8")
        dashboard_path.write_text("future review", encoding="utf-8")
        result["fork_set_result"] = result_path.relative_to(
            workspace.root
        ).as_posix()
        result["fork_set_dashboard"] = dashboard_path.relative_to(
            workspace.root
        ).as_posix()
    return request, result


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


def test_manager_future_review_receipt_distinguishes_keep_and_revise():
    season, fixture = _season_with_decision()
    context = build_manager_future_context(season)
    request, result = _task_evidence(context)

    kept = build_manager_future_review(
        task_id="future123",
        request=request,
        result=result,
        final_decision=fixture["manager_decision"],
        intent="keep_after_review",
    )
    validate_manager_future_review(
        kept,
        season_id=season["season_id"],
        fixture_id=fixture["fixture_id"],
        matchday=fixture["matchday"],
        manager_team="A",
    )
    assert kept["decision_changed"] is False
    assert kept["evidence_summary"]["ranking_performed"] is False
    assert kept["claim_authority"]["match_outcome_causality"] is False

    revised_decision = ManagerDecision(
        team="A", tactic="balanced",
    ).as_dict()
    revised = build_manager_future_review(
        task_id="future123",
        request=request,
        result=result,
        final_decision=revised_decision,
        intent="revise_after_review",
    )
    assert revised["decision_changed"] is True
    with pytest.raises(ValueError, match="does not match decision"):
        build_manager_future_review(
            task_id="future123",
            request=request,
            result=result,
            final_decision=revised_decision,
            intent="keep_after_review",
        )

    tampered = copy.deepcopy(kept)
    tampered["evidence_summary"]["action_divergence_scenarios"] = 3
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_manager_future_review(
            tampered,
            season_id=season["season_id"],
            fixture_id=fixture["fixture_id"],
            matchday=fixture["matchday"],
            manager_team="A",
        )


def test_workspace_records_iterative_future_reviews_in_authoritative_season(
    tmp_path,
):
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Future review", mode="research", seed=7),
    )
    created = workspace.create_season(
        SeasonPlan(
            ("A", "B", "C", "D"), fast=True, manager_team="A",
        )
    )
    fixture_id = created["next_manager_fixture"]["fixture_id"]
    decided = workspace.set_manager_decision(
        ManagerDecision(team="A", tactic="gegenpress"),
        fixture_id=fixture_id,
    )
    context = workspace.manager_future_set_context(fixture_id=fixture_id)
    request, result = _task_evidence(context, "keep123", workspace)

    kept = workspace.review_manager_future_set(
        task_id="keep123",
        task_request=request,
        task_result=result,
        intent="keep_after_review",
        fixture_id=fixture_id,
        expected_revision=decided["revision"],
    )
    managed = kept["next_manager_fixture"]
    assert kept["revision"] == decided["revision"] + 1
    assert managed["manager_decision"]["tactic"] == "gegenpress"
    assert managed["manager_future_reviews"][0]["intent"] == (
        "keep_after_review"
    )
    summary = kept["manager_decision_ledger"]["summary"][
        "world_model_future_reviews"
    ]
    assert summary["reviewed_future_sets"] == 1
    assert summary["kept_after_review"] == 1
    assert summary["causal_effect_authorized"] is False
    replayed = workspace.review_manager_future_set(
        task_id="keep123",
        task_request=request,
        task_result=result,
        intent="keep_after_review",
        fixture_id=fixture_id,
        expected_revision=decided["revision"],
    )
    assert replayed["revision"] == kept["revision"]
    assert len(
        replayed["next_manager_fixture"]["manager_future_reviews"]
    ) == 1
    with pytest.raises(ValueError, match="conflicting replay"):
        workspace.review_manager_future_set(
            task_id="keep123",
            task_request=request,
            task_result=result,
            intent="revise_after_review",
            fixture_id=fixture_id,
            expected_revision=decided["revision"],
            final_decision=ManagerDecision(team="A", tactic="balanced"),
        )

    next_context = workspace.manager_future_set_context(fixture_id=fixture_id)
    next_request, next_result = _task_evidence(
        next_context, "revise123", workspace,
    )
    revised = workspace.review_manager_future_set(
        task_id="revise123",
        task_request=next_request,
        task_result=next_result,
        intent="revise_after_review",
        fixture_id=fixture_id,
        expected_revision=kept["revision"],
        final_decision=ManagerDecision(team="A", tactic="balanced"),
    )
    managed = revised["next_manager_fixture"]
    assert managed["manager_decision"]["tactic"] == "balanced"
    assert [row["intent"] for row in managed["manager_future_reviews"]] == [
        "keep_after_review", "revise_after_review",
    ]
    assert revised["manager_decision_ledger"]["summary"][
        "world_model_future_reviews"
    ]["revised_after_review"] == 1

    third_context = workspace.manager_future_set_context(
        fixture_id=fixture_id,
    )
    third_request, third_result = _task_evidence(
        third_context, "tampered123", workspace,
    )
    artifact_path = workspace.root / third_result["fork_set_result"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["aggregate"]["action_divergence_scenarios"] = 0
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    frozen = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        workspace.review_manager_future_set(
            task_id="tampered123",
            task_request=third_request,
            task_result=third_result,
            intent="keep_after_review",
            fixture_id=fixture_id,
            expected_revision=revised["revision"],
        )
    assert workspace.session_path.read_bytes() == frozen

    frozen = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match="stale"):
        workspace.review_manager_future_set(
            task_id="stale123",
            task_request=next_request,
            task_result={**next_result, "fork_set_id": "stale123"},
            intent="keep_after_review",
            fixture_id=fixture_id,
            expected_revision=kept["revision"],
        )
    assert workspace.session_path.read_bytes() == frozen
