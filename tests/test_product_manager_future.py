import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from src.product.decision_ledger import (
    build_manager_decision_ledger,
    validate_manager_decision_ledger,
)
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
from src.product.world_model_fork_set import (
    project_fork_set_scenario_evidence,
)


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _mechanism_example(index, *, local):
    windows = []
    for duration in (30, 120):
        window = {
            "schema_version": 1,
            "window_sec": duration,
            "delta": {
                "actions": 0,
                "passes": -1 if duration == 30 else 1,
                "shots": 1,
                "goals": 0,
                "turnovers": 1,
            },
            "additional_policy_changes": index,
            "causal_effect_authorized": False,
        }
        windows.append({**window, "window_identity": _identity(window)})
    payload = {
        "schema_version": 1,
        "opportunity_identity": format(index + 100, "064x"),
        "team": "A",
        "t_sec": float(1810 + index),
        "clock": f"30:{10 + index:02d}",
        "baseline_action": "hold",
        "treatment_action": "pass",
        "recommended_action": "pass",
        "directly_observed": True,
        "local_policy_attribution_eligible": local,
        "downstream_windows": windows,
        "downstream_causal_attribution_authorized": False,
    }
    return {**payload, "example_identity": _identity(payload)}


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
                "action_divergence_without_local_attribution": 1,
                "no_realized_action_divergence": 1,
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
    rows = []
    for index, branch in enumerate(plan.branch_times_sec):
        changed = index < 2
        row = {
            "branch_at_sec": float(branch),
            "branch_minute": float(branch) / 60.0,
            "future_status": (
                "local_action_divergence_with_descriptive_future_difference"
                if index == 0 else
                "action_divergence_without_local_attribution"
                if index == 1 else
                "no_realized_action_divergence"
            ),
            "eligible": True,
            "branch_anchor_verified": True,
            "branch_state_identity": format(index + 1, "064x"),
            "changed_actions": 1 if changed else 0,
            "locally_attributable_changes": 1 if index == 0 else 0,
            "descriptive_future_difference_count": 1 if index == 0 else 0,
            "simulator_local_action_attribution": index == 0,
            "outcome_causality": False,
            "real_football_causality": False,
        }
        row["mechanism_examples"] = (
            [_mechanism_example(index, local=index == 0)]
            if changed else []
        )
        row["mechanism_examples_truncated"] = False
        rows.append(row)
    result["scenario_evidence"] = project_fork_set_scenario_evidence({
        "plan": plan.as_dict(), "rows": rows,
    })
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
            "rows": rows,
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
    assert kept["schema_version"] == 3
    assert len(kept["scenario_evidence"]) == 3
    assert kept["scenario_evidence"][0]["schema_version"] == 2
    assert kept["scenario_evidence"][0]["mechanism_examples"][0][
        "baseline_action"
    ] == "hold"
    assert kept["evidence_summary"]["ranking_performed"] is False
    assert kept["claim_authority"]["match_outcome_causality"] is False
    legacy = copy.deepcopy(kept)
    legacy.pop("scenario_evidence")
    legacy.pop("review_identity")
    legacy["schema_version"] = 1
    legacy["review_identity"] = hashlib.sha256(json.dumps(
        legacy, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    validate_manager_future_review(
        legacy,
        season_id=season["season_id"],
        fixture_id=fixture["fixture_id"],
        matchday=fixture["matchday"],
        manager_team="A",
    )
    version_two = copy.deepcopy(kept)
    for scenario in version_two["scenario_evidence"]:
        scenario.pop("mechanism_examples")
        scenario.pop("mechanism_examples_truncated")
        scenario.pop("scenario_identity")
        scenario["schema_version"] = 1
        scenario["scenario_identity"] = _identity(scenario)
    version_two.pop("review_identity")
    version_two["schema_version"] = 2
    version_two["review_identity"] = _identity(version_two)
    validate_manager_future_review(
        version_two,
        season_id=season["season_id"],
        fixture_id=fixture["fixture_id"],
        matchday=fixture["matchday"],
        manager_team="A",
    )
    scenario_tamper = copy.deepcopy(kept)
    scenario_tamper["scenario_evidence"][0]["mechanism_examples"][0][
        "baseline_action"
    ] = "shot"
    with pytest.raises(ValueError, match="mechanism example identity"):
        validate_manager_future_review(
            scenario_tamper,
            season_id=season["season_id"],
            fixture_id=fixture["fixture_id"],
            matchday=fixture["matchday"],
            manager_team="A",
        )
    rehashed_summary_tamper = copy.deepcopy(kept)
    rehashed_summary_tamper["evidence_summary"][
        "action_divergence_scenarios"
    ] = 1
    rehashed_summary_tamper.pop("review_identity")
    rehashed_summary_tamper["review_identity"] = hashlib.sha256(json.dumps(
        rehashed_summary_tamper, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="scenario summary mismatch"):
        validate_manager_future_review(
            rehashed_summary_tamper,
            season_id=season["season_id"],
            fixture_id=fixture["fixture_id"],
            matchday=fixture["matchday"],
            manager_team="A",
        )
    inconsistent = copy.deepcopy(result)
    second = inconsistent["scenario_evidence"][1]
    second["changed_actions"] = 0
    second["future_status"] = "no_realized_action_divergence"
    second["mechanism_examples"] = []
    second["mechanism_examples_truncated"] = False
    second.pop("scenario_identity")
    second["scenario_identity"] = _identity(second)
    with pytest.raises(ValueError, match="scenario aggregate is inconsistent"):
        build_manager_future_review(
            task_id="future123",
            request=request,
            result=inconsistent,
            final_decision=fixture["manager_decision"],
            intent="keep_after_review",
        )

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
    with pytest.raises(ValueError, match="scenario summary mismatch"):
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
    assert summary["retained_mechanism_examples"] == 2
    assert summary["locally_attributable_mechanism_examples"] == 1
    assert summary["examples_with_descriptive_windows"] == 2
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


def test_future_reviews_close_to_selection_and_runtime_without_outcome_claim():
    season, fixture = _season_with_decision()
    context = build_manager_future_context(season)
    request, result = _task_evidence(context, "futurekeep")
    kept = build_manager_future_review(
        task_id="futurekeep",
        request=request,
        result=result,
        final_decision=fixture["manager_decision"],
        intent="keep_after_review",
    )
    season["revision"] += 1
    revised_context = build_manager_future_context(season)
    revised_request, revised_result = _task_evidence(
        revised_context, "futurerevise",
    )
    revised_decision = ManagerDecision(
        team="A", tactic="balanced",
    ).as_dict()
    revised = build_manager_future_review(
        task_id="futurerevise",
        request=revised_request,
        result=revised_result,
        final_decision=revised_decision,
        intent="revise_after_review",
    )
    fixture["manager_decision"] = revised_decision
    fixture["manager_future_reviews"] = [kept, revised]

    pending = build_manager_decision_ledger(season)
    entry = next(
        row for row in pending["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    trace = entry["future_review_execution_trace"]
    assert [row["relation_to_final_selection"] for row in trace[
        "review_chain"
    ]] == ["followed_by_later_review", "selected_for_fixture"]
    assert trace["end_to_end_state"] == (
        "reviewed_selection_awaiting_execution"
    )
    terminal = trace["terminal_review"]
    assert terminal["intent"] == "revise_after_review"
    assert terminal["evidence_level"] == "scenario_evidence"
    assert terminal["fixed_scenario_budget"] == 3
    assert terminal["eligible_scenarios"] == 3
    assert terminal["verified_anchor_scenarios"] == 3
    assert terminal["action_divergence_scenarios"] == 2
    assert terminal["local_attribution_scenarios"] == 1
    assert terminal["descriptive_future_difference_scenarios"] == 1
    assert terminal["timing_sensitivity_observed"] is True
    assert terminal["ranking_performed"] is False
    assert terminal["best_branch_time"] is None
    assert terminal["reviewed_scenarios"] == [
        {
            "schema_version": 1,
            "source_scenario_identity": row["scenario_identity"],
            "branch_at_sec": row["branch_at_sec"],
            "branch_minute": row["branch_minute"],
            "future_status": row["future_status"],
            "eligible": row["eligible"],
            "anchor_verified": row["anchor_verified"],
            "branch_state_identity": row["branch_state_identity"],
            "changed_actions": row["changed_actions"],
            "locally_attributable_changes": row[
                "locally_attributable_changes"
            ],
            "descriptive_future_difference_count": row[
                "descriptive_future_difference_count"
            ],
            "simulator_local_action_attribution": row[
                "simulator_local_action_attribution"
            ],
            "outcome_causality_authorized": False,
            "real_football_causality_authorized": False,
            "archive_identity": terminal["reviewed_scenarios"][index][
                "archive_identity"
            ],
        }
        for index, row in enumerate(revised["scenario_evidence"])
    ]
    assert trace["outcome_comparison_performed"] is False
    assert trace["outcome_effect_estimate"] is None
    assert trace["causal_effect_authorized"] is False
    pending_certificate = entry["world_evolution_thread"][
        "review_to_official_world"
    ]
    assert pending_certificate["status"] == "awaiting_official_match"
    assert entry["world_evolution_thread"][
        "reviewed_world_model_chain_complete"
    ] is False
    assert pending_certificate[
        "scenario_to_runtime_opportunity_matching_performed"
    ] is False

    completed = copy.deepcopy(season)
    completed_fixture = next(
        row for row in completed["fixtures"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    completed_fixture["state"] = "completed"
    completed_fixture["match_id"] = "match-review-runtime"
    completed_fixture["score"] = {"home": 1, "away": 0}
    completed_fixture["report"] = "reports/match-review-runtime.json"
    debrief = {
        "schema_version": 1,
        "available": True,
        "match_id": "match-review-runtime",
        "team": "A",
        "decision": revised_decision,
        "observed_result": {
            "score": {"home": 1, "away": 0},
            "descriptive_only": True,
        },
        "evidence_grade": "direct_runtime_execution",
        "causal_outcome_attribution": False,
        "tactical_binding": {
            "schema_version": 1,
            "available": True,
            "applied_tactic": "balanced",
            "binding_identity": "a" * 64,
            "initial_vector_identity": "b" * 64,
            "changed_controls": [],
            "final_delta_l1": 0.0,
        },
    }
    executed = build_manager_decision_ledger(
        completed,
        execution_by_fixture={fixture["fixture_id"]: debrief},
    )
    executed_entry = next(
        row for row in executed["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    executed_trace = executed_entry["future_review_execution_trace"]
    assert executed_trace["end_to_end_state"] == (
        "reviewed_selection_runtime_verified"
    )
    assert executed_trace["runtime_binding"]["status"] == "verified"
    assert executed_trace["runtime_binding"]["applied_tactic"] == "balanced"
    assert executed_entry["world_evolution_thread"]["stages"][0][
        "action_divergence_scenarios"
    ] == 2
    assert executed_entry["world_evolution_thread"][
        "reviewed_world_model_chain_complete"
    ] is False
    assert executed_entry["world_evolution_thread"][
        "review_to_official_world"
    ]["status"] == "official_action_evidence_unavailable"
    summary = executed["summary"]["world_model_future_review_execution"]
    assert summary["reviewed_selection_runtime_verified"] == 1
    assert summary["outcome_comparison_performed"] is False
    assert summary["causal_effect_authorized"] is False

    superseded = copy.deepcopy(season)
    superseded_fixture = next(
        row for row in superseded["fixtures"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    superseded_fixture["manager_decision"] = ManagerDecision(
        team="A", tactic="direct_vertical",
    ).as_dict()
    superseded_ledger = build_manager_decision_ledger(superseded)
    superseded_entry = next(
        row for row in superseded_ledger["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    superseded_trace = superseded_entry["future_review_execution_trace"]
    assert superseded_trace["review_chain"][-1][
        "relation_to_final_selection"
    ] == "superseded_by_unreviewed_edit"
    assert superseded_trace["end_to_end_state"] == (
        "terminal_review_superseded_before_execution"
    )
    assert superseded_entry["world_evolution_thread"][
        "review_to_official_world"
    ]["status"] == "review_superseded"

    tampered = copy.deepcopy(executed)
    tampered_entry = next(
        row for row in tampered["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    tampered_entry["future_review_execution_trace"][
        "end_to_end_state"
    ] = "reviewed_selection_awaiting_execution"
    frozen_entry = copy.deepcopy(tampered_entry)
    frozen_entry.pop("entry_identity")
    tampered_entry["entry_identity"] = _identity(frozen_entry)
    frozen_ledger = copy.deepcopy(tampered)
    frozen_ledger.pop("ledger_identity")
    tampered["ledger_identity"] = _identity(frozen_ledger)
    with pytest.raises(ValueError, match="trace replay mismatch"):
        validate_manager_decision_ledger(tampered)
