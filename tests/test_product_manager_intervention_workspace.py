import copy
import hashlib
import json

import pytest

from src.product.manager_intervention_workspace import (
    build_manager_intervention_workspace,
    validate_manager_intervention_workspace,
)
from src.product.season import ManagerDecision


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _decision():
    return ManagerDecision(
        team="A", tactic="gegenpress", rotation="balanced",
    ).as_dict()


def _season(*, revision=3, decision=True, reviews=None, complete=False):
    return {
        "season_id": "season-1",
        "revision": revision,
        "next_manager_fixture": (
            None if complete else {
                "fixture_id": "md01-fx01",
                "matchday": 1,
                "home": "A",
                "away": "B",
                "manager_decision": _decision() if decision else None,
                "manager_future_reviews": list(reviews or []),
            }
        ),
    }


def _future_set(
    *,
    revision=3,
    state="completed",
    binding_current=True,
    receipt=None,
    scenario_identity="a" * 64,
):
    decision_identity = _identity(
        ManagerDecision.from_payload(_decision()).as_dict()
    )
    return {
        "task_id": "future01",
        "state": state,
        "branch_times_sec": [1800.0, 2700.0],
        "manager_context": {
            "season_id": "season-1",
            "season_revision": revision,
            "fixture_id": "md01-fx01",
            "decision_identity": decision_identity,
            "context_identity": "c" * 64,
        },
        "binding_current": binding_current,
        "scenario_evidence": (
            [] if scenario_identity is None else [
                {"scenario_identity": scenario_identity},
                {"scenario_identity": "b" * 64},
            ]
        ),
        "action_divergence_scenarios": 1,
        "local_attribution_scenarios": 1,
        "descriptive_future_difference_scenarios": 1,
        "review_receipt": receipt,
        "fork_set_url": "/artifacts/future01",
    }


def test_intervention_workspace_requires_one_frozen_decision_first():
    season = _season(decision=False)

    workspace = build_manager_intervention_workspace(season, [])

    assert workspace["workflow_state"] == "decision_required"
    assert [stage["stage_id"] for stage in workspace["stages"]] == [
        "freeze_intervention",
        "generate_bounded_futures",
        "inspect_local_mechanism",
        "record_manager_review",
        "advance_official_world",
    ]
    assert workspace["allowed_actions"] == {
        "freeze_decision": True,
        "request_future_set": False,
        "review_future_set": False,
        "advance_official_match": False,
    }
    assert workspace["causal_effect_authorized"] is False
    assert workspace["outcome_effect_estimate"] is None


def test_completed_current_future_set_becomes_replayable_review_session():
    season = _season()
    future_set = _future_set()

    workspace = build_manager_intervention_workspace(
        season, [future_set],
    )

    assert workspace["workflow_state"] == "evidence_ready_for_review"
    assert workspace["frozen_intervention"]["decision"]["tactic"] == (
        "gegenpress"
    )
    assert workspace["stages"][1]["status"] == "evidence_complete"
    assert workspace["stages"][2]["status"] == (
        "scenario_evidence_available"
    )
    assert workspace["stages"][3]["status"] == "review_available"
    assert workspace["evidence_summary"] == {
        "future_sets": 1,
        "current_future_sets": 1,
        "completed_future_sets": 1,
        "reviewed_future_sets": 0,
        "registered_scenarios": 2,
        "action_divergence_scenarios": 1,
        "local_attribution_scenarios": 1,
        "descriptive_future_difference_scenarios": 1,
    }
    assert workspace["allowed_actions"]["review_future_set"] is True
    assert workspace["allowed_actions"]["advance_official_match"] is True
    assert workspace["continuity_gaps"] == []
    validate_manager_intervention_workspace(
        workspace, season=season, manager_future_sets=[future_set],
    )

    tampered = copy.deepcopy(workspace)
    tampered["stages"][3]["status"] = "review_recorded"
    frozen = copy.deepcopy(tampered)
    frozen.pop("workspace_identity")
    tampered["workspace_identity"] = _identity(frozen)
    with pytest.raises(ValueError, match="workspace replay mismatch"):
        validate_manager_intervention_workspace(
            tampered, season=season, manager_future_sets=[future_set],
        )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("queued", "future_generation_in_progress"),
        ("running", "future_generation_in_progress"),
        ("interrupted", "future_generation_interrupted"),
        ("failed", "future_generation_failed"),
    ],
)
def test_intervention_workspace_distinguishes_live_task_states(
    state, expected,
):
    workspace = build_manager_intervention_workspace(
        _season(), [_future_set(state=state, scenario_identity=None)],
    )

    assert workspace["workflow_state"] == expected
    assert workspace["allowed_actions"]["review_future_set"] is False


def test_recorded_review_refreezes_decision_without_second_season_state():
    decision_identity = _identity(
        ManagerDecision.from_payload(_decision()).as_dict()
    )
    receipt = {
        "task_id": "future01",
        "review_identity": "d" * 64,
        "final_decision_identity": decision_identity,
        "intent": "keep_after_review",
    }
    season = _season(revision=4, reviews=[receipt])
    historical = _future_set(
        revision=3, binding_current=False, receipt=receipt,
    )

    workspace = build_manager_intervention_workspace(
        season, [historical],
    )

    assert workspace["workflow_state"] == (
        "review_recorded_decision_refrozen"
    )
    assert workspace["stages"][3]["status"] == "review_recorded"
    assert workspace["stages"][3]["source_identity"] == "d" * 64
    assert workspace["evidence_summary"]["reviewed_future_sets"] == 1
    assert workspace["allowed_actions"]["request_future_set"] is True
    assert workspace["allowed_actions"]["advance_official_match"] is True
    assert "future_set_binding_projection_mismatch" not in (
        workspace["continuity_gaps"]
    )


def test_projection_drift_and_missing_scenarios_remain_visible_gaps():
    future_set = _future_set(
        binding_current=False, scenario_identity=None,
    )

    workspace = build_manager_intervention_workspace(
        _season(), [future_set],
    )

    assert workspace["workflow_state"] == "evidence_ready_for_review"
    assert workspace["stages"][2]["status"] == "aggregate_only"
    assert workspace["continuity_gaps"] == [
        "completed_future_set_without_scenario_evidence",
        "future_set_binding_projection_mismatch",
    ]


def test_workspace_identity_ignores_transport_url_and_invalid_counts_fail_closed():
    first = _future_set()
    second = copy.deepcopy(first)
    second["fork_set_url"] = "/another/deployment/path"

    first_workspace = build_manager_intervention_workspace(
        _season(), [first],
    )
    second_workspace = build_manager_intervention_workspace(
        _season(), [second],
    )

    assert (
        first_workspace["workspace_identity"]
        == second_workspace["workspace_identity"]
    )
    invalid = copy.deepcopy(first)
    invalid["local_attribution_scenarios"] = "one"
    invalid_workspace = build_manager_intervention_workspace(
        _season(), [invalid],
    )
    assert invalid_workspace["evidence_summary"][
        "local_attribution_scenarios"
    ] == 0
    assert "future_set_aggregate_count_invalid" in (
        invalid_workspace["continuity_gaps"]
    )


def test_completed_season_has_a_closed_unavailable_workspace():
    season = _season(complete=True)

    workspace = build_manager_intervention_workspace(season, [])

    assert workspace["available"] is False
    assert workspace["workflow_state"] == "season_complete"
    assert workspace["stages"] == []
    validate_manager_intervention_workspace(
        workspace, season=season, manager_future_sets=[],
    )
