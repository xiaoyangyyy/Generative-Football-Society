"""Coach action adoption is prospective, bounded, and explicitly non-causal."""

from __future__ import annotations

from types import SimpleNamespace

from src.match_engine.world_model.decision_adoption import (
    decision_adoption_diagnostics,
    finalize_decision_adoption,
    observe_executed_action,
    register_coach_action_decision,
)


def test_adoption_only_counts_future_matching_team_action():
    state = SimpleNamespace(clock_seconds=10.0)
    record = register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="xg_swing",
        llm_selected_action="shot",
        world_model_recommended_action="shot",
        recommendation_confidence=0.6,
        horizon_s=10.0,
    )
    observe_executed_action(
        state, team_id="A", action_kind="shot", t_sec=10.0,
    )
    assert record["adopted"] is None
    observe_executed_action(
        state, team_id="A", action_kind="pass", t_sec=15.0,
    )
    assert record["adopted"] is None
    observe_executed_action(
        state, team_id="B", action_kind="shot", t_sec=16.0,
    )
    assert record["adopted"] is None
    observe_executed_action(
        state, team_id="A", action_kind="shot", t_sec=18.0,
    )
    diagnostics = decision_adoption_diagnostics(state)
    assert record["adopted"] is True
    assert diagnostics["adoption_rate"] == 1.0
    assert diagnostics["causal_ordering"] == "decision_then_future_team_action"


def test_horizon_expiry_and_match_end_resolve_unmatched_decisions():
    state = SimpleNamespace(clock_seconds=20.0)
    expired = register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="goal",
        llm_selected_action="cross",
        world_model_recommended_action="pass",
        recommendation_confidence=0.4,
        horizon_s=5.0,
    )
    observe_executed_action(
        state, team_id="B", action_kind="hold", t_sec=26.0,
    )
    assert expired["adopted"] is False
    assert expired["resolution"] == "horizon_expired"
    pending = register_coach_action_decision(
        state,
        team_id="B",
        trigger_kind="clock",
        llm_selected_action="hold",
        world_model_recommended_action="hold",
        recommendation_confidence=0.5,
        horizon_s=20.0,
    )
    finalize_decision_adoption(state, t_sec=30.0)
    assert pending["adopted"] is False
    assert pending["resolution"] == "match_ended"
    assert "does not prove" in decision_adoption_diagnostics(state)[
        "interpretation"
    ]
