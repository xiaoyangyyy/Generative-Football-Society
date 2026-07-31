"""Coach action adoption is prospective, bounded, and explicitly non-causal."""

from __future__ import annotations

from types import SimpleNamespace

from src.match_engine.world_model.decision_adoption import (
    decision_adoption_diagnostics,
    finalize_decision_adoption,
    observe_executed_action,
    pending_policy_action_bias,
    record_policy_intervention_result,
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


def test_bounded_policy_intent_is_future_same_team_feasible_and_one_shot():
    state = SimpleNamespace(clock_seconds=40.0)
    record = register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="xg_swing",
        llm_selected_action="shot",
        world_model_recommended_action="pass",
        recommendation_confidence=0.7,
        horizon_s=10.0,
        intervention_enabled=True,
        intervention_strength=9.0,
    )
    assert record["intervention_strength"] == 0.5
    assert pending_policy_action_bias(
        state, team_id="A", feasible_actions={"shot"}, t_sec=40.0,
    ) is None
    assert pending_policy_action_bias(
        state, team_id="B", feasible_actions={"shot"}, t_sec=41.0,
    ) is None
    assert pending_policy_action_bias(
        state, team_id="A", feasible_actions={"pass"}, t_sec=41.0,
    ) is None
    intent = pending_policy_action_bias(
        state, team_id="A", feasible_actions={"pass", "shot"}, t_sec=41.0,
    )
    assert intent == {
        "decision_id": record["decision_id"],
        "action": "shot",
        "logit_bias": 0.5,
    }
    record_policy_intervention_result(
        state,
        decision_id=intent["decision_id"],
        actual_action="pass",
        t_sec=41.0,
    )
    assert record["intervention_applied"] is True
    assert record["adopted"] is False
    assert record["resolution"] == "different_action_after_bounded_bias"
    assert pending_policy_action_bias(
        state, team_id="A", feasible_actions={"shot"}, t_sec=42.0,
    ) is None
    diagnostics = decision_adoption_diagnostics(state)
    assert diagnostics["interventions_applied"] == 1
    assert diagnostics["intervention_adoption_rate"] == 0.0


def test_newer_coach_intent_supersedes_unresolved_same_team_intent():
    state = SimpleNamespace(clock_seconds=10.0)
    older = register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="clock",
        llm_selected_action="hold",
        world_model_recommended_action="hold",
        recommendation_confidence=0.4,
        horizon_s=20.0,
        intervention_enabled=True,
        intervention_strength=0.2,
    )
    state.clock_seconds = 11.0
    newer = register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="goal",
        llm_selected_action="pass",
        world_model_recommended_action="pass",
        recommendation_confidence=0.8,
        horizon_s=20.0,
        intervention_enabled=True,
        intervention_strength=0.3,
    )
    assert older["resolution"] == "superseded_by_newer_coach_decision"
    intent = pending_policy_action_bias(
        state, team_id="A", feasible_actions={"hold", "pass"}, t_sec=12.0,
    )
    assert intent["decision_id"] == newer["decision_id"]
