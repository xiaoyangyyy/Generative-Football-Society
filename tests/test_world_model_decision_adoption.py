"""Coach action adoption is prospective, bounded, and explicitly non-causal."""

from __future__ import annotations

from types import SimpleNamespace

from src.match_engine.world_model.decision_adoption import (
    decision_adoption_diagnostics,
    finalize_decision_adoption,
    observe_executed_action,
    pending_policy_action_bias,
    policy_bridge_reliability_factor,
    randomized_policy_effect_from_counts,
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
        "experiment_arm": "treatment",
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


def test_randomized_control_is_deterministic_and_applies_zero_bias():
    control_state = None
    control_record = None
    for seed in range(100):
        state = SimpleNamespace(
            clock_seconds=20.0,
            _wm_policy_experiment_seed=seed,
        )
        record = register_coach_action_decision(
            state,
            team_id="A",
            trigger_kind="clock",
            llm_selected_action="pass",
            world_model_recommended_action="pass",
            recommendation_confidence=0.8,
            horizon_s=10.0,
            intervention_enabled=True,
            intervention_strength=0.35,
            experiment_control_rate=0.5,
        )
        if record["experiment_arm"] == "control":
            control_state, control_record = state, record
            break
    assert control_state is not None
    repeated_state = SimpleNamespace(
        clock_seconds=20.0,
        _wm_policy_experiment_seed=control_state._wm_policy_experiment_seed,
    )
    repeated = register_coach_action_decision(
        repeated_state,
        team_id="A",
        trigger_kind="clock",
        llm_selected_action="pass",
        world_model_recommended_action="pass",
        recommendation_confidence=0.8,
        horizon_s=10.0,
        intervention_enabled=True,
        intervention_strength=0.35,
        experiment_control_rate=0.5,
    )
    assert repeated["experiment_arm"] == control_record["experiment_arm"]
    intent = pending_policy_action_bias(
        control_state,
        team_id="A",
        feasible_actions={"pass", "hold"},
        t_sec=21.0,
    )
    assert intent["experiment_arm"] == "control"
    assert intent["logit_bias"] == 0.0
    record_policy_intervention_result(
        control_state,
        decision_id=control_record["decision_id"],
        actual_action="pass",
        t_sec=21.0,
    )
    assert control_record["policy_opportunity_observed"]
    assert not control_record["intervention_applied"]
    assert control_record["resolution"] == "matching_action_in_randomized_control"


def test_randomized_effect_and_reliability_feedback_are_conservative():
    positive = randomized_policy_effect_from_counts(
        treatment=8,
        treatment_adopted=7,
        control=8,
        control_adopted=3,
        min_per_arm=8,
    )
    assert positive["ready"]
    assert positive["average_treatment_effect"] == 0.5
    assert positive["causal_interpretation"] == (
        "within_simulator_action_selection"
    )

    records = []
    for arm, adopted in (("treatment", False), ("control", True)):
        for _ in range(2):
            records.append({
                "policy_opportunity_observed": True,
                "experiment_arm": arm,
                "adopted": adopted,
            })
    state = SimpleNamespace(_wm_coach_decision_adoption=records)
    assert policy_bridge_reliability_factor(state, min_per_arm=2) == 0.25
