"""Coach action adoption is prospective, bounded, and explicitly non-causal."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.match_engine.world_model.decision_adoption import (
    decision_adoption_diagnostics,
    finalize_decision_adoption,
    observe_executed_action,
    pending_policy_action_bias,
    record_policy_intervention_result,
    register_coach_action_decision,
)
from src.match_engine.world_model.policy_outcomes import (
    capture_policy_outcome_baseline,
    observe_policy_intervention_outcomes,
)
from src.match_engine.world_model.policy_experiment import (
    policy_bridge_reliability_factor,
    randomized_adoption_effect_from_counts,
    randomized_multi_horizon_effects,
    randomized_outcome_effect,
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
    positive = randomized_adoption_effect_from_counts(
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


def test_policy_outcome_tracks_progress_retention_xg_and_score_delta():
    home = SimpleNamespace(team_id="A", score=0)
    away = SimpleNamespace(team_id="B", score=0)
    state = SimpleNamespace(
        clock_seconds=10.0,
        home=home,
        away=away,
        ball=SimpleNamespace(
            position=[0.40, 0.50], possession_team_id="A",
        ),
        micro_xg_home=0.2,
        micro_xg_away=0.1,
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
        intervention_strength=0.3,
        action_predictions={
            "pass": {
                "transition": {
                    "policy_utility": 0.10,
                    "uncertainty": 0.20,
                },
            },
        },
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state,
        decision_id=record["decision_id"],
        actual_action="pass",
        t_sec=11.0,
        outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.60
    state.ball.possession_team_id = "A"
    state.micro_xg_home = 0.3
    observe_policy_intervention_outcomes(state, t_sec=11.0)
    outcome = record["short_horizon_outcome"]
    assert outcome["progress"] == pytest.approx(0.2)
    assert outcome["retained_possession"]
    assert outcome["xg_net_delta"] == pytest.approx(0.1)
    assert outcome["policy_utility"] == pytest.approx(0.115)
    assert outcome["prediction_residual"] == pytest.approx(0.015)
    assert outcome["standardized_prediction_residual"] == pytest.approx(0.075)


def test_policy_outcome_effect_uses_propensity_weights_and_match_clusters():
    def row(arm, value, propensity):
        return {
            "experiment_arm": arm,
            "experiment_treatment_propensity": propensity,
            "short_horizon_outcome": {"policy_utility": value},
        }

    effect = randomized_outcome_effect(
        [
            [row("treatment", 1.0, 0.8), row("control", 0.0, 0.8)],
            [row("treatment", 1.0, 0.6), row("control", 0.0, 0.6)],
        ],
        min_per_arm=2,
    )
    assert effect["ready"]
    assert effect["cluster_robust"]
    assert effect["clusters"] == 2
    assert effect["average_treatment_effect"] == pytest.approx(1.0)
    assert effect["causal_interpretation"] == (
        "within_simulator_short_horizon_outcome"
    )
    segregated = randomized_outcome_effect(
        [
            [row("treatment", 1.0, 0.8), row("treatment", 0.8, 0.8)],
            [row("control", 0.0, 0.8), row("control", 0.2, 0.8)],
        ],
        min_per_arm=2,
    )
    assert segregated["ready"]
    assert not segregated["cluster_robust"]


def test_outcome_evidence_overrides_positive_adoption_feedback():
    records = []
    for arm, adopted, utility in (
        ("treatment", True, 0.0),
        ("control", False, 1.0),
    ):
        for _ in range(2):
            records.append({
                "policy_opportunity_observed": True,
                "experiment_arm": arm,
                "experiment_treatment_propensity": 0.5,
                "adopted": adopted,
                "short_horizon_outcome": {"policy_utility": utility},
            })
    state = SimpleNamespace(_wm_coach_decision_adoption=records)
    assert policy_bridge_reliability_factor(state, min_per_arm=2) == 0.25


def test_multi_horizon_outcomes_are_recorded_and_future_overlap_is_censored():
    home = SimpleNamespace(team_id="A", score=0)
    away = SimpleNamespace(team_id="B", score=0)
    state = SimpleNamespace(
        clock_seconds=10.0,
        home=home,
        away=away,
        ball=SimpleNamespace(
            position=[0.40, 0.50], possession_team_id="A",
        ),
        micro_xg_home=0.2,
        micro_xg_away=0.1,
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
        intervention_strength=0.3,
        outcome_horizons_s=(0.0, 60.0, 180.0),
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state,
        decision_id=record["decision_id"],
        actual_action="pass",
        t_sec=11.0,
        outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.50
    observe_policy_intervention_outcomes(state, t_sec=11.0)
    assert set(record["multi_horizon_outcomes"]) == {"transition"}

    state.ball.position[0] = 0.65
    state.micro_xg_home = 0.35
    observe_policy_intervention_outcomes(state, t_sec=71.0)
    assert set(record["multi_horizon_outcomes"]) == {"transition", "60s"}

    state.clock_seconds = 100.0
    register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="goal",
        llm_selected_action="hold",
        world_model_recommended_action="hold",
        recommendation_confidence=0.7,
        horizon_s=10.0,
        intervention_enabled=True,
        intervention_strength=0.2,
    )
    state.ball.position[0] = 0.90
    observe_policy_intervention_outcomes(state, t_sec=191.0)
    assert record["outcome_censored_t_sec"] == 100.0
    assert record["outcome_censor_reason"] == (
        "subsequent_same_team_coach_decision"
    )
    assert "180s" not in record["multi_horizon_outcomes"]
    assert "180s" in record["multi_horizon_regime_outcomes"]


def test_semantic_event_hypothesis_is_scored_at_its_declared_horizon():
    home = SimpleNamespace(team_id="A", score=0)
    away = SimpleNamespace(team_id="B", score=0)
    state = SimpleNamespace(
        clock_seconds=10.0,
        home=home,
        away=away,
        ball=SimpleNamespace(
            position=[0.50, 0.50], possession_team_id="A",
        ),
        micro_xg_home=0.2,
        micro_xg_away=0.1,
    )
    event_context = {
        "accepted": True,
        "hypothesis": {
            "action": "pass",
            "horizon": "60s",
            "event": "enter_final_third",
        },
        "llm_event_probability": 0.8,
        "world_model_event_probability": 0.6,
    }
    record = register_coach_action_decision(
        state,
        team_id="A",
        trigger_kind="clock",
        llm_selected_action="pass",
        world_model_recommended_action="pass",
        recommendation_confidence=0.8,
        horizon_s=10.0,
        intervention_enabled=True,
        intervention_strength=0.3,
        outcome_horizons_s=(0.0, 60.0),
        llm_semantic_event_context=event_context,
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state,
        decision_id=record["decision_id"],
        actual_action="pass",
        t_sec=11.0,
        outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.55
    observe_policy_intervention_outcomes(state, t_sec=11.0)
    assert "llm_semantic_event_evaluation" not in record[
        "multi_horizon_regime_outcomes"
    ]["transition"]

    state.ball.position[0] = 0.75
    observe_policy_intervention_outcomes(state, t_sec=71.0)
    score = record["multi_horizon_regime_outcomes"]["60s"][
        "llm_semantic_event_evaluation"
    ]
    assert score["observed"]
    assert score["llm_brier"] == pytest.approx(0.04)
    assert score["world_model_brier"] == pytest.approx(0.16)
    assert score["shadow_only"]


def test_semantic_event_hypothesis_is_not_scored_for_a_different_action():
    home = SimpleNamespace(team_id="A", score=0)
    away = SimpleNamespace(team_id="B", score=0)
    state = SimpleNamespace(
        clock_seconds=10.0,
        home=home,
        away=away,
        ball=SimpleNamespace(
            position=[0.50, 0.50], possession_team_id="A",
        ),
        micro_xg_home=0.2,
        micro_xg_away=0.1,
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
        intervention_strength=0.3,
        outcome_horizons_s=(60.0,),
        llm_semantic_event_context={
            "accepted": True,
            "hypothesis": {
                "action": "pass",
                "horizon": "60s",
                "event": "enter_final_third",
            },
            "llm_event_probability": 0.8,
            "world_model_event_probability": 0.6,
        },
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state,
        decision_id=record["decision_id"],
        actual_action="hold",
        t_sec=11.0,
        outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.75
    observe_policy_intervention_outcomes(state, t_sec=71.0)

    assert "llm_semantic_event_evaluation" not in record[
        "multi_horizon_regime_outcomes"
    ]["60s"]


def test_longest_ready_outcome_horizon_drives_reliability_feedback():
    records = []
    for arm in ("treatment", "control"):
        for _ in range(2):
            transition = 1.0 if arm == "treatment" else 0.0
            longer = 0.0 if arm == "treatment" else 1.0
            records.append({
                "policy_opportunity_observed": True,
                "experiment_arm": arm,
                "experiment_treatment_propensity": 0.5,
                "adopted": arm == "treatment",
                "short_horizon_outcome": {"policy_utility": transition},
                "multi_horizon_outcomes": {
                    "transition": {"policy_utility": transition},
                    "60s": {"policy_utility": longer},
                },
            })
    effects = randomized_multi_horizon_effects(
        [records], min_per_arm=2,
    )
    assert effects["transition"]["average_treatment_effect"] == 1.0
    assert effects["60s"]["average_treatment_effect"] == -1.0
    state = SimpleNamespace(_wm_coach_decision_adoption=records)
    assert policy_bridge_reliability_factor(state, min_per_arm=2) == 0.25
