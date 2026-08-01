"""Online calibration uses executed actions and same-tick transition targets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.match_micro_runner import (
    _begin_world_model_tick,
    _finish_world_model_tick,
)
from src.match_engine.state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.config import WorldModelConfig
from src.match_engine.world_model.inference import WorldModelRuntime
from src.match_engine.world_model.model import WorldModelOutput
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.online_calibration import (
    OnlineTransitionCalibrator,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)


def test_live_transition_error_reduces_trust_after_minimum_samples():
    calibrator = OnlineTransitionCalibrator(
        expected_weighted_mse=0.01, min_samples=4,
    )
    current = np.zeros(OBS_DIM, dtype=np.float32)
    actual = np.full(OBS_DIM, 0.1, dtype=np.float32)
    poor_prediction = np.ones(OBS_DIM, dtype=np.float32)
    for _ in range(4):
        calibrator.observe(
            action_kind="pass",
            current_observation=current,
            predicted_observation=poor_prediction,
            actual_observation=actual,
            predicted_uncertainty=0.1,
        )
    diagnostics = calibrator.diagnostics()
    assert diagnostics["calibration_target"] == "same_tick_next_observation"
    assert diagnostics["branches"]["pass"]["active"]
    assert diagnostics["branches"]["pass"]["trust_factor"] < 1.0
    assert diagnostics["branches"]["general"]["samples"] == 4
    assert diagnostics["branches"]["shot"]["trust_factor"] == 1.0


def test_accurate_transition_model_keeps_full_online_trust():
    calibrator = OnlineTransitionCalibrator(
        expected_weighted_mse=0.01, min_samples=3,
    )
    current = np.zeros(OBS_DIM, dtype=np.float32)
    actual = np.full(OBS_DIM, 0.1, dtype=np.float32)
    for _ in range(3):
        calibrator.observe(
            action_kind="hold",
            current_observation=current,
            predicted_observation=actual,
            actual_observation=actual,
            predicted_uncertainty=0.05,
        )
    assert calibrator.trust_factor("pass") == pytest.approx(1.0)
    assert calibrator.diagnostics()["branches"]["pass"][
        "skill_vs_persistence"
    ] > 0.0


def test_runtime_confidence_is_only_reduced_and_gate_cannot_reopen():
    cfg = WorldModelConfig(min_planner_quality=0.15)
    model = SimpleNamespace(checkpoint_version=6)
    runtime = WorldModelRuntime(model, cfg, {
        "validation": {
            "planner_quality": 0.8,
            "pass_planner_quality": 0.8,
            "shot_planner_quality": 0.0,
            "weighted_obs_mse": 0.01,
        },
    })
    observation = np.ones(OBS_DIM, dtype=np.float32)
    raw_pass = runtime.planner_confidence(observation, kind="pass")
    current = np.zeros(OBS_DIM, dtype=np.float32)
    actual = np.full(OBS_DIM, 0.1, dtype=np.float32)
    prediction = np.ones(OBS_DIM, dtype=np.float32)
    for _ in range(runtime.online_calibrator.min_samples):
        runtime.online_calibrator.observe(
            action_kind="pass",
            current_observation=current,
            predicted_observation=prediction,
            actual_observation=actual,
            predicted_uncertainty=0.1,
        )
    assert runtime.planner_confidence(observation, kind="pass") < raw_pass
    assert runtime.planner_confidence(observation, kind="shot") == 0.0


def test_pass_imagination_uses_independent_pass_quality_gate():
    class Model:
        checkpoint_version = 6

        def imagine(self, obs, action, *, steps, h):
            return WorldModelOutput(
                next_obs=np.asarray(obs, dtype=np.float32).copy(),
                pass_success=0.5,
                progress_delta=0.0,
                shot_goal_prob=0.1,
                latent=np.zeros(8, dtype=np.float32),
                uncertainty=0.0,
            )

    runtime = WorldModelRuntime(Model(), WorldModelConfig(), {
        "validation": {
            "planner_quality": 0.0,
            "pass_planner_quality": 0.4,
            "shot_planner_quality": 0.0,
            "weighted_obs_mse": 0.01,
        },
    })
    observation = np.ones(OBS_DIM, dtype=np.float32)
    action = encode_high_level_action("pass")
    general = runtime.imagine(observation, action)
    pass_branch = runtime.imagine_pass(observation, action)
    assert general.uncertainty == 1.0
    assert pass_branch.uncertainty < 1.0
    assert pass_branch.epistemic_uncertainty == pytest.approx(
        1.0 - runtime.planner_confidence(observation, kind="pass")
    )
    assert pass_branch.uncertainty >= pass_branch.epistemic_uncertainty


class _ObservingRuntime:
    def __init__(self):
        self.calls = []

    def observe_transition(self, observation, action, next_observation):
        self.calls.append((observation.copy(), action.copy(), next_observation.copy()))
        return {"weighted_mse": 0.1, "trust_factor": 0.8}


def test_tick_calibration_runs_without_trace_recorder():
    state = MatchAffectiveState(
        home=TeamAffectiveState(
            team_id="Home", coach=CoachAffectiveState(team_id="Home"),
        ),
        away=TeamAffectiveState(
            team_id="Away", coach=CoachAffectiveState(team_id="Away"),
        ),
        referee=RefereeAffectiveState(),
        crowd=CrowdState(),
    )
    state.ball.possession_team_id = "Home"
    state._wm_obs_pre = np.zeros(OBS_DIM, dtype=np.float32)
    state._wm_last_action = encode_high_level_action("hold")
    runtime = _ObservingRuntime()
    _finish_world_model_tick(
        state,
        wm_recorder=None,
        wm_runtime=runtime,
        wm_cfg=WorldModelConfig(),
        t1=5.0,
        tick=0,
    )
    assert len(runtime.calls) == 1
    assert state._wm_online_last["trust_factor"] == 0.8


def test_begin_tick_captures_pre_action_baseline_only_for_pending_option():
    state = MatchAffectiveState(
        home=TeamAffectiveState(
            team_id="Home", coach=CoachAffectiveState(team_id="Home"),
        ),
        away=TeamAffectiveState(
            team_id="Away", coach=CoachAffectiveState(team_id="Away"),
        ),
        referee=RefereeAffectiveState(),
        crowd=CrowdState(),
    )
    state.ball.possession_team_id = "Home"
    state.ball.position = [0.62, 0.48]
    state._wm_coach_decision_adoption = [{
        "team_id": "Home",
        "event_option_resolved_t_sec": 60.0,
        "event_option_next_action": None,
        "event_option_followup_expired": False,
    }]

    _begin_world_model_tick(
        state,
        dt=5.0,
        wm_recorder=None,
        wm_runtime=object(),
        wm_cfg=WorldModelConfig(),
    )
    assert state._wm_outcome_baseline_pre["ball_x"] == pytest.approx(0.62)
    assert state._wm_outcome_baseline_pre["goal_diff"] == 0.0

    state._wm_coach_decision_adoption[0]["event_option_followup_expired"] = True
    _begin_world_model_tick(
        state,
        dt=5.0,
        wm_recorder=None,
        wm_runtime=object(),
        wm_cfg=WorldModelConfig(),
    )
    assert state._wm_outcome_baseline_pre is None


def test_online_reports_aggregate_transition_and_adoption_evidence():
    def payload(samples, mse, trust, adopted):
        branch = {
            "samples": samples,
            "mean_weighted_mse": mse,
            "expected_weighted_mse": 0.01,
            "skill_vs_persistence": 0.2,
            "trust_factor": trust,
            "uncertainty_error_correlation": 0.3,
        }
        return {
            "world_model_online_calibration": {
                "calibration_target": "same_tick_next_observation",
                "branches": {
                    "general": branch,
                    "pass": branch,
                    "shot": {"samples": 0},
                },
            },
            "world_model_decision_adoption": {
                "registered": 1,
                "resolved": 1,
                "adopted": adopted,
                "interventions_applied": 1,
                "intervention_adopted": adopted,
                "randomized_policy_effect": {
                    "treatment": 1,
                    "treatment_adopted": adopted,
                    "control": 1,
                    "control_adopted": 0,
                },
                "records": [
                    {
                        "experiment_arm": "treatment",
                        "experiment_treatment_propensity": 0.8,
                        "opponent_belief_context": {
                            "opponent_team_id": "Away",
                            "posterior": {"balanced": 0.7, "low_block": 0.3},
                            "map_hypothesis": "balanced",
                            "normalized_entropy": 0.6,
                            "meta_prior": {"available": True, "matches": 3},
                            "change_point": {
                                "version": 1, "status": "stable",
                                "confirmed": False,
                            },
                        },
                        "short_horizon_outcome": {
                            "policy_utility": float(adopted),
                        },
                        "multi_horizon_outcomes": {
                            "transition": {
                                "policy_utility": float(adopted),
                                "world_model_prediction": {
                                    "policy_utility": float(adopted),
                                    "uncertainty": 0.1,
                                    "epistemic_uncertainty": 0.1,
                                    "aleatoric_uncertainty": 0.0,
                                    "uncertainty_components": {
                                        "transition_ensemble_trained": True,
                                        "transition_ensemble_members": 3,
                                        "transition_epistemic": 0.05,
                                    },
                                },
                            },
                            "60s": {
                                "policy_utility": float(adopted),
                                "world_model_prediction": {
                                    "policy_utility": float(adopted),
                                    "uncertainty": 0.1,
                                    "epistemic_uncertainty": 0.1,
                                    "aleatoric_uncertainty": 0.0,
                                    "uncertainty_components": {
                                        "transition_ensemble_trained": True,
                                        "transition_ensemble_members": 3,
                                        "transition_epistemic": 0.05,
                                    },
                                },
                            },
                        },
                    },
                    {
                        "experiment_arm": "control",
                        "experiment_treatment_propensity": 0.8,
                        "opponent_belief_context": {
                            "opponent_team_id": "Away",
                            "posterior": {"balanced": 0.6, "low_block": 0.4},
                            "map_hypothesis": "balanced",
                            "normalized_entropy": 0.7,
                            "meta_prior": {"available": True, "matches": 3},
                            "change_point": {
                                "version": 1, "status": "stable",
                                "confirmed": False,
                            },
                        },
                        "short_horizon_outcome": {"policy_utility": 0.0},
                        "multi_horizon_outcomes": {
                            "transition": {
                                "policy_utility": 0.0,
                                "world_model_prediction": {
                                    "policy_utility": 0.0,
                                    "uncertainty": 0.1,
                                    "epistemic_uncertainty": 0.1,
                                    "aleatoric_uncertainty": 0.0,
                                    "uncertainty_components": {
                                        "transition_ensemble_trained": True,
                                        "transition_ensemble_members": 3,
                                        "transition_epistemic": 0.05,
                                    },
                                },
                            },
                            "60s": {
                                "policy_utility": 0.0,
                                "world_model_prediction": {
                                    "policy_utility": 0.0,
                                    "uncertainty": 0.1,
                                    "epistemic_uncertainty": 0.1,
                                    "aleatoric_uncertainty": 0.0,
                                    "uncertainty_components": {
                                        "transition_ensemble_trained": True,
                                        "transition_ensemble_members": 3,
                                        "transition_epistemic": 0.05,
                                    },
                                },
                            },
                        },
                    },
                ],
            },
        }

    report = aggregate_online_calibration(
        [payload(20, 0.02, 0.8, 1), payload(40, 0.01, 1.0, 0)],
        min_transitions=50,
    )
    assert report["ready"]
    assert report["branches"]["general"]["samples"] == 60
    assert report["branches"]["general"]["mean_weighted_mse"] == (
        pytest.approx((20 * 0.02 + 40 * 0.01) / 60)
    )
    assert report["decision_adoption"]["adoption_rate"] == 0.5
    assert report["decision_adoption"]["interventions_applied"] == 2
    assert report["decision_adoption"]["intervention_adoption_rate"] == 0.5
    randomized = report["decision_adoption"]["randomized_policy_effect"]
    assert randomized["treatment"] == 2
    assert randomized["control"] == 2
    assert randomized["average_treatment_effect"] == 0.5
    assert not randomized["ready"]
    assert report["decision_adoption"]["causal_interpretation"] is False
    assert report["decision_adoption"]["active_learning"][
        "exploration_decisions"
    ] == 0

    policy_ready = aggregate_online_calibration(
        [payload(20, 0.02, 0.8, 1), payload(40, 0.01, 1.0, 0)],
        min_transitions=50,
        min_policy_arm=2,
        require_policy_effect=True,
        required_policy_horizon_s=60.0,
        min_residual_samples=2,
        require_outcome_calibration=True,
        require_uncertainty_decomposition=True,
        require_transition_ensemble=True,
        require_opponent_belief=True,
        require_opponent_meta_belief=True,
        require_opponent_change_detection=True,
    )
    assert policy_ready["ready"]
    assert policy_ready["policy_effect_ready"]
    assert policy_ready["uncertainty_decomposition_ready"]
    assert policy_ready["transition_ensemble_ready"]
    assert policy_ready["opponent_belief_ready"]
    assert policy_ready["opponent_meta_belief_ready"]
    assert policy_ready["opponent_change_detection_ready"]
    assert policy_ready["gates"][
        "world_model_uncertainty_decomposition"
    ]
    assert policy_ready["gates"]["world_model_transition_ensemble"]
    assert policy_ready["gates"]["opponent_belief_decision_evidence"]
    assert policy_ready["gates"]["opponent_meta_belief_evidence"]
    assert policy_ready["gates"]["opponent_change_detection_audit"]
    outcome = policy_ready["decision_adoption"]["randomized_outcome_effect"]
    assert outcome["ready"]
    assert outcome["cluster_robust"]
    assert outcome["average_treatment_effect"] == pytest.approx(0.5)
    assert policy_ready["decision_adoption"][
        "required_policy_horizon_key"
    ] == "60s"
    assert policy_ready["decision_adoption"][
        "required_policy_outcome"
    ]["average_treatment_effect"] == pytest.approx(0.5)
    assert policy_ready["decision_adoption"][
        "required_policy_outcome"
    ]["estimand"] == "policy_regime_total_effect_with_natural_followup"
    calibration = policy_ready["decision_adoption"][
        "world_model_outcome_calibration"
    ]
    assert calibration["overall"]["active"]
    assert calibration["overall"]["trust_factor"] == 1.0


def test_online_report_isolates_residual_profiles_by_policy_environment():
    records = []
    for environment, residual in (("env-a", 0.1), ("env-b", 0.3)):
        for _ in range(8):
            records.append({
                "checkpoint_signature": "checkpoint-a",
                "environment_signature": environment,
                "policy_utility_version": 1,
                "team_id": "Home",
                "intervention_actual_action": "pass",
                "outcome_baseline": {
                    "opponent_team_id": "Away",
                    "zone": "middle",
                    "score_state": "level",
                    "match_phase": "early",
                },
                "multi_horizon_regime_outcomes": {
                    "60s": {
                        "policy_utility": residual,
                        "world_model_prediction": {
                            "raw_policy_utility": 0.0,
                            "policy_utility": 0.0,
                            "uncertainty": 0.2,
                        },
                    },
                },
            })
    report = aggregate_online_calibration([{
        "home": "Home",
        "away": "Away",
        "world_model_decision_adoption": {"records": records},
    }], min_residual_samples=8)
    profiles = report["decision_adoption"][
        "contextual_residual_memory_by_policy_environment"
    ]

    assert not report["uncertainty_decomposition_ready"]
    assert not report["transition_ensemble_ready"]
    assert not report["opponent_meta_belief_ready"]
    assert not report["opponent_change_detection_ready"]
    assert report["decision_adoption"]["uncertainty_decomposition"][
        "legacy_predictions"
    ] == 16
    assert set(profiles) == {
        "checkpoint-a|env-a", "checkpoint-a|env-b",
    }
    assert profiles["checkpoint-a|env-a"]["residual_rows"] == 8
    assert profiles["checkpoint-a|env-b"]["residual_rows"] == 8


def test_strict_opponent_response_gate_requires_realized_learned_forecasts():
    hypotheses = (
        "balanced", "gegenpress", "possession_control", "counter_attack",
        "low_block", "wing_play", "direct_vertical",
    )

    def posterior(primary):
        return {
            name: 0.94 if name == primary else 0.01
            for name in hypotheses
        }

    records = []
    for index in range(3):
        records.append({
            "team_id": "Home",
            "created_t_sec": float(index * 60),
            "intervention_actual_action": "pass" if index < 2 else None,
            "opponent_belief_context": {
                "opponent_team_id": "Away",
                "posterior": posterior("low_block"),
            },
            "opponent_response_context": {
                "prediction": {
                    "action": "pass",
                    "learned_active": True,
                    "source": "validated_action_conditioned_response_memory",
                    "response_posterior": posterior("low_block"),
                    "structural_posterior": posterior("balanced"),
                },
            },
        })
    report = aggregate_online_calibration(
        [{"world_model_decision_adoption": {"records": records}}],
        min_residual_samples=2,
        require_opponent_response_model=True,
    )

    response = report["decision_adoption"]["opponent_response"]
    assert response["realized_predictions"] == 2
    assert response["validated_learned_predictions"] == 2
    assert report["opponent_response_model_ready"]
    assert report["gates"]["opponent_response_model_evidence"]


def test_strict_two_step_planning_gate_requires_validated_bounded_audits():
    records = [{
        "opponent_response_context": {
            "trajectory_rollout": {
                "active": True,
                "branch_evaluations": 32,
                "branch_evaluation_budget": 48,
                "model_calls": 36,
                "causal_interpretation": False,
                "validation_gate": {
                    "active": True,
                    "authority": 0.35,
                },
            },
        },
    } for _ in range(2)]
    report = aggregate_online_calibration(
        [{"world_model_decision_adoption": {"records": records}}],
        min_residual_samples=2,
        require_two_step_trajectory_planning=True,
    )

    diagnostics = report["decision_adoption"]["trajectory_planning"]
    assert diagnostics["validated_predicted_state_decisions"] == 2
    assert diagnostics["budgets_respected"]
    assert diagnostics["all_active_rollouts_holdout_validated"]
    assert report["two_step_trajectory_planning_ready"]
    assert report["gates"]["validated_two_step_trajectory_planning"]


def test_strict_llm_critic_gate_requires_realized_immutable_corrections():
    records = []
    for _ in range(2):
        records.append({
            "intervention_actual_action": "pass",
            "llm_world_model_critique_context": {
                "version": 1,
                "authority_active": True,
                "correction_applied": True,
                "critique": {
                    "action": "pass",
                    "horizon": "60s",
                },
                "applied_ranking_correction": 0.02,
                "world_model_prediction_mutated": False,
                "can_update_world_model": False,
                "can_update_critic_memory": False,
            },
            "multi_horizon_regime_outcomes": {
                "60s": {
                    "policy_utility": 0.08,
                    "world_model_prediction": {
                        "policy_utility": 0.0,
                        "raw_policy_utility": 0.0,
                    },
                },
            },
        })
    report = aggregate_online_calibration(
        [{"world_model_decision_adoption": {"records": records}}],
        min_residual_samples=2,
        require_llm_semantic_critic=True,
    )

    diagnostics = report["decision_adoption"]["llm_semantic_critic"]
    assert diagnostics["realized_authority_active_critiques"] == 2
    assert diagnostics["realized_validated_corrections"] == 2
    assert diagnostics["match_clustered_corrected_mse"] < (
        diagnostics["match_clustered_baseline_mse"]
    )
    assert diagnostics["all_non_persistent"]
    assert diagnostics["all_predictions_immutable"]
    assert diagnostics["all_bridge_authority_non_increasing"]
    assert report["llm_semantic_critic_ready"]
    assert report["gates"]["validated_llm_semantic_critic"]


def test_strict_semantic_event_gate_requires_paired_shadow_match_evidence():
    logs = []
    for match_index in range(4):
        observed = bool(match_index % 2)
        target = float(observed)
        llm_probability = 0.75 if observed else 0.25
        model_probability = 0.65 if observed else 0.35
        projection_probability = 0.55 if observed else 0.45
        learned_probability = 0.75 if observed else 0.25
        evaluation = {
            "version": 2,
            "event": "retain_possession",
            "observed": observed,
            "llm_probability": llm_probability,
            "world_model_probability": model_probability,
            "llm_brier": (llm_probability - target) ** 2,
            "world_model_brier": (model_probability - target) ** 2,
            "llm_brier_gain_vs_world_model": (
                (model_probability - target) ** 2
                - (llm_probability - target) ** 2
            ),
            "shadow_only": True,
            "causal_interpretation": False,
            "event_signature": "llm-event:test-contract",
            "checkpoint_signature": "checkpoint-a",
            "environment_signature": "environment-a",
            "world_model_event_components": {
                "projection_probability": projection_probability,
                "projection_brier": (
                    projection_probability - target
                ) ** 2,
                "learned_probability": learned_probability,
                "learned_brier": (learned_probability - target) ** 2,
                "fused_probability": model_probability,
                "fused_brier": (model_probability - target) ** 2,
                "gate": {
                    "active": True,
                    "authority": 0.4,
                    "reason": "grouped_event_head_gain",
                },
            },
        }
        logs.append({
            "world_model_decision_adoption": {
                "records": [{
                    "multi_horizon_regime_outcomes": {
                        "60s": {
                            "llm_semantic_event_evaluation": evaluation,
                        },
                    },
                }],
            },
        })
    report = aggregate_online_calibration(
        logs,
        min_residual_samples=4,
        require_llm_semantic_events=True,
        require_learned_semantic_events=True,
    )

    diagnostics = report["decision_adoption"]["llm_semantic_events"]
    assert diagnostics["realized_predictions"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["match_clustered_llm_gain_vs_world_model"] > 0.0
    assert diagnostics["all_paired_same_outcome"]
    assert diagnostics["all_shadow_only"]
    assert not diagnostics["authority_active"]
    assert report["llm_semantic_events_ready"]
    assert report["gates"]["paired_llm_semantic_event_evaluation"]
    assert diagnostics["realized_learned_head_predictions"] == 4
    assert diagnostics["learned_head_matches"] == 4
    assert diagnostics["match_clustered_learned_head_brier"] < (
        diagnostics["match_clustered_projection_brier"]
    )
    assert diagnostics["match_clustered_fused_event_brier"] < (
        diagnostics["match_clustered_projection_brier"]
    )
    assert diagnostics["all_learned_authority_bounded"]
    assert report["learned_semantic_events_ready"]
    assert report["gates"]["validated_learned_semantic_event_fusion"]

    last_evaluation = logs[-1]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["60s"]["llm_semantic_event_evaluation"]
    last_evaluation["event_signature"] = "llm-event:different-contract"
    mixed = aggregate_online_calibration(
        logs,
        min_residual_samples=4,
        require_llm_semantic_events=True,
    )
    assert not mixed["decision_adoption"]["llm_semantic_events"][
        "provenance_compatible"
    ]
    assert not mixed["llm_semantic_events_ready"]
    last_evaluation["event_signature"] = "llm-event:test-contract"

    logs.append({
        "world_model_decision_adoption": {
            "records": [{
                "multi_horizon_regime_outcomes": {
                    "60s": {
                        "llm_semantic_event_evaluation": {
                            "event": "retain_possession",
                            "llm_probability": float("nan"),
                            "world_model_probability": 0.5,
                            "llm_brier": 0.25,
                            "world_model_brier": 0.25,
                            "shadow_only": True,
                            "causal_interpretation": False,
                        },
                    },
                },
            }],
        },
    })
    contaminated = aggregate_online_calibration(
        logs,
        min_residual_samples=4,
        require_llm_semantic_events=True,
    )
    assert contaminated["decision_adoption"]["llm_semantic_events"][
        "malformed_evaluations"
    ] == 1
    assert not contaminated["llm_semantic_events_ready"]


def test_strict_event_option_gate_requires_safe_cross_match_followups():
    logs = []
    for index in range(4):
        observed_utility = 0.10 if index % 2 else -0.10
        predicted_utility = observed_utility - 0.01
        evaluation = {
            "version": 2,
            "event_observed": bool(index % 2),
            "expected_continuation_action": "shot",
            "next_action_observed": True,
            "observed_continuation_action": "shot",
            "expected_action_matched": True,
            "conditional_gain_vs_best_fixed": 0.02 + 0.01 * index,
            "member_evaluations": 8,
            "member_evaluation_budget": 16,
            "trajectory_member_paths": 27,
            "trajectory_member_path_budget": 144,
            "shadow_only": True,
            "authority_active": False,
            "policy_mutated": False,
            "can_execute_future_action": False,
            "causal_interpretation": False,
            "continuation_value_calibratable": True,
            "continuation_calibration_eligible": True,
            "continuation_calibration_observed": True,
            "expected_continuation_policy_utility": predicted_utility,
            "realized_continuation_outcome": {
                "policy_utility": observed_utility,
            },
            "continuation_policy_utility_residual": 0.01,
            "continuation_policy_utility_squared_error": 0.0001,
            "option_signature": "llm-event-option:test-contract",
            "checkpoint_signature": "checkpoint-a",
            "environment_signature": "environment-a",
        }
        logs.append({
            "world_model_decision_adoption": {
                "records": [{
                    "multi_horizon_regime_outcomes": {
                        "60s": {"llm_event_option_evaluation": evaluation},
                    },
                }],
            },
        })

    report = aggregate_online_calibration(
        logs,
        min_residual_samples=2,
        require_llm_event_options=True,
        require_llm_event_option_values=True,
    )
    diagnostics = report["decision_adoption"]["llm_event_options"]
    assert diagnostics["resolved_events"] == 4
    assert diagnostics["continuations_observed"] == 4
    assert diagnostics["continuation_match_rate"] == pytest.approx(1.0)
    assert diagnostics["budgets_respected"]
    assert diagnostics["provenance_compatible"]
    assert report["llm_event_options_ready"]
    assert report["gates"]["shadow_llm_event_option_evaluation"]
    assert diagnostics["value_predictions_realized"] == 4
    assert diagnostics["value_calibration_matches"] == 4
    assert diagnostics["match_clustered_value_skill_vs_zero"] > 0.0
    assert report["llm_event_option_values_ready"]
    assert report["gates"]["calibrated_llm_event_option_values"]

    evaluation["option_signature"] = "llm-event-option:mixed-contract"
    mixed = aggregate_online_calibration(
        logs,
        min_residual_samples=2,
        require_llm_event_options=True,
    )
    assert not mixed["decision_adoption"]["llm_event_options"][
        "provenance_compatible"
    ]
    assert not mixed["llm_event_options_ready"]

    evaluation["option_signature"] = "llm-event-option:test-contract"
    evaluation["expected_action_matched"] = False
    contaminated_value = aggregate_online_calibration(
        logs,
        min_residual_samples=2,
        require_llm_event_option_values=True,
    )
    assert contaminated_value["decision_adoption"]["llm_event_options"][
        "malformed_value_evaluations"
    ] == 1
    assert not contaminated_value["llm_event_option_values_ready"]

    evaluation["expected_action_matched"] = True
    evaluation["conditional_gain_vs_best_fixed"] = float("nan")
    malformed = aggregate_online_calibration(
        logs,
        min_residual_samples=2,
        require_llm_event_options=True,
    )
    assert malformed["decision_adoption"]["llm_event_options"][
        "malformed_evaluations"
    ] == 1
    assert not malformed["llm_event_options_ready"]
