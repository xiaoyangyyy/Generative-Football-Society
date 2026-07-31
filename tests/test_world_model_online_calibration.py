"""Online calibration uses executed actions and same-tick transition targets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.match_micro_runner import _finish_world_model_tick
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
    assert pass_branch.uncertainty == pytest.approx(
        1.0 - runtime.planner_confidence(observation, kind="pass")
    )


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
                        "short_horizon_outcome": {
                            "policy_utility": float(adopted),
                        },
                        "multi_horizon_outcomes": {
                            "transition": {
                                "policy_utility": float(adopted),
                                "world_model_prediction": {
                                    "policy_utility": float(adopted),
                                    "uncertainty": 0.1,
                                },
                            },
                            "60s": {
                                "policy_utility": float(adopted),
                                "world_model_prediction": {
                                    "policy_utility": float(adopted),
                                    "uncertainty": 0.1,
                                },
                            },
                        },
                    },
                    {
                        "experiment_arm": "control",
                        "experiment_treatment_propensity": 0.8,
                        "short_horizon_outcome": {"policy_utility": 0.0},
                        "multi_horizon_outcomes": {
                            "transition": {
                                "policy_utility": 0.0,
                                "world_model_prediction": {
                                    "policy_utility": 0.0,
                                    "uncertainty": 0.1,
                                },
                            },
                            "60s": {
                                "policy_utility": 0.0,
                                "world_model_prediction": {
                                    "policy_utility": 0.0,
                                    "uncertainty": 0.1,
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

    policy_ready = aggregate_online_calibration(
        [payload(20, 0.02, 0.8, 1), payload(40, 0.01, 1.0, 0)],
        min_transitions=50,
        min_policy_arm=2,
        require_policy_effect=True,
        required_policy_horizon_s=60.0,
        min_residual_samples=2,
        require_outcome_calibration=True,
    )
    assert policy_ready["ready"]
    assert policy_ready["policy_effect_ready"]
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
