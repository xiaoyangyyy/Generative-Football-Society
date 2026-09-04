from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.match_engine.world_model.inference import WorldModelRuntime
from src.match_engine.world_model.opponent_contract import OPPONENT_HYPOTHESES
from src.match_engine.world_model.opponent_game import (
    apply_llm_response_hypothesis,
    attach_second_order_game,
)
from src.match_engine.world_model.trajectory_game import (
    build_predicted_state_continuations,
)


def _posterior(primary="balanced"):
    return {
        name: 0.94 if name == primary else 0.01
        for name in OPPONENT_HYPOTHESES
    }


def _candidates():
    candidates = []
    for index, action in enumerate(("hold", "pass", "cross", "shot")):
        candidates.append({
            "action": action,
            "target": [0.55 + 0.05 * index, 0.5],
            "physics_prior": 0.7,
            "effective_confidence": 0.8,
            "risk_adjusted_value": 0.2 + 0.01 * index,
            "opponent_hypothesis_values": {
                name: 0.2 + 0.01 * index for name in OPPONENT_HYPOTHESES
            },
        })
    attach_second_order_game(
        candidates, {"posterior": _posterior()}, None,
    )
    return candidates


class _ValidatedRuntime:
    def __init__(self):
        self.imagine_calls = 0
        self.evaluate_calls = 0

    def two_step_planning_gate(self):
        return {
            "active": True,
            "authority": 0.4,
            "reason": "grouped_two_step_holdout_gain",
            "samples": 80,
            "groups": 10,
        }

    def imagine(self, observation, action, **kwargs):
        self.imagine_calls += 1
        following = np.asarray(observation, dtype=np.float32).copy()
        following[200] = np.clip(following[200] + 0.12, 0.0, 1.0)
        return SimpleNamespace(next_obs=following, uncertainty=0.1)

    def evaluate_action_from_observation(
        self, observation, action, *, action_kind, attacking_home, horizon_s,
    ):
        self.evaluate_calls += 1
        action_bonus = {
            "hold": 0.0, "pass": 0.1, "cross": 0.2, "shot": 0.5,
        }[action_kind]
        # Confirms that continuation evaluation sees the predicted state.
        expected = action_bonus + float(observation[200])
        future = SimpleNamespace(
            state_uncertainty=0.1,
            event_probabilities={"turnover": 0.1},
        )
        return {
            "expected_value": expected,
            "future": future,
            "model_calls": 1,
        }


def test_validated_search_reencodes_and_scores_from_predicted_state_with_budget():
    runtime = _ValidatedRuntime()
    observation = np.zeros(307, dtype=np.float32)
    observation[200:202] = [0.50, 0.50]
    candidates = _candidates()
    matrices, audit = build_predicted_state_continuations(
        runtime,
        observation,
        candidates,
        attacking_home=True,
        horizon_s=10.0,
        uncertainty_penalty=0.25,
        max_branch_evaluations=32,
    )

    assert audit["active"]
    assert audit["branch_evaluations"] == 32
    assert audit["branch_evaluations"] <= audit["branch_evaluation_budget"]
    assert audit["first_state_rollouts"] == 4
    assert runtime.imagine_calls == 4
    assert runtime.evaluate_calls == 32
    assert set(matrices) == {"hold", "pass", "cross", "shot"}
    assert matrices["hold"]["shot"]["balanced"] > (
        candidates[3]["opponent_hypothesis_values"]["balanced"]
    )


def test_answer_conditioned_route_preserves_broad_coverage_and_focuses_budget():
    runtime = _ValidatedRuntime()
    observation = np.zeros(307, dtype=np.float32)
    observation[200:202] = [0.50, 0.50]
    route = {
        "target_first_actions": ["shot", "pass"],
        "target_continuation_actions": ["shot", "pass"],
        "hypothesis_priority": list(reversed(OPPONENT_HYPOTHESES)),
        "route_digest": "belief-space-compute-route:test",
    }
    matrices, audit = build_predicted_state_continuations(
        runtime,
        observation,
        _candidates(),
        attacking_home=True,
        horizon_s=10.0,
        uncertainty_penalty=0.25,
        max_branch_evaluations=32,
        belief_space_route=route,
    )

    assert audit["active"]
    assert audit["belief_space_route_applied"]
    assert audit["branch_evaluations"] == 32
    assert audit["broad_coverage_pairs"] == 16
    assert audit["minimum_evaluated_hypotheses_per_branch"] == 1
    assert audit["maximum_evaluated_hypotheses_per_branch"] > 1
    assert audit["routed_branch_evaluations"] > 16
    assert set(matrices) == {"hold", "pass", "cross", "shot"}


def test_closed_two_step_gate_performs_no_rollout_and_keeps_proxy():
    class Closed(_ValidatedRuntime):
        def two_step_planning_gate(self):
            return {"active": False, "authority": 0.0, "reason": "no_gain"}

    runtime = Closed()
    matrices, audit = build_predicted_state_continuations(
        runtime,
        np.zeros(307, dtype=np.float32),
        _candidates(),
        attacking_home=True,
        horizon_s=10.0,
        uncertainty_penalty=0.25,
    )

    assert matrices == {}
    assert not audit["active"]
    assert audit["reason"] == "no_gain"
    assert runtime.imagine_calls == 0
    assert runtime.evaluate_calls == 0


def test_partial_rollout_failures_cannot_receive_planning_authority():
    class Failing(_ValidatedRuntime):
        def evaluate_action_from_observation(self, *args, **kwargs):
            self.evaluate_calls += 1
            raise RuntimeError("synthetic rollout failure")

    runtime = Failing()
    candidates = _candidates()
    matrices, audit = build_predicted_state_continuations(
        runtime,
        np.zeros(307, dtype=np.float32),
        candidates,
        attacking_home=True,
        horizon_s=10.0,
        uncertainty_penalty=0.25,
        max_branch_evaluations=32,
    )

    assert matrices == {}
    assert not audit["active"]
    assert audit["successful_branch_evaluations"] == 0
    assert audit["failed_branch_evaluations"] == 32
    assert all(
        candidate["trajectory_planning_information"] == 0.0
        for candidate in candidates
    )


def test_predicted_state_matrix_upgrades_game_but_keeps_one_belief_policy():
    runtime = _ValidatedRuntime()
    observation = np.zeros(307, dtype=np.float32)
    observation[200] = 0.5
    candidates = _candidates()
    matrices, audit = build_predicted_state_continuations(
        runtime,
        observation,
        candidates,
        attacking_home=True,
        horizon_s=10.0,
        uncertainty_penalty=0.25,
        max_branch_evaluations=48,
    )
    game = attach_second_order_game(
        candidates,
        {"posterior": _posterior()},
        None,
        continuation_value_matrices=matrices,
        trajectory_audit=audit,
    )

    assert game["scope"] == "validated_predicted_state_two_ply_belief_policy"
    assert game["trajectory_rollout"]["active"]
    assert game["recommended_action"] in {"hold", "pass", "cross", "shot"}
    assert all(
        option["evaluation_source"]
        == "validated_predicted_state_rollout_blend"
        for candidate in candidates
        for option in candidate["continuation_options"]
    )
    assert "full response belief" in audit["partial_observability_policy"]


def test_runtime_gate_requires_grouped_holdout_gain_and_caps_authority():
    runtime = SimpleNamespace(
        meta={"validation": {"two_step_rollout": {
            "samples": 200,
            "groups": 20,
            "weighted_mse": 0.01,
            "persistence_weighted_mse": 0.02,
            "skill_vs_persistence": 0.50,
            "action_sequence": "observed_changing_actions",
            "autoregressive_predicted_state": True,
            "trained_with_autoregressive_multi_step_objective": True,
            "training_pairs": 500,
            "training_groups": 30,
            "max_curriculum_weight_applied": 0.25,
            "optimization_steps": 80,
            "member_mean_weighted_mse": 0.012,
            "ensemble_gain_vs_member_mean": 0.002,
            "disagreement_error_correlation": 0.25,
        }}},
        model=SimpleNamespace(transition_ensemble_trained=True),
    )
    gate = WorldModelRuntime.two_step_planning_gate(runtime)

    assert gate["active"]
    assert 0.0 < gate["authority"] <= 0.50
    assert gate["reason"] == "trained_calibrated_grouped_two_step_gain"

    runtime.meta["validation"]["two_step_rollout"]["optimization_steps"] = 0
    missing_training = WorldModelRuntime.two_step_planning_gate(runtime)
    assert not missing_training["active"]
    assert missing_training["reason"] == "two_step_training_contract_missing"
    runtime.meta["validation"]["two_step_rollout"]["optimization_steps"] = 80
    runtime.meta["validation"]["two_step_rollout"]["skill_vs_persistence"] = -0.1
    assert not WorldModelRuntime.two_step_planning_gate(runtime)["active"]


def test_runtime_gate_rejects_validation_only_checkpoint_without_training_contract():
    runtime = SimpleNamespace(
        meta={"validation": {"two_step_rollout": {
            "samples": 200,
            "groups": 20,
            "weighted_mse": 0.01,
            "persistence_weighted_mse": 0.02,
            "skill_vs_persistence": 0.50,
            "action_sequence": "observed_changing_actions",
        }}},
        model=SimpleNamespace(transition_ensemble_trained=True),
    )

    gate = WorldModelRuntime.two_step_planning_gate(runtime)

    assert not gate["active"]
    assert gate["reason"] == "two_step_training_contract_missing"


def test_llm_branch_stress_preserves_validated_trajectory_matrix():
    runtime = _ValidatedRuntime()
    observation = np.zeros(307, dtype=np.float32)
    observation[200] = 0.5
    candidates = _candidates()
    matrices, trajectory_audit = build_predicted_state_continuations(
        runtime,
        observation,
        candidates,
        attacking_home=True,
        horizon_s=10.0,
        uncertainty_penalty=0.25,
    )
    packet = {
        "candidates": candidates,
        "opponent_belief": {"posterior": _posterior()},
    }
    packet["second_order_game"] = attach_second_order_game(
        candidates,
        packet["opponent_belief"],
        None,
        continuation_value_matrices=matrices,
        trajectory_audit=trajectory_audit,
    )
    audit = apply_llm_response_hypothesis(
        packet,
        {
            "if_action": "hold",
            "response_preset": "balanced",
            "confidence": 1.0,
        },
        None,
    )

    assert audit["accepted"]
    assert packet["second_order_game"]["trajectory_rollout"]["active"]
    assert packet["second_order_game"]["scope"] == (
        "validated_predicted_state_two_ply_belief_policy"
    )
