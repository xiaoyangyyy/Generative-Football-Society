"""Budgeted predicted-state continuation search for opponent-aware planning."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.opponent_contract import OPPONENT_HYPOTHESES


TRAJECTORY_GAME_VERSION = 1


def _gate(runtime) -> dict[str, Any]:
    provider = getattr(runtime, "two_step_planning_gate", None)
    if not callable(provider):
        return {
            "version": TRAJECTORY_GAME_VERSION,
            "active": False,
            "authority": 0.0,
            "reason": "runtime_has_no_two_step_validation_contract",
        }
    try:
        gate = dict(provider())
    except (RuntimeError, TypeError, ValueError):
        return {
            "version": TRAJECTORY_GAME_VERSION,
            "active": False,
            "authority": 0.0,
            "reason": "two_step_validation_contract_error",
        }
    gate["authority"] = float(np.clip(
        finite_float(gate.get("authority") or 0.0, 0.0), 0.0, 0.50,
    ))
    gate["active"] = bool(gate.get("active") and gate["authority"] > 0.0)
    return gate


def _target(observation: np.ndarray, action: str, attacking_home: bool) -> np.ndarray:
    ball = np.asarray(observation[200:202], dtype=np.float32).copy()
    direction = 1.0 if attacking_home else -1.0
    if action == "hold":
        return np.clip(ball, 0.02, 0.98)
    if action == "pass":
        return np.clip(ball + np.array([0.14 * direction, 0.0]), 0.02, 0.98)
    if action == "cross":
        return np.asarray(
            [0.91 if attacking_home else 0.09, 0.50], dtype=np.float32,
        )
    return np.asarray(
        [0.995 if attacking_home else 0.005, 0.50], dtype=np.float32,
    )


def _physics_prior(
    observation: np.ndarray, action: str, attacking_home: bool,
) -> float:
    if action == "hold":
        return 0.90
    if action == "pass":
        return 0.72
    if action == "cross":
        return 0.46
    goal_x = 0.995 if attacking_home else 0.005
    distance = abs(goal_x - float(observation[200]))
    return float(np.clip(0.35 * (1.0 - distance), 0.03, 0.45))


def _condition_opponent(
    observation: np.ndarray,
    *,
    team_is_home: bool,
    hypothesis: str,
) -> np.ndarray:
    from src.match_engine.world_model.opponent_contract import (
        tactic_feature_vector,
    )

    conditioned = np.asarray(observation, dtype=np.float32).copy()
    start = 302 if team_is_home else 298
    conditioned[start:start + 4] = tactic_feature_vector(hypothesis)
    return conditioned


def evaluate_predicted_state_action(
    runtime,
    observation: np.ndarray,
    *,
    action_name: str,
    attacking_home: bool,
    horizon_s: float,
) -> dict[str, Any]:
    target = _target(observation, action_name, attacking_home)
    action = encode_high_level_action(
        action_name, target=target, horizon_s=horizon_s,
    )
    action[13] = _physics_prior(observation, action_name, attacking_home)
    evaluator = getattr(runtime, "evaluate_action_from_observation", None)
    if callable(evaluator):
        result = evaluator(
            observation,
            action,
            action_kind=action_name,
            attacking_home=attacking_home,
            horizon_s=horizon_s,
        )
        future = result["future"]
        return {
            "expected_value": finite_float(
                result.get("expected_value") or 0.0, 0.0,
            ),
            "uncertainty": float(np.clip(
                future.state_uncertainty, 0.0, 1.0,
            )),
            "turnover": float(np.clip(
                future.event_probabilities.get("turnover", 1.0), 0.0, 1.0,
            )),
            "model_calls": int(result.get("model_calls", 1)),
        }
    future = runtime.predict_future(
        observation, action, action_kind=action_name, horizon_s=horizon_s,
    )
    expected = (
        runtime.score_shot_action(
            observation, action, attacking_home=attacking_home,
        )
        if action_name == "shot"
        else runtime.score_action(observation, action)
    )
    return {
        "expected_value": finite_float(expected, 0.0),
        "uncertainty": float(np.clip(future.state_uncertainty, 0.0, 1.0)),
        "turnover": float(np.clip(
            future.event_probabilities.get("turnover", 1.0), 0.0, 1.0,
        )),
        "model_calls": 2,
    }


def build_predicted_state_continuations(
    runtime,
    observation: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    attacking_home: bool,
    horizon_s: float,
    uncertainty_penalty: float,
    max_branch_evaluations: int = 48,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    """Return first-action/continuation/hypothesis values and an audit."""
    validation_gate = _gate(runtime)
    budget = max(0, min(112, int(max_branch_evaluations)))
    base_audit = {
        "version": TRAJECTORY_GAME_VERSION,
        "validation_gate": validation_gate,
        "branch_evaluation_budget": budget,
        "branch_evaluations": 0,
        "model_calls": 0,
        "evaluated_hypotheses_per_branch": 0,
        "fallback_values": 0,
        "causal_interpretation": False,
    }
    if runtime is None or not validation_gate["active"]:
        return {}, {
            **base_audit,
            "active": False,
            "reason": validation_gate.get("reason", "validation_gate_closed"),
        }
    eligible = [
        candidate for candidate in candidates
        if float(candidate.get("effective_confidence", 0.0)) > 0.0
        and candidate.get("opponent_hypothesis_values")
        and candidate.get("opponent_response_prediction")
    ]
    if not eligible or budget < len(eligible) * len(eligible):
        return {}, {
            **base_audit,
            "active": False,
            "reason": "insufficient_compute_budget",
        }
    hypothesis_count = min(
        len(OPPONENT_HYPOTHESES),
        max(1, budget // (len(eligible) * len(eligible))),
    )
    matrices: dict[str, dict[str, dict[str, float]]] = {}
    evaluations = successful_evaluations = successful_pairs = 0
    model_calls = fallback_values = 0
    first_rollout_calls = 0
    for first in eligible:
        first_action = str(first["action"])
        first_vector = encode_high_level_action(
            first_action,
            target=np.asarray(first["target"], dtype=np.float32),
            horizon_s=horizon_s,
        )
        first_vector[13] = float(first["physics_prior"])
        first_rollout_calls += 1
        model_calls += 1
        try:
            first_output = runtime.imagine(
                observation,
                first_vector,
                carry_hidden=False,
                quality_kind=("shot" if first_action == "shot" else "pass"),
            )
        except (RuntimeError, TypeError, ValueError, AttributeError):
            continue
        predicted_state = np.clip(np.nan_to_num(
            np.asarray(first_output.next_obs, dtype=np.float32),
            nan=0.0, posinf=1.0, neginf=0.0,
        ), 0.0, 1.0)
        raw_first_uncertainty = getattr(first_output, "uncertainty", None)
        first_uncertainty = float(np.clip(finite_float(
            1.0 if raw_first_uncertainty is None else raw_first_uncertainty,
            1.0,
        ), 0.0, 1.0))
        response = first["opponent_response_prediction"]["response_posterior"]
        hypotheses = sorted(
            OPPONENT_HYPOTHESES,
            key=lambda name: float(response.get(name, 0.0)),
            reverse=True,
        )[:hypothesis_count]
        branch_matrix: dict[str, dict[str, float]] = {}
        branch_value_shifts: list[float] = []
        for continuation in eligible:
            continuation_action = str(continuation["action"])
            proxy = dict(continuation["opponent_hypothesis_values"])
            values = dict(proxy)
            successes_before = successful_evaluations
            for hypothesis in hypotheses:
                if evaluations >= budget:
                    break
                evaluations += 1
                model_calls += 1
                conditioned = _condition_opponent(
                    predicted_state,
                    team_is_home=attacking_home,
                    hypothesis=hypothesis,
                )
                try:
                    result = evaluate_predicted_state_action(
                        runtime,
                        conditioned,
                        action_name=continuation_action,
                        attacking_home=attacking_home,
                        horizon_s=horizon_s,
                    )
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    fallback_values += 1
                    continue
                model_calls += max(0, int(result["model_calls"]) - 1)
                successful_evaluations += 1
                combined_uncertainty = float(np.clip(
                    1.0
                    - (1.0 - first_uncertainty)
                    * (1.0 - result["uncertainty"]),
                    0.0,
                    1.0,
                ))
                trajectory_value = (
                    result["expected_value"]
                    - float(uncertainty_penalty) * combined_uncertainty
                    - 0.15 * result["turnover"]
                )
                blend = float(np.clip(
                    validation_gate["authority"]
                    * (1.0 - combined_uncertainty),
                    0.0,
                    0.50,
                ))
                values[hypothesis] = float(
                    (1.0 - blend) * float(proxy[hypothesis])
                    + blend * trajectory_value
                )
                branch_value_shifts.append(abs(
                    values[hypothesis] - float(proxy[hypothesis])
                ))
            fallback_values += len(OPPONENT_HYPOTHESES) - len(hypotheses)
            successful_pairs += successful_evaluations > successes_before
            branch_matrix[continuation_action] = values
        matrices[first_action] = branch_matrix
        first["trajectory_planning_information"] = float(np.clip(
            np.mean(branch_value_shifts) / 0.25
            if branch_value_shifts else 0.0,
            0.0,
            1.0,
        ))
    required_pairs = len(eligible) * len(eligible)
    active = (
        len(matrices) == len(eligible)
        and successful_pairs == required_pairs
    )
    if not active:
        for candidate in eligible:
            candidate["trajectory_planning_information"] = 0.0
    return (matrices if active else {}), {
        **base_audit,
        "active": active,
        "reason": (
            "validated_budgeted_predicted_state_search"
            if active else "predicted_state_rollout_failed"
        ),
        "branch_evaluations": evaluations,
        "successful_branch_evaluations": successful_evaluations,
        "failed_branch_evaluations": evaluations - successful_evaluations,
        "successful_first_continuation_pairs": successful_pairs,
        "required_first_continuation_pairs": required_pairs,
        "model_calls": model_calls,
        "first_state_rollouts": first_rollout_calls,
        "evaluated_hypotheses_per_branch": hypothesis_count,
        "fallback_values": fallback_values,
        "value_blend_authority_cap": float(validation_gate["authority"]),
        "state_source": "world_model_predicted_next_observation",
        "continuation_action_source": "reencoded_from_predicted_ball_state",
        "partial_observability_policy": (
            "one continuation action is optimized against the full response belief"
        ),
    }


def trajectory_planning_diagnostics(
    logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    audits = []
    for payload in logs:
        adoption = payload.get("world_model_decision_adoption") or {}
        for record in adoption.get("records") or []:
            context = record.get("opponent_response_context") or {}
            audit = context.get("trajectory_rollout")
            if isinstance(audit, dict):
                audits.append(audit)
    active = [audit for audit in audits if audit.get("active")]
    budgets_respected = all(
        int(audit.get("branch_evaluations", 0))
        <= int(audit.get("branch_evaluation_budget", 0))
        for audit in audits
    )
    validated = [
        audit for audit in active
        if (audit.get("validation_gate") or {}).get("active")
        and float((audit.get("validation_gate") or {}).get(
            "authority", 0.0,
        )) <= 0.50
    ]
    return {
        "version": TRAJECTORY_GAME_VERSION,
        "available": bool(audits),
        "planning_decisions": len(audits),
        "validated_predicted_state_decisions": len(validated),
        "proxy_fallback_decisions": len(audits) - len(active),
        "branch_evaluations": sum(
            int(audit.get("branch_evaluations", 0)) for audit in audits
        ),
        "model_calls": sum(int(audit.get("model_calls", 0)) for audit in audits),
        "budgets_respected": budgets_respected,
        "all_active_rollouts_holdout_validated": len(active) == len(validated),
        "all_non_causal": all(
            audit.get("causal_interpretation") is False for audit in audits
        ),
    }
