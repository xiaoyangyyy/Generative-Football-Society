"""Auditable world-model decision packets for strategic LLM consumers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.schema import observation_coverage


DECISION_PACKET_VERSION = 1
COACH_ACTIONS = ("hold", "pass", "cross", "shot")


def _candidate_target(state, action: str, attacking_home: bool) -> np.ndarray:
    ball = np.asarray(state.ball.position, dtype=np.float32).copy()
    direction = 1.0 if attacking_home else -1.0
    if action == "hold":
        return ball
    if action == "pass":
        return np.clip(ball + np.array([0.14 * direction, 0.0]), 0.02, 0.98)
    if action == "cross":
        return np.array([0.91 if attacking_home else 0.09, 0.50], dtype=np.float32)
    return np.array([0.995 if attacking_home else 0.005, 0.50], dtype=np.float32)


def _physics_prior(state, action: str, attacking_home: bool) -> float:
    if action == "hold":
        return 0.90
    if action == "pass":
        return 0.72
    if action == "cross":
        return 0.46
    goal_x = 0.995 if attacking_home else 0.005
    distance = abs(goal_x - float(state.ball.position[0]))
    return float(np.clip(0.35 * (1.0 - distance), 0.03, 0.45))


def build_coach_decision_packet(
    runtime,
    state,
    team_id: str,
    *,
    horizon_s: float = 10.0,
    uncertainty_penalty: float = 0.25,
) -> dict[str, Any]:
    """Compare strategic candidates without granting the LLM direct state writes."""
    if runtime is None:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": "world_model_unavailable",
            "recommended_action": "none",
            "candidates": [],
        }
    attacking_home = team_id == state.home.team_id
    try:
        observation = runtime.encode_state(
            state, attacking_home=attacking_home,
        )
        candidates = []
        for action_name in COACH_ACTIONS:
            target = _candidate_target(state, action_name, attacking_home)
            action = encode_high_level_action(
                action_name, target=target, horizon_s=horizon_s,
            )
            prior = _physics_prior(state, action_name, attacking_home)
            action[13] = prior
            quality_kind = "shot" if action_name == "shot" else "pass"
            confidence = float(runtime.planner_confidence(
                observation, kind=quality_kind,
            ))
            future = runtime.predict_future(
                observation, action, action_kind=action_name,
                horizon_s=horizon_s,
            )
            if action_name == "shot":
                expected_value = runtime.score_shot_action(
                    observation, action, attacking_home=attacking_home,
                )
            else:
                expected_value = runtime.score_action(observation, action)
            uncertainty = float(np.clip(future.state_uncertainty, 0.0, 1.0))
            effective_confidence = confidence * (1.0 - uncertainty)
            turnover = float(future.event_probabilities["turnover"])
            risk_adjusted = (
                finite_float(expected_value, 0.0)
                - uncertainty_penalty * uncertainty
                - 0.15 * turnover
            )
            candidates.append({
                "action": action_name,
                "target": [float(target[0]), float(target[1])],
                "physics_prior": prior,
                "expected_value": finite_float(expected_value, 0.0),
                "risk_adjusted_value": finite_float(risk_adjusted, -1.0),
                "confidence": confidence,
                "effective_confidence": effective_confidence,
                "uncertainty": uncertainty,
                "event_probabilities": {
                    key: float(value)
                    for key, value in future.event_probabilities.items()
                },
                "progress_quantiles": [
                    float(value) for value in future.progress_quantiles
                ],
                "event_time_s": [float(value) for value in future.event_time_s],
            })
        eligible = [
            candidate for candidate in candidates
            if candidate["effective_confidence"] > 0.0
        ]
        best = max(
            eligible, key=lambda item: item["risk_adjusted_value"],
            default=None,
        )
        return {
            "version": DECISION_PACKET_VERSION,
            "available": best is not None,
            "reason": "ok" if best is not None else "quality_gate_closed",
            "team_id": team_id,
            "horizon_s": float(horizon_s),
            "observation_coverage": observation_coverage(observation),
            "recommended_action": best["action"] if best else "none",
            "recommendation_confidence": (
                best["effective_confidence"] if best else 0.0
            ),
            "candidates": candidates,
            "policy": (
                "Use as uncertain evidence; retain bounded controls and never "
                "invent score, xG, or outcome facts."
            ),
        }
    except Exception as exc:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": f"world_model_error:{type(exc).__name__}",
            "recommended_action": "none",
            "candidates": [],
        }
