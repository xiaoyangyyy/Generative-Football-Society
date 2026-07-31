"""Auditable world-model decision packets for strategic LLM consumers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.schema import observation_coverage
from src.match_engine.tactical_catalog import TACTICAL_PRESETS, resolve_tactical_preset


DECISION_PACKET_VERSION = 1
COACH_ACTIONS = ("hold", "pass", "cross", "shot")
PREMATCH_TACTICAL_CANDIDATES = (
    "balanced",
    "gegenpress",
    "possession_control",
    "counter_attack",
    "low_block",
    "wing_play",
    "direct_vertical",
)


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


def _evaluate_action_candidates(
    runtime,
    state,
    team_id: str,
    *,
    horizon_s: float,
    uncertainty_penalty: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run the shared short-horizon action counterfactuals once."""
    attacking_home = team_id == state.home.team_id
    observation = runtime.encode_state(state, attacking_home=attacking_home)
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
    return observation, candidates


def _tactical_action_weights(preset_name: str) -> dict[str, float]:
    """Translate a tactical vector into an auditable high-level policy mixture."""
    preset = TACTICAL_PRESETS[resolve_tactical_preset(preset_name)]
    raw = {
        "hold": (
            0.08
            + 0.24 * preset.get("possession_orientation", 0.5)
            + 0.12 * preset.get("low_block", 0.0)
        ),
        "pass": (
            0.18
            + 0.30 * preset.get("build_up_short", 0.5)
            + 0.18 * preset.get("through_ball_bias", 0.5)
        ),
        "cross": (
            0.05
            + 0.30 * preset.get("cross_frequency", 0.4)
            + 0.20 * preset.get("wing_focus", 0.5)
        ),
        "shot": (
            0.06
            + 0.18 * preset.get("verticality", 0.5)
            + 0.16 * preset.get("risk_budget", 0.5)
        ),
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def build_prematch_tactical_packet(
    runtime,
    state,
    team_id: str,
    *,
    horizon_s: float = 20.0,
    candidate_presets: tuple[str, ...] = PREMATCH_TACTICAL_CANDIDATES,
    uncertainty_penalty: float = 0.25,
) -> dict[str, Any]:
    """Rank tactical policy mixtures using bounded world-model evidence."""
    if runtime is None:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": "world_model_unavailable",
            "recommended_tactical_preset": "none",
            "candidates": [],
        }
    try:
        normalized = tuple(dict.fromkeys(
            resolve_tactical_preset(name) for name in candidate_presets
        ))
        observation, action_evidence = _evaluate_action_candidates(
            runtime, state, team_id, horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
        )
        by_action = {item["action"]: item for item in action_evidence}
        tactical_candidates = []
        for preset_name in normalized:
            weights = _tactical_action_weights(preset_name)

            def weighted(field: str) -> float:
                return float(sum(
                    weights[action] * by_action[action][field]
                    for action in COACH_ACTIONS
                ))

            event_probabilities = {
                event: float(sum(
                    weights[action]
                    * by_action[action]["event_probabilities"].get(event, 0.0)
                    for action in COACH_ACTIONS
                ))
                for event in ("retain", "turnover", "shot", "foul", "out")
            }
            vector = TACTICAL_PRESETS[preset_name]
            fatigue_cost = float(np.clip(
                0.55 * vector.get("pressing_intensity", 0.5)
                + 0.25 * vector.get("tempo", 0.5)
                + 0.20 * vector.get("counterpress", 0.5),
                0.0, 1.0,
            ))
            structural_risk = float(np.clip(
                0.45 * vector.get("risk_budget", 0.5)
                + 0.35 * vector.get("line_height", 0.5)
                + 0.20 * (1.0 - vector.get("compactness", 0.5)),
                0.0, 1.0,
            ))
            model_value = weighted("risk_adjusted_value")
            tactical_value = (
                model_value
                + 0.10 * event_probabilities["shot"]
                + 0.06 * event_probabilities["retain"]
                - 0.08 * fatigue_cost
                - 0.10 * structural_risk
            )
            tactical_candidates.append({
                "tactical_preset": preset_name,
                "policy_action_weights": weights,
                "model_value": finite_float(model_value, -1.0),
                "risk_adjusted_value": finite_float(tactical_value, -1.0),
                "effective_confidence": weighted("effective_confidence"),
                "uncertainty": weighted("uncertainty"),
                "event_probabilities": event_probabilities,
                "fatigue_cost_proxy": fatigue_cost,
                "structural_risk_proxy": structural_risk,
            })
        eligible = [
            candidate for candidate in tactical_candidates
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
            "evaluation_scope": "short_horizon_tactical_policy_proxy",
            "observation_coverage": observation_coverage(observation),
            "recommended_tactical_preset": (
                best["tactical_preset"] if best else "none"
            ),
            "recommendation_confidence": (
                best["effective_confidence"] if best else 0.0
            ),
            "candidates": tactical_candidates,
            "action_evidence": action_evidence,
            "limitations": [
                "Not a full-match win-probability forecast.",
                "Opponent adaptation is delegated to the coach model.",
                "Tactical presets are evaluated as high-level action mixtures.",
            ],
            "policy": (
                "Select only an evaluated candidate, use model uncertainty, "
                "and explain any disagreement with the recommendation."
            ),
        }
    except Exception as exc:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": f"world_model_error:{type(exc).__name__}",
            "recommended_tactical_preset": "none",
            "candidates": [],
        }


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
    try:
        observation, candidates = _evaluate_action_candidates(
            runtime, state, team_id, horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
        )
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
