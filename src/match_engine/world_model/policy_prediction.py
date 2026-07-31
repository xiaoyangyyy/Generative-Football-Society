"""Build action-aligned, multi-horizon policy-utility forecasts."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.policy_experiment import (
    policy_horizon_key,
    policy_horizon_seconds,
)


def _probabilistic_proxy(
    runtime,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    action_name: str,
    horizon_s: float,
    physics_prior: float,
) -> dict[str, Any]:
    future = runtime.predict_future(
        observation,
        action,
        action_kind=action_name,
        horizon_s=horizon_s,
    )
    events = future.event_probabilities
    progress = float(future.progress_quantiles[1])
    retention_edge = float(events["retain"] - events["turnover"])
    xg_net = float(events["shot"] * physics_prior - 0.02 * events["turnover"])
    goal_delta = float(
        events["shot"] * physics_prior if action_name == "shot" else 0.0
    )
    return {
        "prediction_source": "probabilistic_future_proxy",
        "horizon_s": horizon_s,
        "policy_utility": (
            goal_delta
            + 0.35 * xg_net
            + 0.15 * progress
            + 0.05 * retention_edge
        ),
        "progress": progress,
        "retention_probability": float(events["retain"]),
        "xg_net_delta": xg_net,
        "goal_diff_delta": goal_delta,
        "uncertainty": float(future.state_uncertainty),
    }


def build_multi_horizon_policy_predictions(
    runtime,
    observation: np.ndarray,
    *,
    action_name: str,
    target: np.ndarray,
    physics_prior: float,
    confidence: float,
    attacking_home: bool,
    base_horizon_s: float,
    outcome_horizons_s: Iterable[float],
    residual_memory=None,
    decision_context: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    predictions = {}
    for requested in outcome_horizons_s:
        requested = max(0.0, float(requested))
        horizon_s = float(base_horizon_s) if requested <= 1e-9 else requested
        action = encode_high_level_action(
            action_name, target=target, horizon_s=horizon_s,
        )
        action[13] = physics_prior
        predictor = getattr(runtime, "predict_policy_utility", None)
        if callable(predictor):
            prediction = predictor(
                observation,
                action,
                action_kind=action_name,
                attacking_home=attacking_home,
                horizon_s=horizon_s,
            )
        else:
            prediction = _probabilistic_proxy(
                runtime,
                observation,
                action,
                action_name=action_name,
                horizon_s=horizon_s,
                physics_prior=physics_prior,
            )
        prediction = {
            key: finite_float(value, 0.0)
            if isinstance(value, (int, float, np.floating)) else value
            for key, value in prediction.items()
        }
        horizon_key = policy_horizon_key(requested)
        if residual_memory is not None:
            prediction = residual_memory.calibrate(
                prediction,
                context=decision_context or {},
                action=action_name,
                horizon_key=horizon_key,
            )
        uncertainty = float(np.clip(
            float(prediction.get("uncertainty", 1.0)), 0.0, 1.0,
        ))
        prediction["effective_confidence"] = float(
            confidence
            * (1.0 - uncertainty)
            * float(prediction.get("residual_memory_trust_factor", 1.0))
        )
        predictions[horizon_key] = prediction
    return predictions


def summarize_multi_horizon_predictions(
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Collapse forecasts into a bounded, near-term-weighted ranking signal."""
    if not predictions:
        return {
            "policy_utility": 0.0,
            "effective_confidence": 0.0,
            "memory_trust_floor": 1.0,
            "contextual_memory_active": False,
        }
    weighted_value = weighted_confidence = total_weight = 0.0
    memory_factors = []
    memory_active = False
    for horizon_key, prediction in predictions.items():
        horizon_s = policy_horizon_seconds(horizon_key)
        weight = 1.0 / (1.0 + horizon_s / 60.0)
        weighted_value += weight * float(prediction.get("policy_utility", 0.0))
        weighted_confidence += weight * float(
            prediction.get("effective_confidence", 0.0)
        )
        total_weight += weight
        memory_factors.append(float(
            prediction.get("residual_memory_trust_factor", 1.0)
        ))
        memory_active = memory_active or (
            (prediction.get("residual_memory") or {}).get("scope")
            not in {None, "insufficient_contextual_history"}
        )
    return {
        "policy_utility": weighted_value / max(total_weight, 1e-8),
        "effective_confidence": weighted_confidence / max(total_weight, 1e-8),
        "memory_trust_floor": min(memory_factors, default=1.0),
        "contextual_memory_active": memory_active,
        "horizon_weighting": "inverse_1_plus_minutes",
    }
