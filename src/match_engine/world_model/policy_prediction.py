"""Build action-aligned, multi-horizon policy-utility forecasts."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.policy_experiment import policy_horizon_key


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
        uncertainty = float(np.clip(
            float(prediction.get("uncertainty", 1.0)), 0.0, 1.0,
        ))
        prediction["effective_confidence"] = float(
            confidence * (1.0 - uncertainty)
        )
        predictions[policy_horizon_key(requested)] = prediction
    return predictions
