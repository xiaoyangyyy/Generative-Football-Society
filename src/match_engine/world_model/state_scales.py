"""Auditable semantic projections of world-model ensemble trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.observation import OBS_DIM


STATE_SCALE_VERSION = 1
FALSIFIABLE_SEMANTIC_EVENTS = (
    "retain_possession",
    "enter_final_third",
    "positive_territorial_shift",
    "improve_scoreline",
)


def _samples(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[1] != OBS_DIM or len(array) < 1:
        raise ValueError("transition samples must have shape [members, 307]")
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=1.0), 0.0, 1.0)


def _smoothed_probability(mask: np.ndarray) -> float:
    """Jeffreys-smoothed ensemble event probability, avoiding false 0/1s."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return float((values.sum() + 0.5) / (len(values) + 1.0))


def _pressure_at_ball(observations: np.ndarray) -> np.ndarray:
    ball_x = np.clip((observations[:, 200] * 7.0).round().astype(int), 0, 7)
    ball_y = np.clip((observations[:, 201] * 4.0).round().astype(int), 0, 4)
    indices = 80 + ball_x * 5 + ball_y
    return observations[np.arange(len(observations)), indices]


def multiscale_state_forecast(
    current_observation: np.ndarray,
    transition_samples: Any,
    *,
    attacking_home: bool,
    horizon_s: float,
    ensemble_trained: bool = True,
) -> dict[str, Any]:
    """Project raw member states into short, tactical and strategic evidence."""
    current = np.asarray(current_observation, dtype=np.float64).reshape(-1)
    if current.shape != (OBS_DIM,):
        raise ValueError("current observation must have shape [307]")
    current = np.clip(
        np.nan_to_num(current, nan=0.0, posinf=1.0), 0.0, 1.0,
    )
    samples = _samples(transition_samples)
    direction = 1.0 if attacking_home else -1.0
    current_progress = current[200] if attacking_home else 1.0 - current[200]
    future_progress = (
        samples[:, 200] if attacking_home else 1.0 - samples[:, 200]
    )
    progress_delta = future_progress - current_progress
    possession = (
        samples[:, 209] if attacking_home else 1.0 - samples[:, 209]
    )
    current_goal_diff = direction * 5.0 * (current[204] - current[205])
    future_goal_diff = direction * 5.0 * (
        samples[:, 204] - samples[:, 205]
    )
    goal_diff_delta = future_goal_diff - current_goal_diff
    current_pressure = _pressure_at_ball(current.reshape(1, -1))[0]
    future_pressure = _pressure_at_ball(samples)
    speed_components = (samples[:, 202:204] - 0.5) / 2.0
    ball_speed = np.linalg.vector_norm(speed_components, axis=1)
    tactic_shift = np.mean(
        np.abs(samples[:, 298:306] - current[298:306]), axis=1,
    )
    events = {
        "retain_possession": _smoothed_probability(possession >= 0.5),
        "enter_final_third": _smoothed_probability(future_progress >= 0.67),
        "positive_territorial_shift": _smoothed_probability(
            progress_delta >= 0.03
        ),
        "relieve_ball_pressure": _smoothed_probability(
            future_pressure <= current_pressure - 0.05
        ),
        "improve_scoreline": _smoothed_probability(goal_diff_delta >= 0.5),
    }
    if not ensemble_trained:
        events = {event: 0.5 for event in events}
    horizon = max(0.0, float(horizon_s))
    primary_scale = (
        "short" if horizon <= 20.0 else "tactical" if horizon <= 120.0
        else "strategic"
    )
    return {
        "version": STATE_SCALE_VERSION,
        "source": "world_model_transition_ensemble_projection",
        "primary_scale": primary_scale,
        "horizon_s": horizon,
        "ensemble_members": int(len(samples)),
        "short": {
            "mean_progress_delta": float(np.mean(progress_delta)),
            "progress_std": float(np.std(progress_delta)),
            "mean_ball_speed": float(np.mean(ball_speed)),
            "possession_probability": float(np.mean(possession)),
        },
        "tactical": {
            "final_third_probability": events["enter_final_third"],
            "territorial_gain_probability": events[
                "positive_territorial_shift"
            ],
            "mean_pressure_at_ball": float(np.mean(future_pressure)),
            "pressure_relief_probability": events["relieve_ball_pressure"],
            "mean_tactical_shift_l1": float(np.mean(tactic_shift)),
        },
        "strategic": {
            "mean_goal_diff_delta": float(np.mean(goal_diff_delta)),
            "score_improvement_probability": events["improve_scoreline"],
            "mean_clock_fraction": float(np.mean(samples[:, 207])),
        },
        "semantic_event_probabilities": events,
        "claims": {
            "observed": False,
            "causal": False,
            "ensemble_trained": bool(ensemble_trained),
            "probability_source": (
                "jeffreys_smoothed_member_frequency"
                if ensemble_trained else "untrained_ensemble_neutral"
            ),
        },
    }
