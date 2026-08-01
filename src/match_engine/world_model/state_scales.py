"""Auditable semantic projections of world-model ensemble trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.observation import OBS_DIM


STATE_SCALE_VERSION = 3
FALSIFIABLE_SEMANTIC_EVENTS = (
    "retain_possession",
    "enter_final_third",
    "positive_territorial_shift",
    "improve_scoreline",
)
FALSIFIABLE_DOWNSIDE_EVENTS = (
    "lose_possession",
    "negative_territorial_shift",
    "worsen_scoreline",
    "fail_enter_final_third",
)


def semantic_event_targets(
    current_observations: Any,
    future_observations: Any,
) -> np.ndarray:
    """Derive event labels only from current and realized future states."""
    current = np.asarray(current_observations, dtype=np.float64)
    future = np.asarray(future_observations, dtype=np.float64)
    if current.ndim == 1:
        current = current.reshape(1, -1)
    if future.ndim == 1:
        future = future.reshape(1, -1)
    if (
        current.ndim != 2
        or future.shape != current.shape
        or current.shape[1] != OBS_DIM
    ):
        raise ValueError("event states must have matching shape [samples, 307]")
    current = np.clip(np.nan_to_num(current), 0.0, 1.0)
    future = np.clip(np.nan_to_num(future), 0.0, 1.0)
    attacking_home = current[:, -1] >= 0.5
    direction = np.where(attacking_home, 1.0, -1.0)
    current_progress = np.where(
        attacking_home, current[:, 200], 1.0 - current[:, 200],
    )
    future_progress = np.where(
        attacking_home, future[:, 200], 1.0 - future[:, 200],
    )
    retention = np.where(
        attacking_home, future[:, 209], 1.0 - future[:, 209],
    ) >= 0.5
    current_goal_diff = direction * 5.0 * (
        current[:, 204] - current[:, 205]
    )
    future_goal_diff = direction * 5.0 * (
        future[:, 204] - future[:, 205]
    )
    return np.stack([
        retention,
        future_progress >= 0.67,
        future_progress - current_progress >= 0.03,
        future_goal_diff - current_goal_diff >= 0.5,
    ], axis=1).astype(np.float32)


def semantic_event_member_indicators(
    current_observations: Any,
    future_member_observations: Any,
) -> np.ndarray:
    """Return per-member event indicators with shape [members, samples, events]."""
    members = np.asarray(future_member_observations, dtype=np.float64)
    if members.ndim == 2:
        members = members[:, None, :]
    if members.ndim != 3 or members.shape[2] != OBS_DIM:
        raise ValueError("member states must have shape [members, samples, 307]")
    current = np.asarray(current_observations, dtype=np.float64)
    if current.ndim == 1:
        current = current.reshape(1, -1)
    if current.shape != members.shape[1:]:
        raise ValueError("current states must align with member samples")
    return np.stack([
        semantic_event_targets(current, member) for member in members
    ], axis=0)


def projected_semantic_event_probabilities(
    current_observations: Any,
    future_member_observations: Any,
    *,
    ensemble_trained: bool,
) -> np.ndarray:
    """Jeffreys-smoothed event probabilities for every sample and event."""
    indicators = semantic_event_member_indicators(
        current_observations, future_member_observations,
    )
    if not ensemble_trained:
        return np.full(indicators.shape[1:], 0.5, dtype=np.float32)
    return (
        (indicators.sum(axis=0) + 0.5) / (indicators.shape[0] + 1.0)
    ).astype(np.float32)


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


def _downside_matrix(
    current: np.ndarray,
    samples: np.ndarray,
    *,
    attacking_home: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction = 1.0 if attacking_home else -1.0
    current_progress = current[200] if attacking_home else 1.0 - current[200]
    future_progress = (
        samples[:, 200] if attacking_home else 1.0 - samples[:, 200]
    )
    progress_delta = future_progress - current_progress
    retention = (
        samples[:, 209] if attacking_home else 1.0 - samples[:, 209]
    ) >= 0.5
    current_goal_diff = direction * 5.0 * (
        current[204] - current[205]
    )
    future_goal_diff = direction * 5.0 * (
        samples[:, 204] - samples[:, 205]
    )
    goal_diff_delta = future_goal_diff - current_goal_diff
    downside = np.stack([
        ~retention,
        progress_delta <= -0.03,
        goal_diff_delta <= -0.5,
        future_progress < 0.67,
    ], axis=1)
    return downside, progress_delta, retention, goal_diff_delta


def trajectory_mode_forecast(
    current_observation: np.ndarray,
    transition_samples: Any,
    *,
    attacking_home: bool,
    ensemble_trained: bool,
    max_explicit_modes: int = 3,
    transition_trajectory_samples: Any | None = None,
) -> dict[str, Any]:
    """Group members by transparent downside-event signatures."""
    current = np.asarray(current_observation, dtype=np.float64).reshape(-1)
    if current.shape != (OBS_DIM,):
        raise ValueError("current observation must have shape [307]")
    current = np.clip(np.nan_to_num(current), 0.0, 1.0)
    current[-1] = 1.0 if attacking_home else 0.0
    samples = _samples(transition_samples)
    downside, progress_delta, retention, goal_diff_delta = _downside_matrix(
        current, samples, attacking_home=attacking_home,
    )
    trajectory_steps = 1
    path_downside = downside.copy()
    if transition_trajectory_samples is not None:
        trajectory = np.asarray(
            transition_trajectory_samples, dtype=np.float64,
        )
        if trajectory.ndim == 4 and trajectory.shape[2] == 1:
            trajectory = trajectory[:, :, 0, :]
        if (
            trajectory.ndim != 3
            or trajectory.shape[0] < 1
            or trajectory.shape[1:] != samples.shape
            or trajectory.shape[2] != OBS_DIM
        ):
            raise ValueError(
                "trajectory samples must have shape [steps, members, 307]"
            )
        trajectory = np.clip(
            np.nan_to_num(trajectory, nan=0.0, posinf=1.0), 0.0, 1.0,
        )
        if not np.allclose(trajectory[-1], samples, atol=1e-6, rtol=0.0):
            raise ValueError(
                "trajectory endpoint must equal final member states"
            )
        trajectory_steps = int(trajectory.shape[0])
        step_downside = np.stack([
            _downside_matrix(
                current, step, attacking_home=attacking_home,
            )[0]
            for step in trajectory
        ], axis=0)
        path_downside = np.any(step_downside, axis=0)
        current_progress = (
            current[200] if attacking_home else 1.0 - current[200]
        )
        path_downside[:, 3] = bool(current_progress < 0.67) & np.all(
            step_downside[:, :, 3], axis=0,
        )
    counts = {
        event: int(downside[:, index].sum())
        for index, event in enumerate(FALSIFIABLE_DOWNSIDE_EVENTS)
    }
    probabilities = {
        event: (
            _smoothed_probability(downside[:, index])
            if ensemble_trained else 0.5
        )
        for index, event in enumerate(FALSIFIABLE_DOWNSIDE_EVENTS)
    }
    path_counts = {
        event: int(path_downside[:, index].sum())
        for index, event in enumerate(FALSIFIABLE_DOWNSIDE_EVENTS)
    }
    path_probabilities = {
        event: (
            _smoothed_probability(path_downside[:, index])
            if ensemble_trained else 0.5
        )
        for index, event in enumerate(FALSIFIABLE_DOWNSIDE_EVENTS)
    }
    if not ensemble_trained or len(samples) < 2:
        return {
            "version": 2,
            "available": False,
            "reason": "trained_transition_ensemble_required",
            "ensemble_members": int(len(samples)),
            "modes": [],
            "mode_count": 0,
            "normalized_mode_entropy": 0.0,
            "downside_event_counts": counts,
            "downside_event_probabilities": probabilities,
            "path_downside_event_counts": path_counts,
            "path_downside_event_probabilities": path_probabilities,
            "trajectory_steps": trajectory_steps,
            "temporal_path_available": False,
            "path_counts_trusted": False,
            "counts_trusted": False,
            "causal_interpretation": False,
        }
    groups: dict[tuple[int, ...], list[int]] = {}
    signatures = np.concatenate([downside, path_downside], axis=1).astype(int)
    for member_index, signature in enumerate(signatures):
        groups.setdefault(tuple(map(int, signature)), []).append(member_index)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    explicit = ordered[:max(1, int(max_explicit_modes))]
    if len(ordered) > len(explicit):
        tail = [index for _, indices in ordered[len(explicit):] for index in indices]
        explicit.append((tuple([-1] * signatures.shape[1]), tail))
    modes = []
    for mode_index, (signature, member_indices) in enumerate(explicit):
        selected = np.asarray(member_indices, dtype=int)
        event_rates = {
            event: float(np.mean(downside[selected, event_index]))
            for event_index, event in enumerate(FALSIFIABLE_DOWNSIDE_EVENTS)
        }
        path_event_rates = {
            event: float(np.mean(path_downside[selected, event_index]))
            for event_index, event in enumerate(FALSIFIABLE_DOWNSIDE_EVENTS)
        }
        modes.append({
            "mode_id": f"mode_{mode_index}",
            "kind": "residual_mixed_signatures" if -1 in signature else (
                "exact_downside_signature"
            ),
            "members": len(member_indices),
            "member_indices": list(map(int, member_indices)),
            "probability": float(len(member_indices) / len(samples)),
            "downside_events": event_rates,
            "terminal_downside_events": event_rates,
            "path_downside_events": path_event_rates,
            "mean_downside_rate": float(np.mean(list(event_rates.values()))),
            "mean_progress_delta": float(np.mean(progress_delta[selected])),
            "retention_probability": float(np.mean(retention[selected])),
            "mean_goal_diff_delta": float(np.mean(goal_diff_delta[selected])),
        })
    mode_probabilities = np.asarray([
        mode["probability"] for mode in modes
    ], dtype=np.float64)
    entropy = float(-np.sum(
        mode_probabilities * np.log(np.clip(mode_probabilities, 1e-12, 1.0))
    ))
    entropy /= float(np.log(max(2, len(modes))))
    return {
        "version": 2,
        "available": True,
        "reason": "transparent_member_downside_signature_partition",
        "ensemble_members": int(len(samples)),
        "modes": modes,
        "mode_count": len(modes),
        "normalized_mode_entropy": float(np.clip(entropy, 0.0, 1.0)),
        "multimodal": len(modes) >= 2,
        "downside_event_counts": counts,
        "downside_event_probabilities": probabilities,
        "path_downside_event_counts": path_counts,
        "path_downside_event_probabilities": path_probabilities,
        "trajectory_steps": trajectory_steps,
        "temporal_path_available": trajectory_steps >= 2,
        "path_counts_trusted": trajectory_steps >= 2,
        "counts_trusted": True,
        "probability_source": "jeffreys_smoothed_member_frequency",
        "grouping_source": (
            "exact_terminal_and_pathwise_downside_event_signatures"
        ),
        "causal_interpretation": False,
    }


def multiscale_state_forecast(
    current_observation: np.ndarray,
    transition_samples: Any,
    *,
    attacking_home: bool,
    horizon_s: float,
    ensemble_trained: bool = True,
    transition_trajectory_samples: Any | None = None,
) -> dict[str, Any]:
    """Project raw member states into short, tactical and strategic evidence."""
    current = np.asarray(current_observation, dtype=np.float64).reshape(-1)
    if current.shape != (OBS_DIM,):
        raise ValueError("current observation must have shape [307]")
    current = np.clip(
        np.nan_to_num(current, nan=0.0, posinf=1.0), 0.0, 1.0,
    )
    # The explicit runtime orientation is authoritative; sparse or synthetic
    # observations may not carry a reliable attack flag.
    current[-1] = 1.0 if attacking_home else 0.0
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
    projected = projected_semantic_event_probabilities(
        current.reshape(1, -1), samples[:, None, :],
        ensemble_trained=ensemble_trained,
    )[0]
    events = dict(zip(FALSIFIABLE_SEMANTIC_EVENTS, map(float, projected)))
    events["relieve_ball_pressure"] = (
        _smoothed_probability(future_pressure <= current_pressure - 0.05)
        if ensemble_trained else 0.5
    )
    horizon = max(0.0, float(horizon_s))
    primary_scale = (
        "short" if horizon <= 20.0 else "tactical" if horizon <= 120.0
        else "strategic"
    )
    trajectory_modes = trajectory_mode_forecast(
        current,
        samples,
        attacking_home=attacking_home,
        ensemble_trained=ensemble_trained,
        transition_trajectory_samples=transition_trajectory_samples,
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
        "trajectory_modes": trajectory_modes,
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
