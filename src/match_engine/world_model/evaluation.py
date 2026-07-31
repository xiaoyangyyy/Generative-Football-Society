"""Reusable metrics and contracts for world-model capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.schema import (
    ATTACK_FLAG,
    BALL,
    GRID,
    PLAYERS,
    SCORE_CLOCK_CROWD_POSSESSION,
    TACTICS,
    observation_loss_weights,
)


FEATURE_GROUPS = {
    "grid": GRID,
    "ball": BALL,
    "match_context": SCORE_CLOCK_CROWD_POSSESSION,
    "players": PLAYERS,
    "tactics": TACTICS,
    "attack_flag": ATTACK_FLAG,
}


def validate_transition_batch(obs, actions, next_obs) -> tuple[np.ndarray, ...]:
    """Validate and normalize a batch of canonical v6 transitions."""
    observations = np.asarray(obs, dtype=np.float32)
    action_vectors = np.asarray(actions, dtype=np.float32)
    targets = np.asarray(next_obs, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != OBS_DIM:
        raise ValueError(f"obs must have shape (N, {OBS_DIM})")
    if targets.shape != observations.shape:
        raise ValueError("next_obs must have the same shape as obs")
    if action_vectors.ndim != 2 or action_vectors.shape != (
        len(observations), ACTION_DIM,
    ):
        raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
    if not all(np.isfinite(values).all() for values in (
        observations, action_vectors, targets,
    )):
        raise ValueError("world-model transitions must be finite")
    return observations, action_vectors, targets


def expected_calibration_error(
    probabilities: np.ndarray, truth: np.ndarray, *, bins: int = 8,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    if not len(probabilities):
        return math.nan
    score = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (
            probabilities < high if high < 1.0 else probabilities <= high
        )
        if mask.any():
            score += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(truth[mask].mean())
            )
    return float(score)


def binary_auc(probabilities: np.ndarray, truth: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    positive, negative = probabilities[truth], probabilities[~truth]
    if not len(positive) or not len(negative):
        return 0.5
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


@dataclass
class TransitionMetricAccumulator:
    """Streaming transition metrics, suitable for large sealed test sets."""

    samples: int = 0
    nonfinite: int = 0
    squared_error_sum: float = 0.0
    weighted_error_sum: float = 0.0
    persistence_error_sum: float = 0.0
    group_error_sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in FEATURE_GROUPS}
    )
    errors: list[float] = field(default_factory=list)
    uncertainties: list[float] = field(default_factory=list)

    def update(self, prediction, target, current, *, uncertainty=math.nan) -> None:
        prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        if any(values.shape != (OBS_DIM,) for values in (
            prediction, target, current,
        )):
            raise ValueError(f"transition vectors must have shape ({OBS_DIM},)")
        if not all(np.isfinite(values).all() for values in (
            prediction, target, current,
        )):
            self.nonfinite += 1
            return
        squared = np.square(prediction - target, dtype=np.float64)
        baseline_squared = np.square(current - target, dtype=np.float64)
        weights = observation_loss_weights().astype(np.float64)
        error = float(squared.mean())
        self.samples += 1
        self.squared_error_sum += error
        self.weighted_error_sum += float(np.mean(squared * weights))
        self.persistence_error_sum += float(np.mean(baseline_squared * weights))
        for name, section in FEATURE_GROUPS.items():
            self.group_error_sums[name] += float(squared[section].mean())
        if np.isfinite(uncertainty):
            self.errors.append(error)
            self.uncertainties.append(float(uncertainty))

    def finalize(self) -> dict:
        if not self.samples:
            return {"samples": 0, "nonfinite": self.nonfinite}
        weighted = self.weighted_error_sum / self.samples
        persistence = self.persistence_error_sum / self.samples
        uncertainty = _uncertainty_diagnostics(self.errors, self.uncertainties)
        return {
            "samples": self.samples,
            "nonfinite": self.nonfinite,
            "mse": self.squared_error_sum / self.samples,
            "weighted_mse": weighted,
            "persistence_weighted_mse": persistence,
            "skill_vs_persistence": (
                1.0 - weighted / persistence if persistence > 1e-12 else 0.0
            ),
            "feature_mse": {
                name: value / self.samples
                for name, value in self.group_error_sums.items()
            },
            "uncertainty": uncertainty,
        }


def _uncertainty_diagnostics(errors, uncertainties) -> dict:
    errors = np.asarray(errors, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    if len(errors) < 2 or np.std(errors) <= 1e-12 or np.std(uncertainties) <= 1e-12:
        return {"samples": int(len(errors)), "error_correlation": 0.0, "high_uncertainty_error_lift": 1.0}
    correlation = float(np.corrcoef(errors, uncertainties)[0, 1])
    cutoff = float(np.quantile(uncertainties, 0.75))
    selected = errors[uncertainties >= cutoff]
    lift = float(selected.mean() / max(errors.mean(), 1e-12))
    return {
        "samples": int(len(errors)),
        "error_correlation": correlation,
        "high_uncertainty_error_lift": lift,
    }
