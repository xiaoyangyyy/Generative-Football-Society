"""Evidence gate and counterfactual value contract for shot planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

SHOT_FEATURES = ("goal_distance", "goal_angle", "centrality", "nearest_defender", "keeper_distance")


@dataclass(frozen=True)
class ShotProbabilityModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    calibration_knots: tuple[float, ...] = ()
    calibration_values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        expected = (len(SHOT_FEATURES),)
        for vector in (self.mean, self.scale, self.weights):
            if np.asarray(vector).shape != expected or not np.isfinite(vector).all():
                raise ValueError(f"shot vectors must have shape {expected}")

    def predict(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(SHOT_FEATURES),):
            raise ValueError("invalid shot feature shape")
        filled = np.where(np.isfinite(values), values, self.mean)
        z = (filled - self.mean) / np.maximum(self.scale, 1e-8)
        probability = float(1.0 / (1.0 + np.exp(-np.clip(z @ self.weights + self.bias, -30.0, 30.0))))
        if self.calibration_knots:
            probability = float(np.interp(probability, self.calibration_knots, self.calibration_values))
        return probability

    def to_dict(self) -> dict:
        return {"version": 1, "features": list(SHOT_FEATURES), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "weights": self.weights.tolist(), "bias": self.bias,
                "calibration_knots": list(self.calibration_knots),
                "calibration_values": list(self.calibration_values)}

    @classmethod
    def load(cls, path: str | Path) -> "ShotProbabilityModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or tuple(payload.get("features", ())) != SHOT_FEATURES:
            raise ValueError("unsupported shot model schema")
        return cls(np.asarray(payload["mean"], dtype=float), np.asarray(payload["scale"], dtype=float),
                   np.asarray(payload["weights"], dtype=float), float(payload["bias"]),
                   tuple(payload.get("calibration_knots", ())), tuple(payload.get("calibration_values", ())))


def fit_shot_probability_model(features: np.ndarray, goals: np.ndarray, *, steps: int = 1500,
                               learning_rate: float = 0.04) -> ShotProbabilityModel:
    x, y = np.asarray(features, dtype=float), np.asarray(goals, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(SHOT_FEATURES) or y.shape != (len(x),):
        raise ValueError("invalid shot training data")
    mean, scale = np.nanmean(x, axis=0), np.maximum(np.nanstd(x, axis=0), 1e-8)
    z = (np.where(np.isfinite(x), x, mean) - mean) / scale
    weights = np.zeros(z.shape[1], dtype=float)
    prevalence = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    bias = float(np.log(prevalence / (1.0 - prevalence)))
    for _ in range(max(1, steps)):
        probability = 1.0 / (1.0 + np.exp(-np.clip(z @ weights + bias, -30.0, 30.0)))
        error = probability - y
        weights -= learning_rate * (z.T @ error / len(z) + 0.002 * weights)
        bias -= learning_rate * float(error.mean())
    return ShotProbabilityModel(mean, scale, weights, bias)


@dataclass(frozen=True)
class ShotEvidence:
    shots: int
    goals: int
    matches: int
    providers: int
    brier: float
    baseline_brier: float

    def gate(self) -> dict[str, bool]:
        return {
            "shots": self.shots >= 1000,
            "goals": self.goals >= 80,
            "matches": self.matches >= 20,
            "providers": self.providers >= 2,
            "proper_score_gain": self.brier <= self.baseline_brier - 0.005,
        }

    @property
    def enabled(self) -> bool:
        return all(self.gate().values())


def shot_counterfactual_value(
    *,
    goal_probability: float,
    rebound_value: float,
    turnover_cost: float,
    best_continuation_value: float,
    uncertainty: float,
) -> float:
    shot = (
        np.clip(goal_probability, 0.0, 1.0)
        + 0.25 * np.clip(rebound_value, 0.0, 1.0)
        - 0.20 * np.clip(turnover_cost, 0.0, 1.0)
    )
    advantage = shot - float(best_continuation_value)
    return float((1.0 - np.clip(uncertainty, 0.0, 1.0)) * advantage)
