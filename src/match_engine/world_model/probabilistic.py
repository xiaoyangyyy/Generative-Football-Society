"""Event-level, multimodal future representation layered over world-model outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

EVENTS = ("retain", "turnover", "shot", "foul", "out")


@dataclass(frozen=True)
class EventTransitionModel:
    """Auditable action-conditioned posterior predictive event model."""
    action_probabilities: dict[str, tuple[float, ...]]
    action_counts: dict[str, int]

    def __post_init__(self) -> None:
        for action, values in self.action_probabilities.items():
            probability = np.asarray(values, dtype=float)
            if probability.shape != (len(EVENTS),) or not np.isclose(probability.sum(), 1.0):
                raise ValueError(f"invalid event simplex for {action}")

    def predict(self, action_kind: str) -> dict[str, float]:
        values = self.action_probabilities.get(action_kind, self.action_probabilities["other"])
        return dict(zip(EVENTS, values))

    def to_dict(self) -> dict:
        return {"version": 1, "events": list(EVENTS),
                "action_probabilities": {k: list(v) for k, v in self.action_probabilities.items()},
                "action_counts": self.action_counts}

    @classmethod
    def load(cls, path: str | Path) -> "EventTransitionModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or tuple(payload.get("events", ())) != EVENTS:
            raise ValueError("unsupported transition model schema")
        return cls({k: tuple(v) for k, v in payload["action_probabilities"].items()}, payload["action_counts"])


def fit_event_transition_model(actions: np.ndarray, outcomes: np.ndarray, alpha: float = 0.5) -> EventTransitionModel:
    actions, outcomes = np.asarray(actions, str), np.asarray(outcomes, str)
    if actions.shape != outcomes.shape or not set(outcomes).issubset(EVENTS):
        raise ValueError("invalid transition observations")
    probabilities, counts = {}, {}
    for action in sorted(set(actions) | {"other"}):
        selected = outcomes[actions == action] if action != "other" else outcomes
        histogram = np.array([(selected == event).sum() for event in EVENTS], dtype=float) + alpha
        probabilities[action] = tuple((histogram / histogram.sum()).tolist())
        counts[action] = int(len(selected))
    return EventTransitionModel(probabilities, counts)


@dataclass(frozen=True)
class ProbabilisticFuture:
    event_probabilities: dict[str, float]
    event_time_s: tuple[float, float]
    progress_quantiles: tuple[float, float, float]
    state_uncertainty: float

    def __post_init__(self) -> None:
        total = sum(self.event_probabilities.values())
        if set(self.event_probabilities) != set(EVENTS) or not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("event probabilities must cover the canonical simplex")


def future_from_ensemble(
    *,
    pass_probabilities: np.ndarray,
    shot_probabilities: np.ndarray,
    progress_samples: np.ndarray,
    action_kind: str,
    horizon_s: float,
) -> ProbabilisticFuture:
    passes = np.clip(np.asarray(pass_probabilities, dtype=float), 0.0, 1.0)
    shots = np.clip(np.asarray(shot_probabilities, dtype=float), 0.0, 1.0)
    progress = np.asarray(progress_samples, dtype=float)
    if not len(passes) or not len(shots) or not len(progress):
        raise ValueError("ensemble samples are required")
    retain = float(passes.mean()) if action_kind == "pass" else 0.62
    shot = float(shots.mean()) if action_kind == "shot" else 0.04
    turnover = max(0.01, 1.0 - retain)
    raw = np.array([retain, turnover, shot, 0.025, 0.015], dtype=float)
    raw /= raw.sum()
    uncertainty = float(np.clip(np.std(passes) + np.std(shots) + np.std(progress), 0.0, 1.0))
    expected_time = max(0.1, float(horizon_s) * (0.45 + 0.4 * uncertainty))
    return ProbabilisticFuture(
        dict(zip(EVENTS, raw.tolist())),
        (expected_time, max(0.05, expected_time * (0.2 + uncertainty))),
        tuple(float(v) for v in np.quantile(progress, [0.1, 0.5, 0.9])),
        uncertainty,
    )
