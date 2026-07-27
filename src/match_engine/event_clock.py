"""Continuous-time competing-risk event clock for match events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TimedEvent:
    kind: str
    wait_seconds: float
    total_intensity: float
    probabilities: dict[str, float]


class CompetingRiskClock:
    def sample(self, intensities: Mapping[str, float], rng: np.random.Generator) -> TimedEvent | None:
        clean = {str(key): max(0.0, float(value)) for key, value in intensities.items()}
        if not clean or not all(np.isfinite(value) for value in clean.values()):
            raise ValueError("Event intensities must be finite")
        total = float(sum(clean.values()))
        if total <= 0.0:
            return None
        names = tuple(clean)
        probs = np.asarray([clean[name] / total for name in names], dtype=float)
        wait = float(rng.exponential(1.0 / total))
        kind = names[int(rng.choice(len(names), p=probs))]
        return TimedEvent(kind, wait, total, dict(zip(names, probs.tolist())))

    @staticmethod
    def modulate(
        base: Mapping[str, float],
        features: Mapping[str, float],
        coefficients: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        result = {}
        for event, rate in base.items():
            log_multiplier = sum(
                float(weight) * float(features.get(feature, 0.0))
                for feature, weight in coefficients.get(event, {}).items()
            )
            result[event] = max(0.0, float(rate)) * float(np.exp(np.clip(log_multiplier, -8.0, 8.0)))
        return result
