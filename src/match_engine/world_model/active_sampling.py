"""Prioritize informative transitions for world-model data collection."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.match_engine.world_model.schema import observation_coverage


@dataclass(order=True)
class SamplingCandidate:
    priority: float
    sequence: int
    payload: Any = field(compare=False)
    reason: dict[str, float] = field(compare=False, default_factory=dict)


class ActiveSamplingQueue:
    def __init__(self, capacity: int = 1000):
        self.capacity = max(1, int(capacity))
        self._heap: list[SamplingCandidate] = []
        self._sequence = 0

    @staticmethod
    def score(
        obs: np.ndarray,
        *,
        uncertainty: float,
        event_count: int = 0,
        target_event_count: int = 100,
        planner_advantage: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        coverage = observation_coverage(obs)
        rarity = float(1.0 / np.sqrt(1.0 + max(0, event_count)))
        target_pressure = float(np.clip((target_event_count - event_count) / max(1, target_event_count), 0.0, 1.0))
        components = {
            "uncertainty": max(0.0, float(uncertainty)),
            "coverage": coverage,
            "rarity": rarity,
            "target_pressure": target_pressure,
            "planner_advantage": abs(float(planner_advantage)),
        }
        priority = coverage * (
            0.50 * components["uncertainty"]
            + 0.20 * rarity
            + 0.20 * target_pressure
            + 0.10 * components["planner_advantage"]
        )
        return float(priority), components

    def add(self, payload: Any, obs: np.ndarray, **signals: Any) -> SamplingCandidate:
        priority, reason = self.score(obs, **signals)
        candidate = SamplingCandidate(priority, self._sequence, payload, reason)
        self._sequence += 1
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, candidate)
        elif priority > self._heap[0].priority:
            heapq.heapreplace(self._heap, candidate)
        return candidate

    def highest(self, limit: int = 100) -> list[SamplingCandidate]:
        return sorted(self._heap, reverse=True)[: max(0, int(limit))]
