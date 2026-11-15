from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mask_infeasible_actions(
    labels: Sequence[str],
    probabilities: Sequence[float],
    feasible_actions: set[str],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).copy()
    if values.shape != (len(labels),):
        raise ValueError("action probability shape does not match labels")
    feasible = np.asarray([
        action in feasible_actions for action in labels
    ], dtype=bool)
    if not feasible.any():
        raise ValueError("at least one action must be feasible")
    values = np.where(feasible, np.clip(values, 0.0, None), 0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        values = feasible.astype(float)
        total = float(values.sum())
    return values / total


def sample_action(
    labels: Sequence[str], probabilities: Sequence[float], draw: float,
) -> str:
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("action probabilities must have positive finite mass")
    values /= total
    uniform = float(np.clip(draw, 0.0, np.nextafter(1.0, 0.0)))
    index = min(
        len(labels) - 1,
        int(np.searchsorted(np.cumsum(values), uniform, side="right")),
    )
    return str(labels[index])
