"""Development-only shrinkage calibration for multi-step state rollouts."""

from __future__ import annotations

import numpy as np


def apply_rollout_blend(initial, prediction, blend: float):
    value = float(np.clip(blend, 0.0, 1.0))
    return initial + value * (prediction - initial)


def select_rollout_blend(
    initial: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    feature_weights: np.ndarray,
    *,
    candidates: tuple[float, ...] = tuple(np.linspace(0.01, 1.0, 100)),
) -> dict:
    """Select a bounded residual blend on development data only."""
    initial = np.asarray(initial, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    weights = np.asarray(feature_weights, dtype=np.float32)
    if initial.shape != prediction.shape or initial.shape != target.shape:
        raise ValueError("rollout calibration arrays must have equal shapes")
    if initial.ndim != 2 or weights.shape != (initial.shape[1],):
        raise ValueError("invalid rollout calibration feature weights")
    baseline = float(np.mean(((initial - target) ** 2) * weights))
    rows = []
    for blend in candidates:
        calibrated = apply_rollout_blend(initial, prediction, blend)
        mse = float(np.mean(((calibrated - target) ** 2) * weights))
        rows.append({
            "blend": float(blend),
            "weighted_mse": mse,
            "skill_vs_persistence": 1.0 - mse / max(baseline, 1e-12),
        })
    best = min(rows, key=lambda row: (row["weighted_mse"], row["blend"]))
    return {
        "method": "bounded_residual_grid_on_grouped_dev",
        "candidates": rows,
        "selected_blend": best["blend"],
        "weighted_mse": best["weighted_mse"],
        "persistence_weighted_mse": baseline,
        "skill_vs_persistence": best["skill_vs_persistence"],
    }
