"""Shared smooth nonlinearities (no hard thresholds)."""

from __future__ import annotations

import numpy as np


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def tanh_clip(x: float | np.ndarray) -> float | np.ndarray:
    return np.tanh(np.asarray(x, dtype=float))


def softmax(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    tau = max(1e-6, float(tau))
    z = np.asarray(logits, dtype=float).reshape(-1)
    z = np.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    z = z / tau
    z = z - np.max(z)
    ex = np.exp(z)
    s = float(np.sum(ex))
    if not np.isfinite(s) or s < 1e-12:
        return np.ones_like(z) / max(1, len(z))
    return ex / s


def finite_float(x: float, default: float = 0.0) -> float:
    v = float(x)
    return default if not np.isfinite(v) else v


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))
