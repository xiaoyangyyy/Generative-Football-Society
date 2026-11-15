from __future__ import annotations

import numpy as np


def sigmoid(value):
    values = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20, 20)))


def tanh_clip(value):
    return np.tanh(np.asarray(value, dtype=float))


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def finite_float(value: float, default: float = 0.0) -> float:
    value = float(value)
    return default if not np.isfinite(value) else value


def latent_logit(value: float, epsilon: float = 1e-4) -> float:
    value = min(1.0 - epsilon, max(epsilon, float(value)))
    return float(np.log(value / (1.0 - value)))


def softmax(logits, tau: float = 1.0):
    tau = max(1e-6, float(tau))
    values = np.asarray(logits, dtype=float).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
    values = values / tau
    values = values - np.max(values)
    exponentials = np.exp(values)
    total = float(np.sum(exponentials))
    if not np.isfinite(total) or total < 1e-12:
        return np.ones_like(values) / max(1, len(values))
    return exponentials / total
