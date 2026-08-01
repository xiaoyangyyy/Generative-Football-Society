"""Shared probability contract for opponent-tactic inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.match_engine.tactical_catalog import TACTICAL_PRESETS, resolve_tactical_preset


TACTICAL_FEATURES = (
    "pressing_intensity",
    "risk_budget",
    "line_height",
    "rotation_aggressiveness",
)
OPPONENT_HYPOTHESES = (
    "balanced",
    "gegenpress",
    "possession_control",
    "counter_attack",
    "low_block",
    "wing_play",
    "direct_vertical",
)


def normalise_distribution(values: Any) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    clean = np.clip(clean, 0.0, None)
    total = float(clean.sum())
    if total <= 1e-12:
        return np.full(clean.shape, 1.0 / max(1, clean.size), dtype=float)
    return clean / total


def normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-12, 1.0)
    denominator = math.log(max(2, values.size))
    return float(-np.sum(values * np.log(values)) / denominator)


def tactic_feature_vector(preset_name: str) -> np.ndarray:
    preset = TACTICAL_PRESETS[resolve_tactical_preset(preset_name)]
    return np.asarray([
        float(np.clip(preset.get(feature, 0.5), 0.0, 1.0))
        for feature in TACTICAL_FEATURES
    ], dtype=float)


def tactic_likelihoods(observed: np.ndarray, *, sigma: float = 0.18) -> np.ndarray:
    centres = np.stack([
        tactic_feature_vector(name) for name in OPPONENT_HYPOTHESES
    ])
    squared = np.mean(np.square((centres - observed) / max(0.05, sigma)), axis=1)
    log_likelihood = -0.5 * squared
    log_likelihood -= float(np.max(log_likelihood))
    return np.exp(log_likelihood)


def observation_only_tactic_posterior(observed: Any) -> dict[str, float]:
    """Infer from raw controls only, excluding meta-prior and LLM feedback."""
    values = np.asarray(observed, dtype=float).reshape(-1)
    if values.size != len(TACTICAL_FEATURES) or not np.isfinite(values).all():
        raise ValueError("opponent tactical observation must contain four finite values")
    posterior = normalise_distribution(np.power(tactic_likelihoods(values), 0.65))
    return {
        name: float(posterior[index])
        for index, name in enumerate(OPPONENT_HYPOTHESES)
    }
