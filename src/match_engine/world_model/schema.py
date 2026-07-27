"""Stable feature and supervision schema for world-model v6."""

from __future__ import annotations

import numpy as np

GRID = slice(0, 200)
BALL = slice(200, 204)
SCORE_CLOCK_CROWD_POSSESSION = slice(204, 210)
PLAYERS = slice(210, 298)
TACTICS = slice(298, 306)
ATTACK_FLAG = slice(306, 307)

# Training-only outcome slots. They are stripped before actions reach the model.
PASS_OUTCOME_INDEX = 14
SHOT_GOAL_INDEX = 15
SHOT_ON_TARGET_INDEX = 16
LABEL_INDICES = (PASS_OUTCOME_INDEX, SHOT_GOAL_INDEX, SHOT_ON_TARGET_INDEX)
HORIZON_INDEX = 17
HORIZON_SCALE_SECONDS = 60.0


def strip_outcome_leakage(actions: np.ndarray) -> np.ndarray:
    clean = np.asarray(actions, dtype=np.float32).copy()
    if clean.ndim == 1:
        clean[list(LABEL_INDICES)] = 0.0
    else:
        clean[:, list(LABEL_INDICES)] = 0.0
    return clean


def observation_loss_weights() -> np.ndarray:
    """Prevent large low-information grids from dominating the transition loss."""
    weights = np.full(307, 0.25, dtype=np.float32)
    weights[GRID] = 0.35
    weights[BALL] = 4.0
    weights[SCORE_CLOCK_CROWD_POSSESSION] = 2.0
    weights[PLAYERS] = 1.0
    weights[TACTICS] = 1.5
    weights[ATTACK_FLAG] = 2.0
    return weights / float(weights.mean())


def observation_coverage(obs: np.ndarray) -> float:
    """Estimate whether an observation resembles a full state rather than sparse backfill."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    grid_cov = float(np.count_nonzero(np.abs(arr[GRID]) > 1e-5) / 200.0)
    player_cov = float(np.count_nonzero(np.abs(arr[PLAYERS]) > 1e-5) / 88.0)
    core_finite = float(np.isfinite(arr[BALL]).mean())
    return float(np.clip(0.45 * grid_cov + 0.45 * player_cov + 0.10 * core_finite, 0.0, 1.0))
