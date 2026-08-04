"""Schema-grounded context interventions shared by shadow model probes."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_contract import tactic_feature_vector


CONTEXT_FACTORS = (
    "score_context",
    "match_phase",
    "own_tactics",
    "opponent_tactics",
    "crowd_context",
)


def neutralize_context(
    observation: Any, *, factor: str, attacking_home: bool,
) -> tuple[np.ndarray, list[int], float]:
    neutral = np.asarray(observation, dtype=np.float32).copy()
    source = np.asarray(observation, dtype=np.float32)
    if factor == "score_context":
        indices = [204, 205, 206]
        neutral[indices] = [0.0, 0.0, 0.5]
    elif factor == "match_phase":
        indices = [207]
        neutral[207] = 0.5
    elif factor == "crowd_context":
        indices = [208]
        neutral[208] = 0.5
    elif factor in {"own_tactics", "opponent_tactics"}:
        own_start = 298 if attacking_home else 302
        opponent_start = 302 if attacking_home else 298
        start = own_start if factor == "own_tactics" else opponent_start
        indices = list(range(start, start + 4))
        neutral[indices] = tactic_feature_vector("balanced")
    else:
        raise ValueError("unsupported contrastive context factor")
    magnitude = float(np.max(np.abs(neutral[indices] - source[indices])))
    return neutral, indices, magnitude
