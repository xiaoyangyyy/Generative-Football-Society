"""Continuous salience scoring for cognitive triggers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import (
    ENTITY_TIER_ASSISTANT,
    ENTITY_TIER_COACH,
    ENTITY_TIER_CROWD,
    ENTITY_TIER_PLAYER,
    ENTITY_TIER_REFEREE,
)
from src.match_engine.math_utils import sigmoid, tanh_clip
from src.match_engine.micro_events import MicroEvent, MicroEventType

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState

# Map micro events → cognitive kind weights
_KIND_WEIGHT: dict = {
    MicroEventType.GOAL_SCORED: 2.4,
    MicroEventType.GOAL_CONCEDED: 1.6,
    MicroEventType.RED_CARD: 2.2,
    MicroEventType.YELLOW_CARD: 1.0,
    MicroEventType.VAR_CONTROVERSY: 2.0,
    MicroEventType.ICON_PROTEST: 1.4,
    MicroEventType.CROWD_WAVE: 1.2,
    MicroEventType.FOUL_COMMITTED: 0.7,
    MicroEventType.SAVE: 0.9,
}

_TIER_S0_OFFSET = {
    ENTITY_TIER_COACH: 0.0,
    ENTITY_TIER_REFEREE: 0.05,
    ENTITY_TIER_PLAYER: 0.10,
    ENTITY_TIER_ASSISTANT: 0.18,
    ENTITY_TIER_CROWD: 0.12,
}


def base_salience_from_micro(ev: MicroEvent, state: "MatchAffectiveState") -> float:
    w = float(_KIND_WEIGHT.get(ev.event_type, 0.4))
    inten = float(max(0.05, ev.intensity))
    score_term = 0.35 * tanh_clip(abs(state.home.score - state.away.score))
    xg_term = 0.25 * tanh_clip(abs(state.micro_xg_home - state.micro_xg_away))
    psi_term = 0.2 * tanh_clip(abs(state.crowd.psi))
    return float(w * inten + score_term + xg_term + psi_term)


def half_time_salience(t_sec: float, cfg: CognitiveMatchConfig) -> float:
    d1 = abs(t_sec - cfg.half_time_sec)
    d2 = abs(t_sec - cfg.second_half_sec)
    d = min(d1, d2)
    if d > cfg.half_time_window_sec:
        return 0.0
    return float(1.35 * (1.0 - d / cfg.half_time_window_sec))


def crowd_surge_salience(state: "MatchAffectiveState", prev_psi: float) -> float:
    dpsi = abs(state.crowd.psi - prev_psi)
    return float(1.8 * tanh_clip(dpsi * 4.0) + 0.3 * tanh_clip(state.referee.controversy_integral))


def trigger_probability(
    salience: float,
    entity_tier: str,
    cfg: CognitiveMatchConfig,
) -> float:
    s0 = cfg.salience_center + _TIER_S0_OFFSET.get(entity_tier, 0.0)
    return float(sigmoid(3.2 * (salience - s0)))


def should_fire(
    salience: float,
    entity_id: str,
    entity_tier: str,
    t_sec: float,
    last_fire: dict,
    cfg: CognitiveMatchConfig,
    rng: np.random.Generator,
) -> bool:
    p = trigger_probability(salience, entity_tier, cfg)
    last = last_fire.get(entity_id, -1e9)
    if t_sec - last < cfg.cooldown_sec:
        return False
    return bool(rng.random() < p)
