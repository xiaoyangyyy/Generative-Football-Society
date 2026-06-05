"""Wall pass / give-and-go — two-touch ODE combo in one decision step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.match_engine.math_utils import sigmoid
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.state import PlayerAffectiveState


@dataclass
class WallPartner:
    player: PlayerAffectiveState
    distance: float


def find_wall_partner(
    carrier: PlayerAffectiveState,
    teammates: List[PlayerAffectiveState],
    cfg: MicroMatchConfig,
) -> Optional[WallPartner]:
    best: Optional[WallPartner] = None
    for p in teammates:
        if not p.on_pitch or p.player_id == carrier.player_id or p.role == "GK":
            continue
        d = float(np.linalg.norm(p.position - carrier.position))
        if d > cfg.wall_radius:
            continue
        if best is None or d < best.distance:
            best = WallPartner(player=p, distance=d)
    return best


def wall_return_target(
    carrier: PlayerAffectiveState,
    attacking_high_x: bool,
    risk_budget: float,
    cfg: MicroMatchConfig,
) -> np.ndarray:
    """Sprint lane after lay-off — continuous forward offset."""
    fwd = cfg.wall_return_forward * (0.75 + 0.5 * risk_budget)
    if attacking_high_x:
        tgt = carrier.position + np.array([fwd, 0.0])
    else:
        tgt = carrier.position + np.array([-fwd, 0.0])
    return np.clip(tgt, 0.03, 0.97)


def wall_pass_utility(
    carrier: PlayerAffectiveState,
    wall: PlayerAffectiveState,
    press: float,
    lane: float,
    cfg: MicroMatchConfig,
) -> float:
    dist = float(np.linalg.norm(wall.position - carrier.position))
    z = (
        cfg.wall_u_base
        + cfg.wall_u_press * press
        + 0.35 * float(carrier.abilities.vision)
        + 0.25 * float(wall.abilities.pass_skill)
        + 0.2 * lane
        - 2.5 * dist
    )
    return float(sigmoid(z))
