"""
Moving interception — p_pred(t+Δt) with inertial + pursuit toward land point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.match_engine.math_utils import sigmoid
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.state import PlayerAffectiveState


@dataclass
class InterceptResult:
    pred_recv: np.ndarray
    pred_miss: float
    opp_best_dist: float
    intercept_risk: float
    intercepted: bool
    interceptor_id: Optional[str] = None


def _pace_norm(player: PlayerAffectiveState) -> float:
    ab = getattr(player, "abilities", None)
    if ab is not None:
        return float(getattr(ab, "pace", 0.5))
    return 0.5


def pursuit_weight_defender(
    player: PlayerAffectiveState,
    land_xy: np.ndarray,
    press: float,
    lane: float,
    cfg: MicroMatchConfig,
) -> float:
    """Continuous blend inertial ↔ chase land (no discrete 'press trigger')."""
    dist_land = float(np.linalg.norm(land_xy - player.position))
    z = (
        cfg.pass_pursuit_weight_base
        + cfg.pass_pursuit_press_gain * max(0.0, press - 0.25)
        + cfg.pass_pursuit_lane_gain * max(0.0, 0.5 - lane)
        - 1.8 * dist_land
        + 0.4 * _pace_norm(player)
    )
    return float(np.clip(sigmoid(z), 0.08, 0.92))


def predict_player_xy(
    player: PlayerAffectiveState,
    dt_ahead: float,
    cfg: MicroMatchConfig,
    *,
    pursuit_target: Optional[np.ndarray] = None,
    pursuit_weight: float = 0.0,
) -> np.ndarray:
    """
    p_pred(t+Δt): inertial extrapolation blended with max-pace run toward pursuit_target.
  Substeps improve curved pursuit paths.
    """
    dt = max(0.0, float(dt_ahead)) * cfg.pass_pred_dt_scale
    if dt < 1e-6:
        return np.clip(player.position.copy(), 0.02, 0.98)

    pos = player.position.copy()
    vel = player.velocity.copy()
    scale = cfg.pass_pred_vel_scale
    n_sub = max(1, int(cfg.pass_pursuit_substeps))
    sub_dt = dt / n_sub
    pace_cap = cfg.pass_pursuit_pace_scale * (0.55 + 0.9 * _pace_norm(player))

    for _ in range(n_sub):
        inertial = pos + vel * sub_dt * scale
        if pursuit_target is not None and pursuit_weight > 0.01:
            to_tgt = pursuit_target - pos
            dist = float(np.linalg.norm(to_tgt)) + 1e-9
            step = min(dist, pace_cap * sub_dt)
            chase = pos + (to_tgt / dist) * step
            w = float(np.clip(pursuit_weight, 0.0, 1.0))
            pos = (1.0 - w) * inertial + w * chase
        else:
            pos = inertial
        pos = np.clip(pos, 0.02, 0.98)

    return pos


def evaluate_pass_intercept(
    land_xy: np.ndarray,
    time_of_flight: float,
    receiver: PlayerAffectiveState,
    opponents: List[PlayerAffectiveState],
    press: float,
    lane: float,
    cfg: MicroMatchConfig,
    rng: np.random.Generator,
    *,
    base_dir: str = ".",
) -> InterceptResult:
    tof = max(0.05, float(time_of_flight))
    pred_recv = predict_player_xy(
        receiver,
        tof,
        cfg,
        pursuit_target=land_xy,
        pursuit_weight=cfg.pass_recv_pursuit_weight,
    )
    pred_miss = float(np.linalg.norm(land_xy - pred_recv))

    best_dist = 1e9
    best_opp: Optional[PlayerAffectiveState] = None
    for opp in opponents:
        if not opp.on_pitch or opp.role == "GK":
            continue
        w_p = pursuit_weight_defender(opp, land_xy, press, lane, cfg)
        pred_o = predict_player_xy(
            opp,
            tof,
            cfg,
            pursuit_target=land_xy,
            pursuit_weight=w_p,
        )
        d = float(np.linalg.norm(land_xy - pred_o))
        if d < best_dist:
            best_dist = d
            best_opp = opp

    if best_opp is None:
        return InterceptResult(
            pred_recv=pred_recv,
            pred_miss=pred_miss,
            opp_best_dist=1e9,
            intercept_risk=0.0,
            intercepted=False,
        )

    pace = _pace_norm(best_opp)
    z = (
        cfg.pass_intercept_b0
        + cfg.pass_intercept_b_pace * pace
        - cfg.pass_intercept_b_dist * best_dist
        + cfg.pass_intercept_b_press * max(0.0, press - 0.3)
    )
    risk = float(sigmoid(z))

    recv_reach = pred_miss
    opp_reach = best_dist
    reach_edge = float(recv_reach - opp_reach)
    from src.match_engine.pass_calibration import intercept_risk_scale

    # Continuous interception chance (no hard risk threshold); scale to open-data ~2%.
    p_geom = float(
        sigmoid(
            cfg.pass_intercept_b0
            + 3.2 * reach_edge
            - 4.5 * max(0.0, opp_reach - cfg.pass_intercept_radius)
        )
    )
    p_int = float(np.clip(p_geom * risk * intercept_risk_scale(base_dir=base_dir) * 8.0, 0.0, 0.22))
    intercepted = bool(rng.random() < p_int)

    return InterceptResult(
        pred_recv=pred_recv,
        pred_miss=pred_miss,
        opp_best_dist=best_dist,
        intercept_risk=risk,
        intercepted=intercepted,
        interceptor_id=best_opp.player_id if intercepted else None,
    )
