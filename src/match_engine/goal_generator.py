"""
Map micro xG integrals + team status → Poisson lambdas and sampled goals.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.match_engine.math_utils import finite_float

from src.match_engine.micro_config import MicroMatchConfig
from src.memory_engine.poisson_simulator import _soft_goal_cap, simulate_extra_time_score, simulate_penalty_shootout


def lambdas_from_micro(
    micro_xg_home: float,
    micro_xg_away: float,
    status_home: float,
    status_away: float,
    cfg: MicroMatchConfig,
) -> Tuple[float, float]:
    """Blend legacy status lambdas with micro integral xG."""
    sh, sa = max(1.0, status_home), max(1.0, status_away)
    lam_h_legacy = (sh / 35.0) * (sh / sa) ** 0.5
    lam_a_legacy = (sa / 35.0) * (sa / sh) ** 0.5
    lam_h_micro = cfg.lambda0 * max(0.05, micro_xg_home) ** cfg.gamma_xg
    lam_a_micro = cfg.lambda0 * max(0.05, micro_xg_away) ** cfg.gamma_xg
    eta = cfg.eta_status_blend
    lam_h = (1.0 - eta) * lam_h_legacy + eta * lam_h_micro
    lam_a = (1.0 - eta) * lam_a_legacy + eta * lam_a_micro
    return float(max(0.08, lam_h)), float(max(0.08, lam_a))


def simulate_match_score_from_micro(
    micro_xg_home: float,
    micro_xg_away: float,
    status_home: float,
    status_away: float,
    cfg: MicroMatchConfig,
    *,
    is_knockout: bool = False,
    rng: np.random.Generator | None = None,
) -> Tuple[int, int, float, float]:
    rng = rng or np.random.default_rng()
    shrink = 0.93 if is_knockout else 1.0
    lam_h, lam_a = lambdas_from_micro(micro_xg_home, micro_xg_away, status_home, status_away, cfg)
    lam_h *= shrink
    lam_a *= shrink

    if is_knockout:
        g1 = int(rng.poisson(lam_h))
        g2 = int(rng.poisson(lam_h))
        h1 = int(rng.poisson(lam_a))
        h2 = int(rng.poisson(lam_a))
        goals_h = int(round((g1 + g2) / 2.0))
        goals_a = int(round((h1 + h2) / 2.0))
    else:
        goals_h = int(rng.poisson(lam_h))
        goals_a = int(rng.poisson(lam_a))

    goals_h = _soft_goal_cap(goals_h, lam_h)
    goals_a = _soft_goal_cap(goals_a, lam_a)
    return int(goals_h), int(goals_a), round(lam_h, 2), round(lam_a, 2)


def supplement_goals_from_micro_xg(
    goals_home: int,
    goals_away: int,
    micro_xg_home: float,
    micro_xg_away: float,
    cfg: MicroMatchConfig,
    *,
    rng: np.random.Generator | None = None,
) -> Tuple[int, int, Dict[str, object]]:
    """
    Physics-first score with xG safety net: if a side scored 0 but μxG > threshold,
    sample 0..cap goals from Poisson(μxG * shrink).
    """
    rng = rng or np.random.default_rng()
    gh, ga = int(goals_home), int(goals_away)
    xg_h = finite_float(micro_xg_home, 0.0)
    xg_a = finite_float(micro_xg_away, 0.0)
    thr = float(cfg.xg_supplement_threshold)
    cap = int(cfg.xg_supplement_cap)
    shrink = float(cfg.xg_supplement_shrink)
    meta: Dict[str, object] = {
        "physics": [gh, ga],
        "micro_xg": [round(xg_h, 2), round(xg_a, 2)],
        "supplemented": False,
    }
    if gh == 0 and xg_h > thr:
        gh = min(cap, int(rng.poisson(max(0.08, xg_h * shrink))))
        meta["home_supplement"] = gh
        meta["supplemented"] = True
    if ga == 0 and xg_a > thr:
        ga = min(cap, int(rng.poisson(max(0.08, xg_a * shrink))))
        meta["away_supplement"] = ga
        meta["supplemented"] = True
    meta["final"] = [gh, ga]
    return gh, ga, meta


def simulate_knockout_extras(
    goals_h: int,
    goals_a: int,
    lam_h: float,
    lam_a: float,
    status_h: float,
    status_a: float,
    rng: np.random.Generator,
):
    """Returns (final_h, final_a, pen_h, pen_a, aet_h, aet_a, et_xg_h, et_xg_a, went_et, pens)."""
    pen_h = pen_a = None
    aet_h = aet_a = 0
    et_xg_h = et_xg_a = 0.0
    went_et = False
    if goals_h == goals_a:
        went_et = True
        aet_h, aet_a, et_xg_h, et_xg_a = simulate_extra_time_score(status_h, status_a)
        goals_h += aet_h
        goals_a += aet_a
        if goals_h == goals_a:
            pen_h, pen_a = simulate_penalty_shootout()
    return goals_h, goals_a, pen_h, pen_a, aet_h, aet_a, et_xg_h, et_xg_a, went_et, pen_h is not None
