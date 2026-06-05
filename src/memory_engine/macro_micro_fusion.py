"""Fuse macro λ dynamics xG with micro integral xG for unified scoreline."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.memory_engine.macro_goal_dynamics import clamp_match_xg, simulate_match_score_dynamics
from src.memory_engine.poisson_simulator import _soft_goal_cap
from src.simulation.score_path import physics_official_enabled, resolve_score_path_mode, ScorePathMode

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def _eta_blend() -> float:
    raw = os.environ.get("MATCH_MACRO_MICRO_ETA", "0.55")
    try:
        return float(np.clip(float(raw), 0.0, 1.0))
    except ValueError:
        return 0.55


def fuse_xg(
    xg_macro_home: float,
    xg_macro_away: float,
    xg_micro_home: float,
    xg_micro_away: float,
    eta: Optional[float] = None,
) -> Tuple[float, float]:
    eta = _eta_blend() if eta is None else float(eta)
    xh = (1.0 - eta) * xg_macro_home + eta * xg_micro_home
    xa = (1.0 - eta) * xg_macro_away + eta * xg_micro_away
    return clamp_match_xg(xh), clamp_match_xg(xa)


def macro_narrative_xg(
    home: "SocietyAgent",
    away: "SocietyAgent",
    eff_status_home: float,
    eff_status_away: float,
    *,
    is_knockout: bool = False,
    stage_pressure: float = 0.3,
    fused_volatility_home: float = 0.0,
    fused_volatility_away: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Macro λ integration → xG prior only (no goal sampling). Phase 4 narrative path."""
    rng = rng or np.random.default_rng()
    _gh, _ga, xh, xa, meta = simulate_match_score_dynamics(
        home,
        away,
        eff_status_home,
        eff_status_away,
        is_knockout=is_knockout,
        stage_pressure=stage_pressure,
        fused_volatility_home=fused_volatility_home,
        fused_volatility_away=fused_volatility_away,
        rng=rng,
    )
    meta = dict(meta)
    meta["macro_goals_sampled"] = [_gh, _ga]
    meta["narrative_only"] = True
    return float(xh), float(xa), meta


def resolve_unified_score(
    home: "SocietyAgent",
    away: "SocietyAgent",
    eff_status_home: float,
    eff_status_away: float,
    *,
    is_knockout: bool = False,
    stage_pressure: float = 0.3,
    fused_volatility_home: float = 0.0,
    fused_volatility_away: float = 0.0,
    micro_summary: Any = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int, float, float, Dict[str, Any]]:
    """
    Macro λ integration + optional micro xG fusion.

    Phase 4: when score path is physics_official and micro_summary is present,
    official goals always come from micro physics; macro never re-samples Poisson.
    """
    rng = rng or np.random.default_rng()
    path = resolve_score_path_mode()

    if path == ScorePathMode.PHYSICS_OFFICIAL and micro_summary is not None:
        xh_mi = float(getattr(micro_summary, "micro_xg_home", 0.0))
        xa_mi = float(getattr(micro_summary, "micro_xg_away", 0.0))
        gh = int(getattr(micro_summary, "goals_micro_home", 0))
        ga = int(getattr(micro_summary, "goals_micro_away", 0))
        _, xh_m, xa_m, meta_m = simulate_match_score_dynamics(
            home,
            away,
            eff_status_home,
            eff_status_away,
            is_knockout=is_knockout,
            stage_pressure=stage_pressure,
            fused_volatility_home=fused_volatility_home,
            fused_volatility_away=fused_volatility_away,
            rng=rng,
        )
        meta: Dict[str, Any] = {
            "macro": meta_m,
            "source": "physics_official",
            "score_path": path.value,
            "macro_xg": [xh_m, xa_m],
            "micro_xg": [xh_mi, xa_mi],
            "narrative_only_macro": True,
        }
        xh, xa = xh_mi, xa_mi
        meta["xg_final"] = [round(xh, 2), round(xa, 2)]
        meta["goals"] = [gh, ga]
        return gh, ga, round(xh, 2), round(xa, 2), meta

    gh, ga, xh_m, xa_m, meta_m = simulate_match_score_dynamics(
        home,
        away,
        eff_status_home,
        eff_status_away,
        is_knockout=is_knockout,
        stage_pressure=stage_pressure,
        fused_volatility_home=fused_volatility_home,
        fused_volatility_away=fused_volatility_away,
        rng=rng,
    )
    meta: Dict[str, Any] = {"macro": meta_m, "source": "macro_dynamics"}
    xh, xa = xh_m, xa_m

    if micro_summary is not None:
        xh_mi = float(getattr(micro_summary, "micro_xg_home", xh_m))
        xa_mi = float(getattr(micro_summary, "micro_xg_away", xa_m))
        eta = _eta_blend()
        xh, xa = fuse_xg(xh_m, xa_m, xh_mi, xa_mi, eta=eta)
        meta["micro_xg"] = [xh_mi, xa_mi]
        meta["eta"] = eta
        meta["source"] = "macro_micro_fusion"
        if physics_official_enabled() and os.environ.get("MATCH_MICRO_SCORE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            gh = int(getattr(micro_summary, "goals_micro_home", gh))
            ga = int(getattr(micro_summary, "goals_micro_away", ga))
            meta["source"] = "physics_official"
            meta["score_path"] = ScorePathMode.PHYSICS_OFFICIAL.value
        else:
            if is_knockout:
                g1h, g2h = int(rng.poisson(max(0.05, xh))), int(rng.poisson(max(0.05, xh)))
                g1a, g2a = int(rng.poisson(max(0.05, xa))), int(rng.poisson(max(0.05, xa)))
                gh, ga = int(round((g1h + g2h) / 2)), int(round((g1a + g2a) / 2))
            else:
                gh = int(rng.poisson(max(0.05, xh)))
                ga = int(rng.poisson(max(0.05, xa)))
            gh = _soft_goal_cap(gh, xh)
            ga = _soft_goal_cap(ga, xa)

    meta["xg_final"] = [round(xh, 2), round(xa, 2)]
    meta["goals"] = [gh, ga]
    return gh, ga, round(xh, 2), round(xa, 2), meta
