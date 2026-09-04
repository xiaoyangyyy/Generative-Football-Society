"""Fuse macro λ dynamics xG with micro integral xG for unified scoreline."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.memory_engine.macro_goal_dynamics import (
    clamp_match_xg,
    expected_match_xg,
    sample_goals_from_xg,
)
from src.simulation.score_path import resolve_score_path_mode, ScorePathMode
from src.simulation.runtime import environment_snapshot, env_float

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def _eta_blend() -> float:
    return float(np.clip(env_float(environment_snapshot(), "MATCH_MACRO_MICRO_ETA", 0.55), 0.0, 1.0))


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
    xh, xa, meta = expected_match_xg(
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
        xh_m, xa_m, meta_m = expected_match_xg(
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

    xh_m, xa_m, meta_m = expected_match_xg(
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
    gh, ga = sample_goals_from_xg(xh, xa, rng=rng, is_knockout=is_knockout)

    meta["xg_final"] = [round(xh, 2), round(xa, 2)]
    meta["goals"] = [gh, ga]
    return gh, ga, round(xh, 2), round(xa, 2), meta
