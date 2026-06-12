"""
Macro goal-rate dynamics: λ̇ = f(x_T, x_opp, λ, λ_opp).

x_T ∈ R^5 from team_dynamics (attack, defense, press, morale_field, institutional_pressure).
Match xG = ∫ λ(t) dt over regulation; goals ~ Poisson(xG) with soft tail cap.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _tanh_clip(x: float) -> float:
    return float(math.tanh(float(x)))

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent

TEAM_STATE_KEYS: Tuple[str, ...] = (
    "attack",
    "defense",
    "press",
    "morale_field",
    "institutional_pressure",
)

# Regulation match horizon (minutes) for ∫λ dt
MATCH_MINUTES = 90.0
EXTRA_TIME_MINUTES = 30.0

# λ is goals-per-minute intensity; ∫λ dt over 90' → regulation xG (typical 0.8–2.6)
XG_PER_TEAM_MIN = 0.28
XG_PER_TEAM_MAX = 3.6
LAM_MIN_PER_MINUTE = 0.004
LAM_MAX_PER_MINUTE = 0.065


def softplus(x: float) -> float:
    x = float(x)
    if x > 20.0:
        return x
    return float(math.log1p(math.exp(x)))


def clamp_match_xg(xg: float) -> float:
    return float(np.clip(float(xg), XG_PER_TEAM_MIN, XG_PER_TEAM_MAX))


def clamp_lambda_rate(lam: float) -> float:
    return float(np.clip(float(lam), LAM_MIN_PER_MINUTE, LAM_MAX_PER_MINUTE))


def team_vector_from_status(status: float) -> np.ndarray:
    """Fallback x_T when only historical status_score is available."""
    s = _tanh_clip((float(status) - 35.0) / 18.0)
    return np.array([s, s * 0.92, s * 0.55, s * 0.65, _tanh_clip((50.0 - status) / 25.0)], dtype=float)


def team_vector_from_agent(
    agent: Optional["SocietyAgent"],
    eff_status: float,
    *,
    fused_volatility: float = 0.0,
) -> np.ndarray:
    """
    Build x_T from roster team_dynamics, perturbed by effective match status (fusion/tactics).
    No hard clips — bounded via tanh.
    """
    base = getattr(agent, "team_dynamics", None) if agent is not None else None
    if base and isinstance(base, dict):
        x = np.array([float(base.get(k, 0.0)) for k in TEAM_STATE_KEYS], dtype=float)
    else:
        raw_status = float(getattr(agent, "status_score", eff_status) if agent is not None else eff_status)
        x = team_vector_from_status(eff_status if eff_status > 0 else raw_status)

    anchor = float(getattr(agent, "status_score", eff_status) if agent is not None else eff_status)
    delta = _tanh_clip((float(eff_status) - anchor) / 12.0)
    x = x + np.array([0.42, 0.18, 0.22, 0.35, 0.0], dtype=float) * delta
    x = x + np.array([0.0, 0.0, 0.12, 0.08, 0.0], dtype=float) * _tanh_clip(fused_volatility)
    from src.match_engine.math_utils import finite_float

    return np.array([finite_float(_tanh_clip(v), 0.0) for v in x], dtype=float)


def lambda_dot(
    x_self: np.ndarray,
    x_opp: np.ndarray,
    lam_self: float,
    lam_opp: float,
    *,
    is_knockout: bool = False,
    stage_pressure: float = 0.0,
) -> float:
    """
    λ̇ = drive(x_T, x_opp) - decay(λ) - opp_def coupling.

    drive: attack advantage vs opponent defense, morale gap, shared press tempo.
    """
    x_self = np.asarray(x_self, dtype=float).reshape(-1)
    x_opp = np.asarray(x_opp, dtype=float).reshape(-1)
    att_def = float(x_self[0] - x_opp[1])
    morale_gap = float(x_self[3] - x_opp[3])
    tempo = float(0.5 * (x_self[2] + x_opp[2]))
    pressure_load = float(x_self[4])

    # Per-minute scoring intensity (not goals-per-90); steady λ ≈ 0.012–0.028
    drive = (
        0.011 * softplus(1.35 * att_def)
        + 0.0055 * softplus(0.9 * morale_gap)
        + 0.0035 * softplus(0.7 * tempo)
        + 0.0025
    )
    decay = 2.85 * float(lam_self) + 0.04 * float(lam_opp) * _sigmoid(att_def)
    ko_drag = 0.0035 if is_knockout else 0.0
    stage_drag = 0.0025 * _tanh_clip(stage_pressure) * _sigmoid(pressure_load)
    return float(drive - decay - ko_drag - stage_drag)


def integrate_match_xg(
    x_home: np.ndarray,
    x_away: np.ndarray,
    *,
    minutes: float = MATCH_MINUTES,
    n_steps: int = 90,
    is_knockout: bool = False,
    stage_pressure: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Integrate coupled λ dynamics for both teams; return xG = ∫λ dt.
    """
    rng = rng or np.random.default_rng()
    dt = float(minutes) / max(1, int(n_steps))
    lam_h = clamp_lambda_rate(0.011 + 0.0055 * float(x_home[0]))
    lam_a = clamp_lambda_rate(0.011 + 0.0055 * float(x_away[0]))
    acc_h = 0.0
    acc_a = 0.0
    noise_scale = 0.0012

    for _ in range(n_steps):
        dh = lambda_dot(
            x_home, x_away, lam_h, lam_a, is_knockout=is_knockout, stage_pressure=stage_pressure
        )
        da = lambda_dot(
            x_away, x_home, lam_a, lam_h, is_knockout=is_knockout, stage_pressure=stage_pressure
        )
        lam_h = clamp_lambda_rate(lam_h + dt * (dh + rng.normal(0, noise_scale)))
        lam_a = clamp_lambda_rate(lam_a + dt * (da + rng.normal(0, noise_scale)))
        acc_h += lam_h * dt
        acc_a += lam_a * dt

    xg_h = clamp_match_xg(acc_h)
    xg_a = clamp_match_xg(acc_a)
    from src.match_engine.math_utils import finite_float

    xg_h = finite_float(xg_h, 1.0)
    xg_a = finite_float(xg_a, 1.0)
    meta = {
        "model": "macro_goal_dynamics_v2_calibrated",
        "lambda_end_home": lam_h,
        "lambda_end_away": lam_a,
        "raw_integral_home": float(acc_h),
        "raw_integral_away": float(acc_a),
        "minutes": minutes,
        "n_steps": n_steps,
    }
    return xg_h, xg_a, meta


def sample_goals_from_xg(
    xg_home: float,
    xg_away: float,
    rng: Optional[np.random.Generator] = None,
    *,
    is_knockout: bool = False,
) -> Tuple[int, int]:
    """Poisson observation noise on integrated xG (not status lookup)."""
    from src.memory_engine.poisson_simulator import _soft_goal_cap

    rng = rng or np.random.default_rng()
    if is_knockout:
        g1 = int(rng.poisson(max(0.05, xg_home)))
        g2 = int(rng.poisson(max(0.05, xg_home)))
        h1 = int(rng.poisson(max(0.05, xg_away)))
        h2 = int(rng.poisson(max(0.05, xg_away)))
        gh = int(round((g1 + g2) / 2.0))
        ga = int(round((h1 + h2) / 2.0))
    else:
        gh = int(rng.poisson(max(0.05, xg_home)))
        ga = int(rng.poisson(max(0.05, xg_away)))
    gh = _soft_goal_cap(gh, xg_home)
    ga = _soft_goal_cap(ga, xg_away)
    return gh, ga


def expected_match_xg(
    agent_home: Optional["SocietyAgent"],
    agent_away: Optional["SocietyAgent"],
    eff_status_home: float,
    eff_status_away: float,
    *,
    is_knockout: bool = False,
    stage_pressure: float = 0.0,
    fused_volatility_home: float = 0.0,
    fused_volatility_away: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Team-dynamics xG prior for micro event density only — does not sample goals."""
    x_h = team_vector_from_agent(
        agent_home, eff_status_home, fused_volatility=fused_volatility_home
    )
    x_a = team_vector_from_agent(
        agent_away, eff_status_away, fused_volatility=fused_volatility_away
    )
    xg_h, xg_a, meta = integrate_match_xg(
        x_h,
        x_a,
        is_knockout=is_knockout,
        stage_pressure=stage_pressure,
        rng=rng,
    )
    meta["x_home"] = x_h.tolist()
    meta["x_away"] = x_a.tolist()
    return round(xg_h, 2), round(xg_a, 2), meta


def simulate_match_score_dynamics(
    agent_home: Optional["SocietyAgent"],
    agent_away: Optional["SocietyAgent"],
    eff_status_home: float,
    eff_status_away: float,
    *,
    is_knockout: bool = False,
    stage_pressure: float = 0.0,
    fused_volatility_home: float = 0.0,
    fused_volatility_away: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int, float, float, Dict[str, Any]]:
    xg_h, xg_a, meta = expected_match_xg(
        agent_home,
        agent_away,
        eff_status_home,
        eff_status_away,
        is_knockout=is_knockout,
        stage_pressure=stage_pressure,
        fused_volatility_home=fused_volatility_home,
        fused_volatility_away=fused_volatility_away,
        rng=rng,
    )
    gh, ga = sample_goals_from_xg(xg_h, xg_a, rng=rng, is_knockout=is_knockout)
    return gh, ga, xg_h, xg_a, meta


def simulate_extra_time_dynamics(
    agent_home: Optional["SocietyAgent"],
    agent_away: Optional["SocietyAgent"],
    eff_status_home: float,
    eff_status_away: float,
    *,
    stage_pressure: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int, float, float]:
    x_h = team_vector_from_agent(agent_home, eff_status_home)
    x_a = team_vector_from_agent(agent_away, eff_status_away)
    xg_h, xg_a, _ = integrate_match_xg(
        x_h,
        x_a,
        minutes=EXTRA_TIME_MINUTES,
        n_steps=30,
        is_knockout=True,
        stage_pressure=stage_pressure,
        rng=rng,
    )
    gh, ga = sample_goals_from_xg(xg_h, xg_a, rng=rng, is_knockout=True)
    return gh, ga, round(xg_h, 2), round(xg_a, 2)


def simulate_match_score_from_vectors(
    x_home: np.ndarray,
    x_away: np.ndarray,
    *,
    is_knockout: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int, float, float]:
    """API for stats-only simulators (no SocietyAgent)."""
    xg_h, xg_a, _ = integrate_match_xg(x_home, x_away, is_knockout=is_knockout, rng=rng)
    gh, ga = sample_goals_from_xg(xg_h, xg_a, rng=rng, is_knockout=is_knockout)
    return gh, ga, round(xg_h, 2), round(xg_a, 2)
