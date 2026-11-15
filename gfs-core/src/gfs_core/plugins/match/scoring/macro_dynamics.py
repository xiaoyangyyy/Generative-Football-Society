from __future__ import annotations

import math

import numpy as np

MATCH_MINUTES = 90.0
XG_PER_TEAM_MIN = 0.28
XG_PER_TEAM_MAX = 3.6
LAM_MIN_PER_MINUTE = 0.004
LAM_MAX_PER_MINUTE = 0.065


def team_vector_from_status(status: float) -> np.ndarray:
    value = np.tanh((float(status) - 35.0) / 18.0)
    return np.array([
        value,
        value * 0.92,
        value * 0.55,
        value * 0.65,
        np.tanh((50.0 - status) / 25.0),
    ], dtype=float)


def lambda_dot(
    own: np.ndarray,
    opponent: np.ndarray,
    own_lambda: float,
    opponent_lambda: float,
    *,
    is_knockout: bool = False,
    stage_pressure: float = 0.0,
) -> float:
    attack_defense = float(own[0] - opponent[1])
    morale_gap = float(own[3] - opponent[3])
    tempo = float(0.5 * (own[2] + opponent[2]))
    pressure_load = float(own[4])
    drive = (
        0.011 * _softplus(1.35 * attack_defense)
        + 0.0055 * _softplus(0.9 * morale_gap)
        + 0.0035 * _softplus(0.7 * tempo)
        + 0.0025
    )
    decay = (
        2.85 * float(own_lambda)
        + 0.04 * float(opponent_lambda) * _sigmoid(attack_defense)
    )
    knockout_drag = 0.0035 if is_knockout else 0.0
    stage_drag = (
        0.0025 * np.tanh(stage_pressure) * _sigmoid(pressure_load)
    )
    return float(np.clip(drive - decay - knockout_drag - stage_drag, -1.95, 2.95))


def integrate_match_xg(
    home: np.ndarray,
    away: np.ndarray,
    *,
    minutes: float = MATCH_MINUTES,
    steps: int = 90,
    is_knockout: bool = False,
    stage_pressure: float = 0.0,
    rng: np.random.Generator,
) -> tuple[float, float]:
    dt = float(minutes) / max(1, int(steps))
    lambda_home = _clamp_lambda(0.011 + 0.0055 * float(home[0]))
    lambda_away = _clamp_lambda(0.011 + 0.0055 * float(away[0]))
    xg_home = 0.0
    xg_away = 0.0
    for _ in range(steps):
        change_home = lambda_dot(
            home, away, lambda_home, lambda_away,
            is_knockout=is_knockout,
            stage_pressure=stage_pressure,
        )
        change_away = lambda_dot(
            away, home, lambda_away, lambda_home,
            is_knockout=is_knockout,
            stage_pressure=stage_pressure,
        )
        lambda_home = _clamp_lambda(
            lambda_home + dt * (change_home + rng.normal(0, 0.0012))
        )
        lambda_away = _clamp_lambda(
            lambda_away + dt * (change_away + rng.normal(0, 0.0012))
        )
        xg_home += lambda_home * dt
        xg_away += lambda_away * dt
    return _finite_xg(xg_home), _finite_xg(xg_away)


def sample_goals(
    home_xg: float,
    away_xg: float,
    *,
    is_knockout: bool,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if is_knockout:
        home = round(np.mean([
            rng.poisson(max(0.05, home_xg)),
            rng.poisson(max(0.05, home_xg)),
        ]))
        away = round(np.mean([
            rng.poisson(max(0.05, away_xg)),
            rng.poisson(max(0.05, away_xg)),
        ]))
    else:
        home = rng.poisson(max(0.05, home_xg))
        away = rng.poisson(max(0.05, away_xg))
    return soft_goal_cap(home, home_xg), soft_goal_cap(away, away_xg)


def soft_goal_cap(goals: int, expected_goals: float) -> int:
    expected_goals = float(max(0.05, expected_goals))
    cap = int(max(
        5,
        np.ceil(expected_goals + 3.2 * np.sqrt(expected_goals + 0.15) + 1.0),
    ))
    return min(int(max(0, goals)), cap)


def _softplus(value: float) -> float:
    return value if value > 20.0 else math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -20.0, 20.0)))))


def _clamp_lambda(value: float) -> float:
    return float(np.clip(value, LAM_MIN_PER_MINUTE, LAM_MAX_PER_MINUTE))


def _finite_xg(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, XG_PER_TEAM_MIN, XG_PER_TEAM_MAX))
