from __future__ import annotations

import numpy as np

from .macro_dynamics import soft_goal_cap


def lambdas_from_micro(
    micro_home: float,
    micro_away: float,
    status_home: float,
    status_away: float,
    config,
) -> tuple[float, float]:
    status_home = max(1.0, status_home)
    status_away = max(1.0, status_away)
    legacy_home = (status_home / 35.0) * (status_home / status_away) ** 0.5
    legacy_away = (status_away / 35.0) * (status_away / status_home) ** 0.5
    micro_home = config.lambda0 * max(0.05, micro_home) ** config.gamma_xg
    micro_away = config.lambda0 * max(0.05, micro_away) ** config.gamma_xg
    blend = config.eta_status_blend
    home = (1.0 - blend) * legacy_home + blend * micro_home
    away = (1.0 - blend) * legacy_away + blend * micro_away
    return float(max(0.08, home)), float(max(0.08, away))


def supplement_goals(
    goals_home: int,
    goals_away: int,
    micro_home: float,
    micro_away: float,
    config,
    *,
    rng: np.random.Generator,
) -> tuple[int, int, dict]:
    home = int(goals_home)
    away = int(goals_away)
    micro_home = float(micro_home) if np.isfinite(micro_home) else 0.0
    micro_away = float(micro_away) if np.isfinite(micro_away) else 0.0
    meta = {
        "physics": [home, away],
        "micro_xg": [round(micro_home, 2), round(micro_away, 2)],
        "supplemented": False,
    }
    if home == 0 and micro_home > config.xg_supplement_threshold:
        home = min(
            config.xg_supplement_cap,
            int(rng.poisson(max(0.08, micro_home * config.xg_supplement_shrink))),
        )
        meta["home_supplement"] = home
        meta["supplemented"] = True
    if away == 0 and micro_away > config.xg_supplement_threshold:
        away = min(
            config.xg_supplement_cap,
            int(rng.poisson(max(0.08, micro_away * config.xg_supplement_shrink))),
        )
        meta["away_supplement"] = away
        meta["supplemented"] = True
    meta["final"] = [home, away]
    return home, away, meta


def simulate_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    *,
    is_knockout: bool,
    rng: np.random.Generator,
) -> tuple[int, int, float, float]:
    shrink = 0.93 if is_knockout else 1.0
    lambda_home *= shrink
    lambda_away *= shrink
    if is_knockout:
        home = round(np.mean([rng.poisson(lambda_home), rng.poisson(lambda_home)]))
        away = round(np.mean([rng.poisson(lambda_away), rng.poisson(lambda_away)]))
    else:
        home = rng.poisson(lambda_home)
        away = rng.poisson(lambda_away)
    return (
        soft_goal_cap(home, lambda_home),
        soft_goal_cap(away, lambda_away),
        round(lambda_home, 2),
        round(lambda_away, 2),
    )
