from __future__ import annotations

import math

import numpy as np

from gfs_core.contracts.match import MatchScore
from .macro_dynamics import (
    integrate_match_xg,
    sample_goals,
    soft_goal_cap,
    team_vector_from_status,
)
from .micro import lambdas_from_micro, simulate_from_lambdas, supplement_goals


class MacroScoreService:
    def simulate(
        self,
        status_home: float,
        status_away: float,
        *,
        is_knockout: bool,
        rng: np.random.Generator,
    ) -> MatchScore:
        home = team_vector_from_status(status_home)
        away = team_vector_from_status(status_away)
        home_xg, away_xg = integrate_match_xg(
            home, away, is_knockout=is_knockout, rng=rng,
        )
        home_goals, away_goals = sample_goals(
            home_xg, away_xg, is_knockout=is_knockout, rng=rng,
        )
        return MatchScore(
            int(home_goals),
            int(away_goals),
            round(home_xg, 2),
            round(away_xg, 2),
        )

    @staticmethod
    def penalties(rng: np.random.Generator) -> tuple[int, int]:
        return simulate_penalty_shootout(rng)

    lambdas_from_micro = staticmethod(lambdas_from_micro)
    supplement_goals = staticmethod(supplement_goals)
    simulate_from_lambdas = staticmethod(simulate_from_lambdas)

    def simulate_legacy(
        self,
        status_home: float,
        status_away: float,
        *,
        is_knockout: bool,
        rng: np.random.Generator,
    ) -> MatchScore:
        shrink = 0.93 if is_knockout else 1.0
        home_lambda = (
            status_home / 35.0
            * (status_home / max(1.0, status_away)) ** 0.5
            * shrink
        )
        away_lambda = (
            status_away / 35.0
            * (status_away / max(1.0, status_home)) ** 0.5
            * shrink
        )
        if is_knockout:
            home_goals = round(np.mean([
                rng.poisson(home_lambda), rng.poisson(home_lambda),
            ]))
            away_goals = round(np.mean([
                rng.poisson(away_lambda), rng.poisson(away_lambda),
            ]))
        else:
            home_goals = rng.poisson(home_lambda)
            away_goals = rng.poisson(away_lambda)
        return MatchScore(
            soft_goal_cap(home_goals, home_lambda),
            soft_goal_cap(away_goals, away_lambda),
            round(float(home_lambda), 2),
            round(float(away_lambda), 2),
        )


def simulate_penalty_shootout(rng: np.random.Generator) -> tuple[int, int]:
    home = 0
    away = 0
    for _ in range(5):
        if rng.random() > 0.25:
            home += 1
        if rng.random() > 0.25:
            away += 1
    sudden_round = 0
    while home == away:
        sudden_round += 1
        tail_penalty = min(0.20, 0.02 * max(0, sudden_round - 2))
        conversion = 0.75 - tail_penalty
        if rng.random() < conversion:
            home += 1
        if rng.random() < conversion:
            away += 1
        if sudden_round >= 8 and home == away:
            if rng.random() < 0.5:
                home += 1
            else:
                away += 1
    return home, away


def match_probabilities(status_home: float, status_away: float) -> dict[str, float]:
    difference = status_home - status_away
    home_base = 1 / (1 + math.pow(10, -difference / 80.0))
    away_base = 1 - home_base
    draw = 0.33 * math.exp(-(difference**2) / (2 * (40**2)))
    home = home_base * (1 - draw)
    away = away_base * (1 - draw)
    total = home + away + draw
    return {
        "p_win_a": round(home / total * 100, 1),
        "p_draw": round(draw / total * 100, 1),
        "p_win_b": round(away / total * 100, 1),
    }
