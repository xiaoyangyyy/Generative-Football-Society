import numpy as np
import pytest

from gfs_core.plugins.match.scoring.macro_dynamics import integrate_match_xg
from gfs_core.plugins.match.scoring.service import (
    MacroScoreService,
    match_probabilities,
    simulate_penalty_shootout,
)


def test_stronger_team_produces_higher_expected_goals():
    strong = np.array([0.8, 0.5, 0.4, 0.6, 0.0])
    weak = np.array([0.2, 0.4, 0.3, 0.4, 0.0])
    strong_xg, weak_xg = integrate_match_xg(
        strong, weak, rng=np.random.default_rng(0),
    )
    assert strong_xg > weak_xg


def test_score_service_is_deterministic_and_bounded():
    service = MacroScoreService()
    first = service.simulate(
        70, 55, is_knockout=False, rng=np.random.default_rng(1),
    )
    second = service.simulate(
        70, 55, is_knockout=False, rng=np.random.default_rng(1),
    )
    assert first == second
    assert first.home_goals >= 0
    assert first.away_goals >= 0
    assert first.home_xg > 0
    assert first.away_xg > 0


def test_penalties_terminate_and_probabilities_sum_to_one_hundred():
    home, away = simulate_penalty_shootout(np.random.default_rng(42))
    assert home != away
    probabilities = match_probabilities(70, 55)
    assert sum(probabilities.values()) == pytest.approx(100.0, abs=0.11)
    assert probabilities["p_win_a"] > probabilities["p_win_b"]
