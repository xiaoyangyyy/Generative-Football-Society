import numpy as np

from src.memory_engine.macro_goal_dynamics import (
    integrate_match_xg,
    lambda_dot,
    simulate_match_score_from_vectors,
    team_vector_from_status,
)


def test_lambda_dot_bounded():
    x_h = team_vector_from_status(75)
    x_a = team_vector_from_status(45)
    d = lambda_dot(x_h, x_a, 1.0, 1.0)
    assert -2.0 < d < 3.0


def test_stronger_attack_higher_xg():
    rng = np.random.default_rng(0)
    strong = np.array([0.8, 0.5, 0.4, 0.6, 0.0])
    weak = np.array([0.2, 0.4, 0.3, 0.4, 0.0])
    xg_s, xg_w, _ = integrate_match_xg(strong, weak, rng=rng)
    assert xg_s > xg_w


def test_simulate_produces_scores():
    x_h = team_vector_from_status(70)
    x_a = team_vector_from_status(55)
    gh, ga, xh, xa = simulate_match_score_from_vectors(x_h, x_a, rng=np.random.default_rng(1))
    assert gh >= 0 and ga >= 0
    assert xh > 0 and xa > 0
