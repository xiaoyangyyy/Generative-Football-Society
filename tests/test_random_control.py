import json

import numpy as np
import pytest

from src.memory_engine.poisson_simulator import simulate_match_score, simulate_penalty_shootout
from src.simulation.random_control import derive_seed, named_rng
from src.simulation.tournament_checkpoint import load_checkpoint, save_checkpoint


def test_named_streams_are_stable_and_isolated():
    assert derive_seed(42, "match", "A", "B") == derive_seed(42, "match", "A", "B")
    assert derive_seed(42, "match", "A", "B") != derive_seed(42, "match", "B", "A")
    assert named_rng(42, "score").random() == named_rng(42, "score").random()
    assert 0 <= derive_seed(42, "fixture") < 2**32


def test_score_and_penalties_accept_reproducible_rngs():
    first = simulate_match_score(50, 45, rng=np.random.default_rng(7))
    second = simulate_match_score(50, 45, rng=np.random.default_rng(7))
    assert first == second
    assert simulate_penalty_shootout(np.random.default_rng(9)) == simulate_penalty_shootout(
        np.random.default_rng(9)
    )


def test_lightweight_monte_carlo_is_reproducible():
    from src.app import run_monte_carlo
    assert run_monte_carlo(2, seed=91) == run_monte_carlo(2, seed=91)


def test_checkpoint_round_trip_and_version_validation(tmp_path):
    kwargs = dict(
        standings={"A": {}}, qualified_teams=[], phase="groups",
        group_schedule_progress={}, ko_round=None, ko_fixture_index=0,
        r32_fixtures=[], completed_matches=[], final_result={}, match_index=3,
    )
    save_checkpoint(str(tmp_path), **kwargs)
    assert load_checkpoint(str(tmp_path))["match_index"] == 3

    path = tmp_path / "data" / "persistence" / "tournament_checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported tournament checkpoint"):
        load_checkpoint(str(tmp_path))
