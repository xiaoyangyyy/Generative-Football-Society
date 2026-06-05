"""Pass completion should track StatsBomb WC2022 baselines (~82% overall)."""

from __future__ import annotations

import os

import pytest

from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.pass_calibration import load_pass_calibration, target_completion


def test_calibration_file_has_wc2022_targets():
    cal = load_pass_calibration(".")
    assert cal["overall_completion"] > 0.78
    assert cal["sim_targets"]["overall"] >= 0.78


def test_target_completion_ordering():
    assert target_completion("short", 0.1) > target_completion("long", 0.4)
    assert target_completion("through", 0.25) > target_completion("long", 0.45)


@pytest.mark.slow
def test_micro_match_pass_completion_near_professional_band():
    if os.environ.get("RUN_SLOW_TESTS", "").strip() not in ("1", "true", "yes"):
        pytest.skip("Set RUN_SLOW_TESTS=1 for full micro match calibration test")
    from src.simulation.world_cup_runner import build_world_and_tournament

    engine, _, _ = build_world_and_tournament(".", require_tactics=False)
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    summary = run_match_micro_simulation(
        engine.agents["Mexico"],
        engine.agents["South Korea"],
        goals_home=0,
        goals_away=0,
        xg_home=0.6,
        xg_away=0.9,
        eff_status_home=68,
        eff_status_away=52,
        seed=99,
        config=cfg,
        writeback_agents=False,
        match_seconds=1200.0,
    )
    avg_cmp = 0.5 * (summary.pass_completion_home + summary.pass_completion_away)
    assert 0.68 <= avg_cmp <= 0.92, f"avg completion {avg_cmp:.1%} outside pro band"
