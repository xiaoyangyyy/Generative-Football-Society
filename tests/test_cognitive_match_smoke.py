"""Smoke: full micro match with cognitive enabled (rule fallback, no API)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["MATCH_COGNITIVE"] = "1"
os.environ["MATCH_COGNITIVE_CACHE"] = ""


def test_micro_match_with_cognitive_plans():
    from src.simulation.world_cup_runner import build_world_and_tournament
    from src.simulation.match_pipeline import prepare_match_agents, run_micro_layer
    from src.memory_engine.macro_goal_dynamics import simulate_match_score_dynamics

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    engine, _, _ = build_world_and_tournament(base, load_coaches=False)
    teams = list(engine.agents.keys())
    h, a = teams[0], teams[1]
    ah, aa = engine.agents[h], engine.agents[a]
    prepare_match_agents(ah, aa, base)
    ph, pa, xh, xa, _ = simulate_match_score_dynamics(
        ah, aa, ah.status_score, aa.status_score, is_knockout=False, stage_pressure=0.3
    )
    summary = run_micro_layer(
        ah,
        aa,
        goals_home=ph,
        goals_away=pa,
        xg_home=xh,
        xg_away=xa,
        referee={"strictness": 0.5},
        stage_pressure=0.3,
        drama_score=0.4,
        internal_home={},
        internal_away={},
        seed=99,
    )
    assert summary.ticks > 0
    assert isinstance(summary.cognitive_plans, list)
    assert len(summary.cognitive_triggers) >= 0
