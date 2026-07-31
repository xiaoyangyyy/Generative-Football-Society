"""The coach policy bridge reaches normal action sampling without forcing it."""

from __future__ import annotations

import numpy as np

from src.match_engine.action_engine import ActionEngine
from src.match_engine.aerial_duel import AerialDuelEngine
from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.match_micro_runner import _init_micro_state
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.state import PlayerModulators
from src.match_engine.world_model.decision_adoption import (
    register_coach_action_decision,
)
from tests.test_affective_phase1b import _FakeAgent


def test_action_engine_consumes_bridge_and_audits_actual_sample():
    cfg = MicroMatchConfig.fast_demo()
    cfg.enable_wall_pass = False
    rng = np.random.default_rng(17)
    state = build_match_affective_state(
        _FakeAgent("Home"), _FakeAgent("Away"), rng=rng,
    )
    _init_micro_state(state, cfg, rng)
    state.clock_seconds = 10.0
    record = register_coach_action_decision(
        state,
        team_id=state.home.team_id,
        trigger_kind="xg_swing",
        llm_selected_action="hold",
        world_model_recommended_action="hold",
        recommendation_confidence=0.8,
        horizon_s=10.0,
        intervention_enabled=True,
        intervention_strength=0.35,
    )
    state.clock_seconds = 11.0
    spatial = SpatialIntelligenceEngine(cfg)
    passing = PassingEngine(cfg, spatial)
    engine = ActionEngine(
        cfg,
        passing,
        ShotEngine(cfg, spatial),
        AerialDuelEngine(cfg),
    )
    mod_home = [PlayerModulators(player_id=p.player_id) for p in state.home.players]
    mod_away = [PlayerModulators(player_id=p.player_id) for p in state.away.players]

    actual_action, _ = engine.step(state, mod_home, mod_away, rng)

    assert record["intervention_applied"] is True
    assert record["intervention_actual_action"] == actual_action
    assert record["adopted"] is (actual_action == "hold")
    assert record["resolution"] in {
        "matching_action_after_bounded_bias",
        "different_action_after_bounded_bias",
    }
