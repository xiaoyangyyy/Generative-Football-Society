"""Tests for Phase 2a/2b spatial + passing."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.match_engine.formation import interpolate_anchors, mirror_for_away
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.spatial_field import SpatialFieldEngine
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.macro_bridge import build_match_affective_state
from tests.test_affective_phase1b import _FakeAgent


class TestPhase2(unittest.TestCase):
    def test_away_mirror(self):
        p = interpolate_anchors("ST", 0.5, attacks_high_x=True)
        m = mirror_for_away(p)
        self.assertAlmostEqual(p[0] + m[0], 1.0, places=4)

    def test_spatial_field_nonnegative(self):
        cfg = MicroMatchConfig.fast_demo()
        home, away = _FakeAgent("H"), _FakeAgent("A")
        st = build_match_affective_state(home, away, rng=np.random.default_rng(0))
        eng = SpatialFieldEngine(cfg)
        eng.step(st, 1.0)
        self.assertTrue(np.all(st.spatial.rho_home >= 0))

    def test_phi_lane_pass(self):
        cfg = MicroMatchConfig.fast_demo()
        home, away = _FakeAgent("Brazil"), _FakeAgent("Argentina")
        st = build_match_affective_state(home, away, rng=np.random.default_rng(1))
        sf = SpatialFieldEngine(cfg)
        sie = SpatialIntelligenceEngine(cfg)
        sf.step(st, 1.0)
        sie.step(st)
        carrier = st.home.players[5]
        recv = st.home.players[9]
        st.ball.possessor_id = carrier.player_id
        st.ball.possession_team_id = st.home.team_id
        st.ball.position = carrier.position.copy()
        lane = sie.lane_quality(st, carrier.position, recv.position, st.away)
        self.assertGreater(lane, 0)
        self.assertLessEqual(lane, 1.0)

    def test_passing_produces_attempts(self):
        cfg = MicroMatchConfig.fast_demo()
        cfg.dt_default = 30.0
        home, away = _FakeAgent("Brazil"), _FakeAgent("Argentina")
        st = build_match_affective_state(home, away, rng=np.random.default_rng(2))
        from src.match_engine.match_micro_runner import _init_micro_state

        _init_micro_state(st, cfg, np.random.default_rng(2))
        sf = SpatialFieldEngine(cfg)
        sie = SpatialIntelligenceEngine(cfg)
        pe = PassingEngine(cfg, sie)
        from src.match_engine.affective_coupling import AffectiveSpatialCoupling

        aff = AffectiveSpatialCoupling(cfg)
        rng = np.random.default_rng(2)
        for _ in range(5):
            sf.step(st, cfg.dt_default)
            sie.step(st)
            mods, _ = aff.step(st, cfg.dt_default)
            pe.step(st, mods["home"], mods["away"], rng)
        self.assertGreater(pe.stats["home_attempts"] + pe.stats["away_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
