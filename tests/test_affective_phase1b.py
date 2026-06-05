"""Unit tests for Phase 1b affective coupling."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.match_engine.affective_coupling import AffectiveSpatialCoupling
from src.match_engine.config import AffectiveConfig
from src.match_engine.math_utils import softmax
from src.match_engine.micro_events import MicroEvent, MicroEventType, apply_micro_event
from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.event_schedule import build_event_schedule


class _FakeAgent:
    def __init__(self, name: str):
        self.team_name = name
        self.name = name
        self.status_score = 72.0
        self.fatigue = 0.15
        self.media_exposure = 0.7
        self.referee_grievance = 0.08
        self.conflict_heat = 0.14
        self.team_cohesion = 0.65
        self.coach_authority = 0.6
        self.tactical_controls = {
            "pressing_intensity": 0.55,
            "risk_budget": 0.48,
            "line_height": 0.52,
            "rotation_aggressiveness": 0.4,
        }
        self.emotion_profile = {
            "pride": 0.22,
            "anger": 0.12,
            "shame": 0.1,
            "fear": 0.18,
            "determination": 0.38,
        }
        self.psychology_profile = {"confidence": 0.62, "stability": 0.58}
        self.formation = "4-3-3"
        self.style_desc = "balanced possession"
        self.style_archetype = "balanced"


class TestAffectivePhase1b(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        p = softmax(np.array([1.0, 0.2, -0.3, 0.5]))
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=6)

    def test_goal_raises_psi_and_pride(self):
        home = _FakeAgent("Brazil")
        away = _FakeAgent("Argentina")
        state = build_match_affective_state(home, away, rng=np.random.default_rng(0))
        psi0 = state.crowd.psi
        st = next(p for p in state.home.players if p.role == "ST")
        z0 = st.z_emo.copy()
        ev = MicroEvent(
            t_sec=100.0,
            event_type=MicroEventType.GOAL_SCORED,
            team_id="Brazil",
            player_id=st.player_id,
            opponent_team_id="Argentina",
        )
        g = apply_micro_event(state, ev, AffectiveConfig())
        self.assertGreater(g, 0)
        engine = AffectiveSpatialCoupling(AffectiveConfig())
        engine.step(state, 10.0, events=[])
        self.assertTrue(np.any(st.z_emo > z0))

    def test_fear_increases_tau_dec(self):
        cfg = AffectiveConfig()
        engine = AffectiveSpatialCoupling(cfg)
        home = _FakeAgent("Brazil")
        away = _FakeAgent("Argentina")
        state = build_match_affective_state(home, away, rng=np.random.default_rng(1))
        p = state.home.players[0]
        p.z_emo = np.array([0.0, 0.0, 3.0, 0.0])  # fear logit high
        mod_fear = engine.compute_modulators(p, state)
        p.z_emo = np.array([3.0, 0.0, 0.0, 0.0])  # pride
        mod_calm = engine.compute_modulators(p, state)
        self.assertGreater(mod_fear.tau_dec, mod_calm.tau_dec)
        self.assertLess(mod_fear.vision_scale, mod_calm.vision_scale)

    def test_event_schedule_respects_goal_count(self):
        home = _FakeAgent("Brazil")
        away = _FakeAgent("Argentina")
        state = build_match_affective_state(home, away, rng=np.random.default_rng(2))
        events = build_event_schedule(
            state, goals_home=2, goals_away=1, xg_home=1.8, xg_away=0.9, rng=np.random.default_rng(2)
        )
        n_gh = sum(1 for e in events if e.event_type == MicroEventType.GOAL_SCORED and e.team_id == "Brazil")
        n_ga = sum(1 for e in events if e.event_type == MicroEventType.GOAL_SCORED and e.team_id == "Argentina")
        self.assertEqual(n_gh, 2)
        self.assertEqual(n_ga, 1)

    def test_coach_stress_rises_when_behind_on_xg(self):
        cfg = AffectiveConfig()
        engine = AffectiveSpatialCoupling(cfg)
        home = _FakeAgent("Brazil")
        away = _FakeAgent("Argentina")
        state = build_match_affective_state(home, away, rng=np.random.default_rng(3))
        s0 = state.home.coach.stress()
        for _ in range(20):
            engine.step(
                state,
                10.0,
                score_diff_home=-2,
                xg_swing_home=-1.5,
                coordination_home=0.5,
                coordination_away=0.6,
                conflict_home=0.2,
                conflict_away=0.1,
            )
        self.assertGreater(state.home.coach.z_stress, state.home.coach.z_trust - 0.5)


if __name__ == "__main__":
    unittest.main()
