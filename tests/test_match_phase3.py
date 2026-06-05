"""Phase 3 — ball physics, shots, GK, micro λ."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.match_engine.ball_physics import (
    BallActionParams,
    azimuth_to_goal,
    integrate_trajectory,
    sample_shot_params,
)
from src.match_engine.goal_generator import lambdas_from_micro
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.state import PlayerModulators
from tests.test_affective_phase1b import _FakeAgent


class TestPhase3(unittest.TestCase):
    def test_trajectory_crosses_goal_line(self):
        cfg = MicroMatchConfig()
        rng = np.random.default_rng(0)
        start = np.array([0.75, 0.5])
        params = sample_shot_params("driven", 0.25, type("A", (), {"tech": 0.7, "shot": 0.7, "curve": 0.5, "power": 0.6, "knuckle": 0.4})(), 0.0, rng, cfg)
        params.azim = azimuth_to_goal(start, True)
        traj = integrate_trajectory(start, params, cfg, attacking_high_x=True, rng=rng)
        self.assertTrue(traj.crossed_goal_line or traj.peak_height > 0.05)

    def test_curved_has_higher_spin_than_driven(self):
        cfg = MicroMatchConfig()
        rng = np.random.default_rng(1)
        ab = type("A", (), {"tech": 0.65, "shot": 0.7, "curve": 0.85, "power": 0.6, "knuckle": 0.4})()
        p_d = sample_shot_params("driven", 0.2, ab, 0.0, rng, cfg)
        p_c = sample_shot_params("curved", 0.2, ab, 0.0, rng, cfg)
        self.assertGreater(p_c.omega, p_d.omega)

    def test_xg_decreases_with_distance(self):
        cfg = MicroMatchConfig()
        home, away = _FakeAgent("H"), _FakeAgent("A")
        st = build_match_affective_state(home, away, rng=np.random.default_rng(2))
        sie = SpatialIntelligenceEngine(cfg)
        shots = ShotEngine(cfg, sie)
        rng = np.random.default_rng(2)
        carrier = st.home.players[9]
        carrier.position = np.array([0.82, 0.5])
        mod = PlayerModulators(player_id=carrier.player_id)
        xg_near = []
        xg_far = []
        for _ in range(30):
            carrier.position = np.array([0.82, 0.5])
            xg_near.append(shots.resolve_shot(st, carrier, mod, rng, forced_kind="driven").xg)
            carrier.position = np.array([0.55, 0.5])
            xg_far.append(shots.resolve_shot(st, carrier, mod, rng, forced_kind="driven").xg)
        self.assertGreater(np.mean(xg_near), np.mean(xg_far))

    def test_gk_reflex_improves_save_prob(self):
        cfg = MicroMatchConfig()
        from src.match_engine.ball_physics import TrajectoryResult

        traj = TrajectoryResult(
            landed=np.array([0.02, 0.5]),
            peak_height=0.8,
            time_of_flight=0.5,
            crossed_goal_line=True,
            goal_y=0.5,
            in_goal_mouth=True,
            curve_amount=0.1,
        )
        self.assertGreater(traj.time_of_flight, 0.0)
        self.assertTrue(0.0 <= float(traj.landed[0]) <= 1.0)
        params = BallActionParams(0.4, 0.1, 0.0, 5.0, np.array([0, 0, 1.0]))
        gk_weak = type("GK", (), {"abilities": type("Ab", (), {"gk_reflex": 0.2, "gk_aerial": 0.3})(), "position": np.array([0.02, 0.5])})()
        gk_strong = type("GK", (), {"abilities": type("Ab", (), {"gk_reflex": 0.95, "gk_aerial": 0.9})(), "position": np.array([0.02, 0.5])})()
        sie = SpatialIntelligenceEngine(cfg)
        eng = ShotEngine(cfg, sie)
        p_weak = eng._gk_save_prob(traj, gk_weak, params, 0.2)
        p_strong = eng._gk_save_prob(traj, gk_strong, params, 0.2)
        self.assertGreater(p_strong, p_weak)

    def test_lambdas_from_micro_monotone_in_xg(self):
        cfg = MicroMatchConfig()
        l1, _ = lambdas_from_micro(0.5, 0.3, 32.0, 30.0, cfg)
        l2, _ = lambdas_from_micro(1.2, 0.3, 32.0, 30.0, cfg)
        self.assertGreater(l2, l1)

    def test_micro_runner_phase3_stats(self):
        from src.match_engine.match_micro_runner import run_match_micro_simulation

        cfg = MicroMatchConfig.fast_demo()
        cfg.enable_phase3 = True
        home, away = _FakeAgent("Brazil"), _FakeAgent("Argentina")
        summary = run_match_micro_simulation(
            home,
            away,
            goals_home=1,
            goals_away=1,
            xg_home=1.1,
            xg_away=0.9,
            config=cfg,
            seed=7,
            writeback_agents=False,
        )
        self.assertGreater(summary.shots_home + summary.shots_away, 0)
        self.assertGreater(summary.micro_xg_home + summary.micro_xg_away, 0.0)


if __name__ == "__main__":
    unittest.main()

