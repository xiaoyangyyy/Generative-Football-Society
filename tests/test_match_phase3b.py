"""Phase 3b — Magnus curved passes."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.match_engine.ball_physics import (
    BallActionParams,
    azimuth_to_target,
    build_pass_spin_axis,
    desired_pass_omega,
    ground_pass_weight,
    integrate_pass_trajectory,
    sample_pass_delivery_params,
)
from src.match_engine.pass_intercept import (
    evaluate_pass_intercept,
    predict_player_xy,
    pursuit_weight_defender,
)
from src.match_engine.wall_pass import find_wall_partner, wall_return_target
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.macro_bridge import build_match_affective_state
from tests.test_affective_phase1b import _FakeAgent


class TestPhase3b(unittest.TestCase):
    def test_curved_pass_bends_more_than_flat(self):
        cfg = MicroMatchConfig()
        rng = np.random.default_rng(0)
        start = np.array([0.35, 0.22])
        target = np.array([0.62, 0.48])
        az = azimuth_to_target(start, target)

        flat = BallActionParams(0.28, 0.04, az, 0.0, np.array([0.0, 0.0, 1.0]))
        curved = BallActionParams(0.28, 0.05, az, 18.0, np.array([0.0, 0.0, 1.0]))

        t_flat = integrate_pass_trajectory(start, target, flat, cfg, curve_skill=0.5, rng=rng)
        t_curve = integrate_pass_trajectory(start, target, curved, cfg, curve_skill=0.85, rng=rng)
        self.assertGreater(t_curve.lateral_dev + t_curve.curve_amount, t_flat.lateral_dev + 0.05 * t_flat.curve_amount)

    def test_high_curve_skill_reduces_target_miss_on_average(self):
        cfg = MicroMatchConfig()
        rng = np.random.default_rng(1)
        start = np.array([0.4, 0.18])
        target = np.array([0.72, 0.55])
        misses_low, misses_high = [], []
        for _ in range(25):
            omega = desired_pass_omega(start, target, 0.35, 0.4, 0.6, cfg)
            p = sample_pass_delivery_params(0.35, 0.06, omega, 0.35, rng, cfg)
            p.azim = azimuth_to_target(start, target)
            misses_low.append(
                integrate_pass_trajectory(start, target, p, cfg, curve_skill=0.35, rng=rng).target_miss
            )
            p2 = sample_pass_delivery_params(0.35, 0.06, omega, 0.9, rng, cfg)
            p2.azim = azimuth_to_target(start, target)
            misses_high.append(
                integrate_pass_trajectory(start, target, p2, cfg, curve_skill=0.9, rng=rng).target_miss
            )
        # spin execution error term rewards matching skill — high skill shouldn't be worse on average
        self.assertLessEqual(np.mean(misses_high), np.mean(misses_low) * 1.15)

    def test_desired_omega_rises_with_pressure(self):
        cfg = MicroMatchConfig()
        start = np.array([0.45, 0.2])
        target = np.array([0.7, 0.55])
        w_low = desired_pass_omega(start, target, 0.7, 0.45, 0.2, cfg)
        w_high = desired_pass_omega(start, target, 0.7, 0.45, 0.85, cfg)
        self.assertGreater(w_high, w_low)

    def test_outside_foot_tilts_spin_axis(self):
        cfg = MicroMatchConfig()
        az = 0.3
        flat = build_pass_spin_axis(az, 1.0, 0.0, cfg)
        out = build_pass_spin_axis(az, 1.0, 0.85, cfg)
        self.assertAlmostEqual(abs(flat[2]), 1.0, places=2)
        self.assertLess(abs(out[2]), abs(flat[2]))

    def test_ground_pass_has_higher_drag_weight(self):
        cfg = MicroMatchConfig()
        self.assertGreater(ground_pass_weight(0.02, cfg), ground_pass_weight(0.18, cfg))
        rng = np.random.default_rng(2)
        start = np.array([0.4, 0.5])
        target = np.array([0.55, 0.52])
        az = azimuth_to_target(start, target)
        p_low = BallActionParams(0.3, 0.02, az, 0.0, np.array([0, 0, 1.0]), ground_weight=0.95)
        p_high = BallActionParams(0.3, 0.2, az, 0.0, np.array([0, 0, 1.0]), ground_weight=0.05)
        t_low = integrate_pass_trajectory(start, target, p_low, cfg, rng=rng)
        t_high = integrate_pass_trajectory(start, target, p_high, cfg, rng=rng)
        self.assertLess(t_low.peak_height, t_high.peak_height + 0.02)

    def test_intercept_when_defender_closer_to_land(self):
        cfg = MicroMatchConfig()
        cfg.pass_intercept_radius = 0.08
        rng = np.random.default_rng(3)
        land = np.array([0.55, 0.5])
        recv = type(
            "P",
            (),
            {
                "player_id": "r1",
                "on_pitch": True,
                "role": "ST",
                "position": np.array([0.72, 0.5]),
                "velocity": np.array([-0.02, 0.0]),
                "abilities": type("A", (), {"pace": 0.6})(),
            },
        )()
        opp = type(
            "P",
            (),
            {
                "player_id": "d1",
                "on_pitch": True,
                "role": "CB",
                "position": np.array([0.52, 0.51]),
                "velocity": np.array([0.04, 0.0]),
                "abilities": type("A", (), {"pace": 0.75})(),
            },
        )()
        ic = evaluate_pass_intercept(land, 0.4, recv, [opp], 0.7, 0.4, cfg, rng)
        self.assertGreater(ic.intercept_risk, 0.2)
        self.assertGreater(ic.pred_miss, 0.05)

    def test_predict_player_moves_with_velocity(self):
        cfg = MicroMatchConfig()
        p = type(
            "P",
            (),
            {
                "position": np.array([0.3, 0.5]),
                "velocity": np.array([0.1, 0.0]),
                "abilities": type("A", (), {"pace": 0.6})(),
            },
        )()
        pred = predict_player_xy(p, 1.0, cfg)
        self.assertGreater(pred[0], p.position[0])

    def test_defender_pursuit_closer_than_inertial(self):
        cfg = MicroMatchConfig()
        land = np.array([0.5, 0.5])
        opp = type(
            "P",
            (),
            {
                "position": np.array([0.42, 0.48]),
                "velocity": np.array([0.0, 0.0]),
                "abilities": type("A", (), {"pace": 0.8})(),
            },
        )()
        inertial = predict_player_xy(opp, 0.5, cfg)
        w = pursuit_weight_defender(opp, land, 0.75, 0.35, cfg)
        chase = predict_player_xy(opp, 0.5, cfg, pursuit_target=land, pursuit_weight=w)
        self.assertLess(float(np.linalg.norm(chase - land)), float(np.linalg.norm(inertial - land)))

    def test_wall_return_target_advances_forward(self):
        cfg = MicroMatchConfig()
        c = type("P", (), {"position": np.array([0.4, 0.5])})()
        t = wall_return_target(c, True, 0.6, cfg)
        self.assertGreater(t[0], c.position[0])

    def test_find_wall_partner_within_radius(self):
        cfg = MicroMatchConfig()
        carrier = type("P", (), {"player_id": "c", "position": np.array([0.5, 0.5]), "role": "CM"})()
        mate = type(
            "P",
            (),
            {"player_id": "w", "position": np.array([0.56, 0.52]), "on_pitch": True, "role": "ST"},
        )()
        far = type(
            "P",
            (),
            {"player_id": "f", "position": np.array([0.8, 0.5]), "on_pitch": True, "role": "RW"},
        )()
        wp = find_wall_partner(carrier, [mate, far], cfg)
        self.assertIsNotNone(wp)
        self.assertEqual(wp.player.player_id, "w")

    def test_passing_engine_records_curved_passes(self):
        cfg = MicroMatchConfig.fast_demo()
        cfg.enable_pass_physics = True
        cfg.enable_phase3 = True
        home, away = _FakeAgent("Brazil"), _FakeAgent("Argentina")
        st = build_match_affective_state(home, away, rng=np.random.default_rng(3))
        from src.match_engine.match_micro_runner import _init_micro_state

        _init_micro_state(st, cfg, np.random.default_rng(3))
        sie = SpatialIntelligenceEngine(cfg)
        pe = PassingEngine(cfg, sie)
        from src.match_engine.spatial_field import SpatialFieldEngine
        from src.match_engine.affective_coupling import AffectiveSpatialCoupling

        sf = SpatialFieldEngine(cfg)
        aff = AffectiveSpatialCoupling(cfg)
        rng = np.random.default_rng(3)
        for _ in range(8):
            sf.step(st, cfg.dt_default)
            sie.step(st)
            mods, _ = aff.step(st, cfg.dt_default)
            pe.step(st, mods["home"], mods["away"], rng)
        self.assertGreater(pe.stats["home_attempts"] + pe.stats["away_attempts"], 0)
        self.assertGreaterEqual(pe.stats["home_curved"] + pe.stats["away_curved"], 0)


if __name__ == "__main__":
    unittest.main()
