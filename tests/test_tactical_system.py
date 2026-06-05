"""Mainstream tactical catalog + micro integration."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.match_engine.formation import normalize_formation_key, roles_for_formation
from src.match_engine.tactical_catalog import (
    TACTICAL_PRESETS,
    infer_archetype_from_text,
)
from src.match_engine.tactical_profile import build_tactical_vector_for_agent
from tests.test_affective_phase1b import _FakeAgent


class TestTacticalSystem(unittest.TestCase):
    def test_catalog_covers_mainstream_styles(self):
        required = {
            "tiki_taka",
            "gegenpress",
            "low_block_counter",
            "catenaccio",
            "wing_play",
            "long_ball",
            "counter_attack",
            "false_nine",
            "total_football",
            "park_the_bus",
            "mid_block",
            "bielsa_man_marking",
        }
        self.assertTrue(required.issubset(set(TACTICAL_PRESETS.keys())))

    def test_infer_chinese_press(self):
        arch = infer_archetype_from_text("高位压迫，边路主导", "4-3-3")
        self.assertIn(arch, ("gegenpress", "high_press", "bielsa_man_marking"))

    def test_infer_five_at_back(self):
        arch = infer_archetype_from_text("五后卫低位防守", "5-3-2")
        self.assertIn(arch, ("park_the_bus", "catenaccio", "low_block_counter"))

    def test_formation_roles_442(self):
        roles = roles_for_formation("4-4-2")
        self.assertEqual(len(roles), 11)
        self.assertEqual(roles.count("ST"), 2)

    def test_vector_from_agent(self):
        agent = _FakeAgent("Spain")
        agent.style_desc = "Tiki-Taka 传控"
        agent.formation = "4-3-3"
        agent.tactical_controls = {
            "pressing_intensity": 0.6,
            "risk_budget": 0.4,
            "line_height": 0.55,
            "rotation_aggressiveness": 0.65,
        }
        v = build_tactical_vector_for_agent(agent)
        self.assertGreater(v["possession_orientation"], 0.7)
        self.assertIn("through_ball_bias", v)

    def test_normalize_formation_multi(self):
        self.assertEqual(normalize_formation_key("4-2-3-1 - 4-3-3"), "4231")
        self.assertEqual(normalize_formation_key("4-4-2"), "442")


if __name__ == "__main__":
    unittest.main()
