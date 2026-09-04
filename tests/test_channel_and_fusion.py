"""Channel passing + macro/micro fusion + tactics sync."""

import json
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.match_engine.player_channel_passing import channel_pass_utility_boost
from src.match_engine.state import PlayerAffectiveState
from src.memory_engine.macro_micro_fusion import fuse_xg
from src.simulation.tactics_sync import apply_coach_tactics_from_llm


class _FakeAgent:
  team_name = "Test"
  formation = "4-3-3"
  style_desc = ""
  style_archetype = "balanced"
  tactical_controls = {
    "pressing_intensity": 0.5,
    "risk_budget": 0.5,
    "line_height": 0.5,
    "rotation_aggressiveness": 0.5,
  }
  coach_profile = None
  semantic_memory = {}

  def set_tactical_controls(self, c):
    self.tactical_controls.update(c)


def test_channel_boost_through():
    p = PlayerAffectiveState(
        player_id="x",
        name="x",
        role="AM",
        team_id="T",
        channel_affinities={"creative_hub": 0.7, "progressive_passer": 0.2},
    )
    u_through = channel_pass_utility_boost(p, "through")
    u_long = channel_pass_utility_boost(p, "long")
    assert u_through > u_long


def test_fuse_xg_blend():
    xh, xa = fuse_xg(1.2, 0.8, 2.0, 1.0, eta=0.5)
    assert abs(xh - 1.6) < 1e-6
    assert abs(xa - 0.9) < 1e-6


def test_tactics_sync_vector():
    agent = _FakeAgent()
    payload = json.dumps(
        {
            "formation": "3-5-2",
            "style": "high press wide",
            "controls": {"pressing_intensity": 0.85, "risk_budget": 0.6},
            "tactical_hints": {"through_ball_bias": 0.8, "high_press": 0.9},
        }
    )
    apply_coach_tactics_from_llm(agent, payload)
    assert hasattr(agent, "tactical_vector")
    assert len(agent.tactical_vector) >= 21
    assert agent.tactical_controls["pressing_intensity"] >= 0.7
