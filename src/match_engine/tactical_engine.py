"""
Apply full tactical vectors inside the micro match tick loop.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.tactical_profile import (
    apply_vector_to_team_coach,
    build_tactical_vector_for_agent,
    legacy_controls_from_vector,
)

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState, TeamAffectiveState
    from src.simulation.agent import SocietyAgent


class TacticalMicroEngine:
    def __init__(self, cfg: MicroMatchConfig):
        self.cfg = cfg

    def bootstrap(
        self,
        state: "MatchAffectiveState",
        home_agent: "SocietyAgent",
        away_agent: "SocietyAgent",
    ) -> None:
        th = build_tactical_vector_for_agent(home_agent)
        ta = build_tactical_vector_for_agent(away_agent)
        apply_vector_to_team_coach(state.home, th)
        apply_vector_to_team_coach(state.away, ta)
        state.home.formation_key = getattr(home_agent, "_formation_key", "433")
        state.away.formation_key = getattr(away_agent, "_formation_key", "433")

    def get_team_tactics(self, team: "TeamAffectiveState") -> Dict[str, float]:
        return dict(team.coach.tactical_current or team.coach.tactical_base or {})

    def press_multiplier(self, tac: Dict[str, float]) -> float:
        return float(
            0.55
            + 0.45 * tac.get("pressing_intensity", 0.5)
            + 0.25 * tac.get("counterpress", 0.5)
            + 0.15 * tac.get("man_oriented_press", 0.5)
        )

    def passing_bias(self, tac: Dict[str, float]) -> Dict[str, float]:
        return {
            "short": 0.25 * tac.get("possession_orientation", 0.5) + 0.20 * tac.get("build_up_short", 0.5),
            "through": 0.35 * tac.get("through_ball_bias", 0.5) + 0.20 * tac.get("verticality", 0.5),
            "long": 0.35 * tac.get("long_ball_bias", 0.5) + 0.15 * (1.0 - tac.get("possession_orientation", 0.5)),
            "wall": 0.15 * tac.get("possession_orientation", 0.5) * tac.get("build_up_short", 0.5),
            "cross": 0.40 * tac.get("cross_frequency", 0.5) * tac.get("wing_focus", 0.5),
            "tempo_scale": 0.85 + 0.30 * tac.get("tempo", 0.5),
        }

    def kinematic_scales(self, tac: Dict[str, float]) -> Dict[str, float]:
        return {
            "width": tac.get("width_play", 0.5),
            "line_height": tac.get("line_height", 0.5),
            "rotation": tac.get("rotation_aggressiveness", 0.5),
            "overlap": tac.get("overlap_fullbacks", 0.5),
            "compactness": tac.get("compactness", 0.5),
        }

    def shot_bias(self, tac: Dict[str, float]) -> float:
        return float(0.95 + 0.45 * tac.get("verticality", 0.5) + 0.28 * tac.get("risk_budget", 0.5))

    def step_in_match_pressure(self, state: "MatchAffectiveState", dt: float) -> None:
        """Slow drift of tactical_current under score / possession (continuous)."""
        score_h = state.home.score - state.away.score
        for team, swing in ((state.home, score_h), (state.away, -score_h)):
            tac = self.get_team_tactics(team)
            if not tac:
                continue
            cur = dict(tac)
            if swing < -1:
                cur["low_block"] = min(1.0, cur.get("low_block", 0.5) + 0.02 * dt)
                cur["risk_budget"] = max(0.0, cur.get("risk_budget", 0.5) - 0.015 * dt)
            elif swing > 1:
                cur["high_press"] = min(1.0, cur.get("high_press", 0.5) + 0.02 * dt)
                cur["risk_budget"] = min(1.0, cur.get("risk_budget", 0.5) + 0.012 * dt)
            team.coach.tactical_current = cur
