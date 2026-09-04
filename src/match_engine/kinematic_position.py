"""§4 Phase 2a — player positions, orientation, phase."""

from __future__ import annotations

import numpy as np

from src.match_engine.formation import interpolate_anchors, mirror_for_away
from src.match_engine.math_utils import sigmoid
from typing import TYPE_CHECKING

from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.state import MatchAffectiveState, PlayerModulators, TeamAffectiveState

if TYPE_CHECKING:
    from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine


class KinematicPositionLayer:
    def __init__(self, cfg: MicroMatchConfig, sie: "SpatialIntelligenceEngine | None" = None):
        self.cfg = cfg
        self.sie = sie

    def _update_team_phase(self, team: TeamAffectiveState, possession: float, press_against: float) -> None:
        lh = float(team.coach.tactical_current.get("line_height", 0.5))
        target = sigmoid(possession + lh - press_against)
        team.phase += self.cfg.kappa_phase * (target - team.phase)
        team.phase = float(np.clip(team.phase, 0.0, 1.0))

    def step_team(
        self,
        state: MatchAffectiveState,
        team: TeamAffectiveState,
        modulators: dict[str, PlayerModulators],
        *,
        is_home: bool,
    ) -> None:
        cfg = self.cfg
        ball = state.ball.position
        tac = team.coach.tactical_current or {}
        width = float(tac.get("width_play", tac.get("rotation_aggressiveness", 0.5)))
        line_h = float(tac.get("line_height", 0.5))
        rot = float(tac.get("rotation_aggressiveness", 0.5))
        overlap = float(tac.get("overlap_fullbacks", 0.5))
        width = 0.5 * width + 0.5 * rot
        width = 0.85 * width + 0.15 * overlap

        for p in team.players:
            if not p.on_pitch:
                continue
            mod = modulators.get(p.player_id)
            alpha = mod.move_alpha if mod else cfg.move_alpha_base
            alpha *= cfg.max_speed

            anchor = interpolate_anchors(
                p.role,
                team.phase,
                line_height=line_h,
                width=width,
                attacks_high_x=team.attacks_high_x,
                cb_wide=p.cb_wide,
                formation_key=getattr(team, "formation_key", "433"),
            )
            if not team.attacks_high_x:
                anchor = mirror_for_away(anchor)

            # potential: form anchor + support ball + phi gradient proxy (toward ball when attacking)
            to_anchor = anchor - p.position
            to_ball = ball - p.position
            dist_ball = float(np.linalg.norm(to_ball)) + 1e-6

            if p.player_id == state.ball.possessor_id:
                # on ball: slight drift with ball
                vel = 0.3 * state.ball.velocity + 0.1 * to_anchor
            else:
                w_form = cfg.w_form_move
                w_supp = cfg.w_supp_move * (1.0 - min(1.0, dist_ball / 0.35))
                w_ball = 0.08 * team.phase
                vel = w_form * to_anchor + w_supp * (to_ball / dist_ball) * 0.05 + w_ball * (
                    to_ball / dist_ball
                ) * 0.03
                if (
                    cfg.enable_phi_gradient_move
                    and self.sie is not None
                    and state.ball.possession_team_id == team.team_id
                ):
                    grad = self.sie.phi_gradient_at(state, p.position, is_home)
                    vel += cfg.w_phi_move * team.phase * grad * cfg.phi_move_scale

            p.velocity = (1.0 - 0.35) * p.velocity + 0.35 * vel
            speed = float(np.linalg.norm(p.velocity))
            if speed > alpha:
                p.velocity = p.velocity * (alpha / speed)
            p.position = np.clip(p.position + p.velocity * cfg.dt_default / 10.0, 0.02, 0.98)

            # orientation toward ball
            target_theta = float(np.arctan2(to_ball[1], to_ball[0]))
            p.orientation += cfg.orient_rate * 0.1 * (target_theta - p.orientation)

    def step(
        self,
        state: MatchAffectiveState,
        mod_home: dict,
        mod_away: dict,
    ) -> None:
        poss_h = state.home.possession_share
        press_h = float(state.away.coach.tactical_current.get("pressing_intensity", 0.5))
        press_a = float(state.home.coach.tactical_current.get("pressing_intensity", 0.5))
        self._update_team_phase(state.home, poss_h, press_a)
        self._update_team_phase(state.away, 1.0 - poss_h, press_h)

        self.step_team(state, state.home, {m.player_id: m for m in mod_home}, is_home=True)
        self.step_team(state, state.away, {m.player_id: m for m in mod_away}, is_home=False)

        # GK stays deep
        for team in (state.home, state.away):
            for p in team.players:
                if p.role == "GK":
                    gx = 0.06 if team.attacks_high_x else 0.94
                    p.position[0] = 0.85 * p.position[0] + 0.15 * gx
