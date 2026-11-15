from __future__ import annotations

import numpy as np

from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.contracts.micro_math import sigmoid
from gfs_core.contracts.micro_state import (
    MatchAffectiveState,
    PlayerModulators,
    TeamAffectiveState,
)


class KinematicPositionLayer:
    def __init__(self, config: MicroMatchConfig, spatial, spatial_intelligence=None) -> None:
        self.config = config
        self.spatial = spatial
        self.spatial_intelligence = spatial_intelligence

    def step_team(
        self,
        state: MatchAffectiveState,
        team: TeamAffectiveState,
        modulators: dict[str, PlayerModulators],
        *,
        is_home: bool,
    ) -> None:
        config = self.config
        ball = state.ball.position
        tactical = team.coach.tactical_current or {}
        width = float(tactical.get("width_play", tactical.get("rotation_aggressiveness", 0.5)))
        line_height = float(tactical.get("line_height", 0.5))
        rotation = float(tactical.get("rotation_aggressiveness", 0.5))
        overlap = float(tactical.get("overlap_fullbacks", 0.5))
        width = 0.5 * width + 0.5 * rotation
        width = 0.85 * width + 0.15 * overlap

        for player in team.players:
            if not player.on_pitch:
                continue
            modulator = modulators.get(player.player_id)
            alpha = modulator.move_alpha if modulator else config.move_alpha_base
            alpha *= config.max_speed
            anchor = self.spatial.interpolate_anchors(
                player.role,
                team.phase,
                line_height=line_height,
                width=width,
                attacks_high_x=team.attacks_high_x,
                cb_wide=player.cb_wide,
                formation_key=team.formation_key,
            )
            if not team.attacks_high_x:
                anchor = self.spatial.mirror_for_away(anchor)
            to_anchor = anchor - player.position
            to_ball = ball - player.position
            ball_distance = float(np.linalg.norm(to_ball)) + 1e-6

            if player.player_id == state.ball.possessor_id:
                velocity = 0.3 * state.ball.velocity + 0.1 * to_anchor
            else:
                form_weight = config.w_form_move
                support_weight = config.w_supp_move * (
                    1.0 - min(1.0, ball_distance / 0.35)
                )
                ball_weight = 0.08 * team.phase
                velocity = (
                    form_weight * to_anchor
                    + support_weight * (to_ball / ball_distance) * 0.05
                    + ball_weight * (to_ball / ball_distance) * 0.03
                )
                if (
                    config.enable_phi_gradient_move
                    and self.spatial_intelligence is not None
                    and state.ball.possession_team_id == team.team_id
                ):
                    gradient = self.spatial_intelligence.phi_gradient_at(
                        state, player.position, is_home,
                    )
                    velocity += (
                        config.w_phi_move * team.phase * gradient * config.phi_move_scale
                    )
            player.velocity = 0.65 * player.velocity + 0.35 * velocity
            speed = float(np.linalg.norm(player.velocity))
            if speed > alpha:
                player.velocity = player.velocity * (alpha / speed)
            player.position = integrate_position(
                player.position, player.velocity, config.dt_default,
            )
            target_orientation = float(np.arctan2(to_ball[1], to_ball[0]))
            angle_delta = float(np.arctan2(
                np.sin(target_orientation - player.orientation),
                np.cos(target_orientation - player.orientation),
            ))
            player.orientation += angle_delta * float(
                -np.expm1(-config.orient_rate * config.dt_default)
            )

    def step(self, state: MatchAffectiveState, mod_home: dict, mod_away: dict) -> None:
        possession_home = state.home.possession_share
        press_home = float(
            state.away.coach.tactical_current.get("pressing_intensity", 0.5)
        )
        press_away = float(
            state.home.coach.tactical_current.get("pressing_intensity", 0.5)
        )
        self._update_phase(state.home, possession_home, press_away)
        self._update_phase(state.away, 1.0 - possession_home, press_home)
        self.step_team(
            state,
            state.home,
            {item.player_id: item for item in mod_home},
            is_home=True,
        )
        self.step_team(
            state,
            state.away,
            {item.player_id: item for item in mod_away},
            is_home=False,
        )
        for team in (state.home, state.away):
            for player in team.players:
                if player.role == "GK":
                    goal_x = 0.06 if team.attacks_high_x else 0.94
                    delta = goal_x - player.position[0]
                    limit = self.config.max_speed * self.config.dt_default
                    player.position[0] += float(np.clip(delta, -limit, limit))

    def _update_phase(
        self,
        team: TeamAffectiveState,
        possession: float,
        press_against: float,
    ) -> None:
        line_height = float(team.coach.tactical_current.get("line_height", 0.5))
        target = sigmoid(possession + line_height - press_against)
        team.phase += (target - team.phase) * float(
            -np.expm1(-self.config.kappa_phase * self.config.dt_default)
        )
        team.phase = float(np.clip(team.phase, 0.0, 1.0))


def integrate_position(position, velocity, dt: float):
    if dt < 0:
        raise ValueError("motion timestep cannot be negative")
    return np.clip(position + velocity * dt, 0.02, 0.98)
