"""Unified action selection: pass / shot / cross / hold."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.match_engine.math_utils import softmax
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEvent
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.aerial_duel import AerialDuelEngine
from src.match_engine.state import MatchAffectiveState, PlayerModulators
from src.match_engine.tactical_engine import TacticalMicroEngine
from src.match_engine.possession_helpers import (
    mark_shot_taken,
    shot_cooldown_ok,
    turnover_to_defence_after_shot,
)
from src.match_engine.wall_pass import find_wall_partner


class ActionEngine:
    def __init__(
        self,
        cfg: MicroMatchConfig,
        passing: PassingEngine,
        shots: ShotEngine,
        aerial: AerialDuelEngine,
        wm_runtime=None,
    ):
        self.cfg = cfg
        self.passing = passing
        self.shots = shots
        self.aerial = aerial
        self.wm_runtime = wm_runtime
        self._tac_eng = TacticalMicroEngine(cfg)

    def step(
        self,
        state: MatchAffectiveState,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        carrier = self.passing._get_carrier(state)
        if carrier is None:
            self.passing._assign_random_possession(state, rng)
            return "kickoff", []

        attacking_home = carrier.team_id == state.home.team_id
        mod_list = mod_home if attacking_home else mod_away
        mod_map = {m.player_id: m for m in mod_list}
        mod_c = mod_map.get(carrier.player_id, mod_list[0])

        dist_goal = self.shots._dist_to_goal(carrier.position, attacking_home)
        tac = state.team(carrier.team_id).coach.tactical_current or {}
        u_shot = (
            self.cfg.action_shot_base
            + self.shots.shot_utility_max(state, carrier, mod_c) * self._tac_eng.shot_bias(tac)
            + self.cfg.action_shot_dist_bonus * max(0.0, self.cfg.shot_max_dist - dist_goal)
        )
        u_pass = self.cfg.action_pass_base
        if self.cfg.enable_wall_pass:
            team = state.team(carrier.team_id)
            if find_wall_partner(carrier, team.players, self.cfg) is not None:
                press = float(
                    state.spatial.press[
                        int(np.clip(carrier.position[0] * (state.spatial.press.shape[0] - 1), 0, state.spatial.press.shape[0] - 1)),
                        int(np.clip(carrier.position[1] * (state.spatial.press.shape[1] - 1), 0, state.spatial.press.shape[1] - 1)),
                    ]
                )
                u_pass += 0.12 + 0.2 * max(0.0, press - 0.35)
        u_cross = 0.0
        if carrier.role in ("LW", "RW", "LB", "RB") and dist_goal > 0.16:
            u_cross = self.cfg.action_cross_base + 0.32 * float(carrier.abilities.curve) - 0.15 * dist_goal
        elif carrier.role in ("LW", "RW", "LB", "RB"):
            u_cross = self.cfg.action_cross_base * 0.65
        u_hold = -0.02 + 0.04 * (1.0 - dist_goal)

        utils = np.array([u_pass, u_shot, u_cross, u_hold], dtype=float)
        utils = np.nan_to_num(utils, nan=0.0, posinf=5.0, neginf=-5.0)
        labels = ["pass", "shot", "cross", "hold"]
        if self.wm_runtime is not None:
            from src.match_engine.world_model.planner import action_imagination_adjustments

            utils = action_imagination_adjustments(
                self.wm_runtime,
                state,
                carrier,
                attacking_home,
                utils,
                labels,
                dist_goal=dist_goal,
            )
        tau = self.cfg.action_tau * mod_c.tau_dec
        probs = softmax(utils, tau=max(0.2, tau))
        choice = labels[int(rng.choice(len(labels), p=probs))]

        events: List[MicroEvent] = []

        if choice == "shot" and dist_goal < self.cfg.shot_max_dist:
            if hasattr(state, "_wm_last_action"):
                from src.match_engine.world_model.action_codec import encode_high_level_action

                state._wm_last_action = encode_high_level_action("shot", carrier.position)
            now = float(state.clock_seconds)
            if not shot_cooldown_ok(state, attacking_home, self.cfg, now):
                _, pass_ev = self.passing.step(state, mod_home, mod_away, rng)
                events.extend(pass_ev)
                return "pass", events
            out = self.shots.resolve_shot(state, carrier, mod_c, rng, from_schedule=False)
            mark_shot_taken(state, attacking_home, now)
            events.extend(out.events)
            if out.goal:
                state.ball.position = out.traj.landed
                state.ball.possessor_id = None
            else:
                def_id = state.away.team_id if attacking_home else state.home.team_id
                gk = self.shots._gk_player(state, def_id)
                turnover_to_defence_after_shot(state, attacking_home, gk, rng)
            return "shot", events

        if choice == "cross" and carrier.role in ("LW", "RW", "LB", "RB"):
            traj, aerial_out = self.aerial.resolve_cross(state, carrier, rng)
            state.ball.position = aerial_out.landed_xy
            from src.match_engine.aerial_duel import apply_aerial_xg_to_state

            apply_aerial_xg_to_state(state, aerial_out.xg_added, attacking_home, self.cfg)
            if aerial_out.winner_id:
                for p in state.home.players + state.away.players:
                    if p.player_id == aerial_out.winner_id:
                        state.ball.possessor_id = p.player_id
                        state.ball.possession_team_id = p.team_id
                        break
            return "cross", events

        if choice == "hold":
            if hasattr(state, "_wm_last_action"):
                from src.match_engine.world_model.action_codec import encode_high_level_action

                state._wm_last_action = encode_high_level_action("hold")
            return "hold", events

        _, pass_ev = self.passing.step(state, mod_home, mod_away, rng)
        events.extend(pass_ev)
        return "pass", events
