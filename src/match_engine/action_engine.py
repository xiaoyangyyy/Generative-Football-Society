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
        continuous_clock=None,
        subtick_queue=None,
    ):
        self.cfg = cfg
        self.passing = passing
        self.shots = shots
        self.aerial = aerial
        self.wm_runtime = wm_runtime
        self.continuous_clock = continuous_clock
        self.subtick_queue = subtick_queue
        self._tac_eng = TacticalMicroEngine(cfg)
        self._hierarchical = None
        if cfg.enable_hierarchical_policy:
            from src.match_engine.hierarchical_policy import HierarchicalPolicy

            self._hierarchical = HierarchicalPolicy()

    def step(
        self,
        state: MatchAffectiveState,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        if self.subtick_queue is not None:
            self.subtick_queue.capture_pre_action(state)
        planned_kind = None
        if self.continuous_clock is not None:
            self.continuous_clock.observe(state)
            if self.continuous_clock.is_gated(state.clock_seconds):
                return "hold", []
            planned_kind = self.continuous_clock.pop_due_kind(state.clock_seconds)
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
        low_block = float(tac.get("low_block", 0.35))
        u_shot = (
            self.cfg.action_shot_base
            + self.shots.shot_utility_max(state, carrier, mod_c) * self._tac_eng.shot_bias(tac)
            + self.cfg.action_shot_dist_bonus * max(0.0, self.cfg.shot_max_dist - dist_goal) * (1.0 - 0.55 * low_block)
        )
        side_key = "home_shots" if attacking_home else "away_shots"
        opp_key = "away_shots" if attacking_home else "home_shots"
        team_shots = int(self.shots.stats.get(side_key, 0))
        opp_shots = int(self.shots.stats.get(opp_key, 0))
        def_block = float(np.clip((low_block - 0.50) / 0.45, 0.0, 1.0))
        if def_block > 0.05:
            vol_decay = float(np.exp(-self.cfg.action_shot_volume_decay * def_block * team_shots))
            u_shot *= vol_decay
            if team_shots >= 6 and opp_shots <= max(2, team_shots // 4):
                skew = float(np.exp(-self.cfg.action_shot_skew_decay * def_block * (team_shots - opp_shots)))
                u_shot *= skew
        if team_shots + 2 <= opp_shots and dist_goal < self.cfg.shot_max_dist * 0.85:
            u_shot += 0.14 * min(3.0, (opp_shots - team_shots) ** 0.5)
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
        u_hold = -0.02 + 0.04 * (1.0 - dist_goal) + 0.10 * low_block * min(1.0, dist_goal / max(1e-6, self.cfg.shot_max_dist))

        utils = np.array([u_pass, u_shot, u_cross, u_hold], dtype=float)
        if self._hierarchical is not None:
            options = self._hierarchical.decompose(tac)
            role_unit = (
                "back_line" if carrier.role in {"GK", "CB", "LB", "RB"}
                else "midfield" if carrier.role in {"DM", "CM", "AM", "LM", "RM"}
                else "front_line"
            )
            option = next(item for item in options if item.unit == role_unit)
            intent = self._hierarchical.player_intent(option, carrier.role)
            utils[0] += 0.08 * float(intent.get("support", 0.0))
            utils[1] += 0.08 * float(intent.get("risk", 0.0))
            utils[3] += 0.05 * float(intent.get("cover", 0.0))
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
                tac=tac,
                team_shots=team_shots,
            )
        feasible_actions = {"pass", "hold"}
        if dist_goal < self.cfg.shot_max_dist and shot_cooldown_ok(
            state, attacking_home, self.cfg, float(state.clock_seconds),
        ):
            feasible_actions.add("shot")
        if carrier.role in ("LW", "RW", "LB", "RB"):
            feasible_actions.add("cross")
        from src.match_engine.world_model.decision_adoption import (
            capture_policy_outcome_baseline,
            pending_policy_action_bias,
            record_policy_intervention_result,
        )

        policy_intent = pending_policy_action_bias(
            state,
            team_id=carrier.team_id,
            feasible_actions=feasible_actions,
            t_sec=float(state.clock_seconds),
        )
        if policy_intent is not None:
            selected_index = labels.index(policy_intent["action"])
            utils[selected_index] += min(
                0.5, max(0.0, float(policy_intent["logit_bias"]))
            )
        policy_outcome_baseline = (
            capture_policy_outcome_baseline(state, team_id=carrier.team_id)
            if policy_intent is not None else None
        )

        def record_policy_result(actual_action: str) -> None:
            if policy_intent is not None:
                record_policy_intervention_result(
                    state,
                    decision_id=policy_intent["decision_id"],
                    actual_action=actual_action,
                    t_sec=float(state.clock_seconds),
                    outcome_baseline=policy_outcome_baseline,
                )

        tau = self.cfg.action_tau * mod_c.tau_dec
        probs = softmax(utils, tau=max(0.2, tau))
        choice = labels[int(rng.choice(len(labels), p=probs))]
        if planned_kind in {"pass", "shot"}:
            choice = planned_kind

        events: List[MicroEvent] = []

        if choice == "shot" and dist_goal < self.cfg.shot_max_dist:
            if hasattr(state, "_wm_last_action"):
                from src.match_engine.world_model.action_codec import encode_high_level_action

                state._wm_last_action = encode_high_level_action(
                    "shot", carrier.position,
                    horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
                )
            now = float(state.clock_seconds)
            if not shot_cooldown_ok(state, attacking_home, self.cfg, now):
                pass_action, pass_ev = self.passing.step(state, mod_home, mod_away, rng)
                if self.continuous_clock is not None:
                    self.continuous_clock.schedule_after_pass(state, pass_action)
                if self.subtick_queue is not None:
                    self.subtick_queue.schedule_after_pass(state, pass_action)
                    self.subtick_queue.reconcile(state.clock_seconds + self.cfg.dt_default, state)
                events.extend(pass_ev)
                record_policy_result("pass")
                return "pass", events
            out = self.shots.resolve_shot(state, carrier, mod_c, rng, from_schedule=False)
            if getattr(state, "_wm_last_action", None) is not None:
                from src.match_engine.world_model.schema import SHOT_GOAL_INDEX, SHOT_ON_TARGET_INDEX

                state._wm_last_action[13] = float(np.clip(out.xg, 0.0, 1.0))
                state._wm_last_action[SHOT_GOAL_INDEX] = 1.0 if out.goal else 0.0
                state._wm_last_action[SHOT_ON_TARGET_INDEX] = 1.0 if out.on_target else 0.0
            mark_shot_taken(state, attacking_home, now)
            events.extend(out.events)
            if out.goal:
                state.ball.position = out.traj.landed
                state.ball.possessor_id = None
            else:
                def_id = state.away.team_id if attacking_home else state.home.team_id
                gk = self.shots._gk_player(state, def_id)
                turnover_to_defence_after_shot(state, attacking_home, gk, rng)
            record_policy_result("shot")
            return "shot", events

        if choice == "cross" and carrier.role in ("LW", "RW", "LB", "RB"):
            if hasattr(state, "_wm_last_action"):
                from src.match_engine.world_model.action_codec import encode_high_level_action

                state._wm_last_action = encode_high_level_action(
                    "cross", carrier.position,
                    horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
                )
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
            record_policy_result("cross")
            return "cross", events

        if choice == "hold":
            if hasattr(state, "_wm_last_action"):
                from src.match_engine.world_model.action_codec import encode_high_level_action

                state._wm_last_action = encode_high_level_action(
                    "hold",
                    target=state.ball.position,
                    horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
                )
            record_policy_result("hold")
            return "hold", events

        pass_action, pass_ev = self.passing.step(state, mod_home, mod_away, rng)
        if self.continuous_clock is not None:
            self.continuous_clock.schedule_after_pass(state, pass_action)
        if self.subtick_queue is not None:
            self.subtick_queue.schedule_after_pass(state, pass_action)
            self.subtick_queue.reconcile(state.clock_seconds + self.cfg.dt_default, state)
        events.extend(pass_ev)
        record_policy_result("pass")
        return "pass", events
