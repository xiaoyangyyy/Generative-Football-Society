"""Unified action selection: pass / shot / cross / hold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from src.match_engine.math_utils import softmax
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEvent
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.aerial_duel import AerialDuelEngine
from src.match_engine.state import (
    MatchAffectiveState,
    PlayerAffectiveState,
    PlayerModulators,
)
from src.match_engine.tactical_engine import TacticalMicroEngine
from src.match_engine.possession_helpers import (
    mark_shot_taken,
    shot_cooldown_ok,
    turnover_to_defence_after_shot,
)
from src.match_engine.wall_pass import find_wall_partner


_ACTION_LABELS = ("pass", "shot", "cross", "hold")
_WIDE_ROLES = frozenset({"LW", "RW", "LB", "RB"})


@dataclass(frozen=True)
class _ActionContext:
    carrier: PlayerAffectiveState
    attacking_home: bool
    modulator: PlayerModulators
    distance_to_goal: float
    tactics: dict[str, Any]
    team_shots: int
    feasible_actions: set[str]
    temperature: float


@dataclass(frozen=True)
class _SampledAction:
    choice: str
    counterfactual_baseline_action: str
    baseline_probabilities: np.ndarray
    adjusted_probabilities: np.ndarray
    sampling_uniform: float
    policy_intent: dict[str, Any] | None
    policy_outcome_baseline: dict[str, Any] | None
    planned_kind: str | None


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

    def _consume_scheduled_kind(
        self, state: MatchAffectiveState
    ) -> tuple[bool, str | None]:
        if self.continuous_clock is None:
            return False, None
        self.continuous_clock.observe(state)
        if self.continuous_clock.is_gated(state.clock_seconds):
            return True, None
        return False, self.continuous_clock.pop_due_kind(state.clock_seconds)

    def _build_action_context(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
    ) -> tuple[_ActionContext, np.ndarray]:
        attacking_home = carrier.team_id == state.home.team_id
        mod_list = mod_home if attacking_home else mod_away
        mod_map = {mod.player_id: mod for mod in mod_list}
        modulator = mod_map.get(carrier.player_id, mod_list[0])
        distance_to_goal = self.shots._dist_to_goal(carrier.position, attacking_home)
        tactics = state.team(carrier.team_id).coach.tactical_current or {}
        low_block = float(tactics.get("low_block", 0.35))
        shot_utility, team_shots = self._shot_utility(
            state,
            carrier,
            modulator,
            attacking_home=attacking_home,
            distance_to_goal=distance_to_goal,
            tactics=tactics,
            low_block=low_block,
        )
        utilities = np.array(
            [
                self._pass_utility(state, carrier),
                shot_utility,
                self._cross_utility(carrier, distance_to_goal),
                self._hold_utility(distance_to_goal, low_block),
            ],
            dtype=float,
        )
        utilities = self._apply_hierarchical_bias(utilities, carrier, tactics)
        utilities = np.nan_to_num(utilities, nan=0.0, posinf=5.0, neginf=-5.0)
        feasible_actions = {"pass", "hold"}
        if distance_to_goal < self.cfg.shot_max_dist and shot_cooldown_ok(
            state,
            attacking_home,
            self.cfg,
            float(state.clock_seconds),
        ):
            feasible_actions.add("shot")
        if carrier.role in _WIDE_ROLES:
            feasible_actions.add("cross")
        return (
            _ActionContext(
                carrier=carrier,
                attacking_home=attacking_home,
                modulator=modulator,
                distance_to_goal=distance_to_goal,
                tactics=tactics,
                team_shots=team_shots,
                feasible_actions=feasible_actions,
                temperature=self.cfg.action_tau * modulator.tau_dec,
            ),
            utilities,
        )

    def _shot_utility(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        modulator: PlayerModulators,
        *,
        attacking_home: bool,
        distance_to_goal: float,
        tactics: dict[str, Any],
        low_block: float,
    ) -> tuple[float, int]:
        utility = (
            self.cfg.action_shot_base
            + self.shots.shot_utility_max(state, carrier, modulator)
            * self._tac_eng.shot_bias(tactics)
            + self.cfg.action_shot_dist_bonus
            * max(0.0, self.cfg.shot_max_dist - distance_to_goal)
            * (1.0 - 0.55 * low_block)
        )
        side_key = "home_shots" if attacking_home else "away_shots"
        opponent_key = "away_shots" if attacking_home else "home_shots"
        team_shots = int(self.shots.stats.get(side_key, 0))
        opponent_shots = int(self.shots.stats.get(opponent_key, 0))
        defensive_block = float(np.clip((low_block - 0.50) / 0.45, 0.0, 1.0))
        if defensive_block > 0.05:
            utility *= float(
                np.exp(
                    -self.cfg.action_shot_volume_decay * defensive_block * team_shots
                )
            )
            if team_shots >= 6 and opponent_shots <= max(2, team_shots // 4):
                utility *= float(
                    np.exp(
                        -self.cfg.action_shot_skew_decay
                        * defensive_block
                        * (team_shots - opponent_shots)
                    )
                )
        if (
            team_shots + 2 <= opponent_shots
            and distance_to_goal < self.cfg.shot_max_dist * 0.85
        ):
            utility += 0.14 * min(3.0, (opponent_shots - team_shots) ** 0.5)
        return utility, team_shots

    def _pass_utility(
        self, state: MatchAffectiveState, carrier: PlayerAffectiveState
    ) -> float:
        utility = self.cfg.action_pass_base
        if not self.cfg.enable_wall_pass:
            return utility
        team = state.team(carrier.team_id)
        if find_wall_partner(carrier, team.players, self.cfg) is None:
            return utility
        press = float(
            state.spatial.press[
                int(
                    np.clip(
                        carrier.position[0] * (state.spatial.press.shape[0] - 1),
                        0,
                        state.spatial.press.shape[0] - 1,
                    )
                ),
                int(
                    np.clip(
                        carrier.position[1] * (state.spatial.press.shape[1] - 1),
                        0,
                        state.spatial.press.shape[1] - 1,
                    )
                ),
            ]
        )
        return utility + 0.12 + 0.2 * max(0.0, press - 0.35)

    def _cross_utility(
        self, carrier: PlayerAffectiveState, distance_to_goal: float
    ) -> float:
        if carrier.role not in _WIDE_ROLES:
            return 0.0
        if distance_to_goal > 0.16:
            return (
                self.cfg.action_cross_base
                + 0.32 * float(carrier.abilities.curve)
                - 0.15 * distance_to_goal
            )
        return self.cfg.action_cross_base * 0.65

    def _hold_utility(self, distance_to_goal: float, low_block: float) -> float:
        return (
            -0.02
            + 0.04 * (1.0 - distance_to_goal)
            + 0.10
            * low_block
            * min(
                1.0,
                distance_to_goal / max(1e-6, self.cfg.shot_max_dist),
            )
        )

    def _apply_hierarchical_bias(
        self,
        utilities: np.ndarray,
        carrier: PlayerAffectiveState,
        tactics: dict[str, Any],
    ) -> np.ndarray:
        if self._hierarchical is None:
            return utilities
        options = self._hierarchical.decompose(tactics)
        if carrier.role in {"GK", "CB", "LB", "RB"}:
            role_unit = "back_line"
        elif carrier.role in {"DM", "CM", "AM", "LM", "RM"}:
            role_unit = "midfield"
        else:
            role_unit = "front_line"
        option = next(item for item in options if item.unit == role_unit)
        intent = self._hierarchical.player_intent(option, carrier.role)
        utilities[0] += 0.08 * float(intent.get("support", 0.0))
        utilities[1] += 0.08 * float(intent.get("risk", 0.0))
        utilities[3] += 0.05 * float(intent.get("cover", 0.0))
        return utilities

    def _apply_coach_bias(
        self,
        probabilities: np.ndarray,
        *,
        policy_intent: dict[str, Any] | None,
        feasible_actions: set[str],
        temperature: float,
    ) -> np.ndarray:
        biased = probabilities.copy()
        if policy_intent is None or float(policy_intent["logit_bias"]) <= 0.0:
            return biased
        from src.match_engine.world_model.action_adoption import (
            mask_infeasible_action_probabilities,
        )

        selected_index = _ACTION_LABELS.index(policy_intent["action"])
        biased[selected_index] *= float(
            np.exp(min(0.5, float(policy_intent["logit_bias"])) / max(0.2, temperature))
        )
        return mask_infeasible_action_probabilities(
            _ACTION_LABELS, biased, feasible_actions
        )

    def _sample_action(
        self,
        state: MatchAffectiveState,
        context: _ActionContext,
        utilities: np.ndarray,
        planned_kind: str | None,
        rng: np.random.Generator,
    ) -> _SampledAction:
        from src.match_engine.world_model.action_adoption import (
            mask_infeasible_action_probabilities,
            sample_action_from_uniform,
        )
        from src.match_engine.world_model.decision_adoption import (
            pending_policy_action_bias,
        )
        from src.match_engine.world_model.policy_outcomes import (
            capture_policy_outcome_baseline,
        )

        labels = list(_ACTION_LABELS)
        pre_world_model_utilities = utilities.copy()
        if self.wm_runtime is not None:
            from src.match_engine.world_model.config import (
                world_model_controls_side,
            )
            from src.match_engine.world_model.planner import (
                action_imagination_adjustments,
            )

            if world_model_controls_side(context.attacking_home):
                # This computes and registers auditable planner deltas. The
                # returned utility vector is deliberately not sampled: direct
                # authority is applied once by the probability controller.
                action_imagination_adjustments(
                    self.wm_runtime,
                    state,
                    context.carrier,
                    context.attacking_home,
                    utilities,
                    labels,
                    dist_goal=context.distance_to_goal,
                    tac=context.tactics,
                    team_shots=context.team_shots,
                    feasible_actions=context.feasible_actions,
                    temperature=context.temperature,
                )
        policy_intent = pending_policy_action_bias(
            state,
            team_id=context.carrier.team_id,
            feasible_actions=context.feasible_actions,
            t_sec=float(state.clock_seconds),
        )
        policy_outcome_baseline = (
            capture_policy_outcome_baseline(state, team_id=context.carrier.team_id)
            if policy_intent is not None
            else None
        )
        base_probabilities = softmax(
            pre_world_model_utilities,
            tau=max(0.2, context.temperature),
        )
        base_probabilities = mask_infeasible_action_probabilities(
            labels, base_probabilities, context.feasible_actions
        )
        # The probability controller is the single high-level adoption point.
        # Planner utility deltas remain auditable evidence and must not be
        # applied a second time before the bounded policy blend.
        direct_world_model_probabilities = base_probabilities.copy()
        if self.wm_runtime is not None:
            from src.match_engine.world_model.action_adoption import (
                mix_direct_action_probabilities,
            )

            direct_world_model_probabilities = mix_direct_action_probabilities(
                state,
                labels=labels,
                probabilities=direct_world_model_probabilities,
            )
        baseline_sampling_probabilities = self._apply_coach_bias(
            base_probabilities,
            policy_intent=policy_intent,
            feasible_actions=context.feasible_actions,
            temperature=context.temperature,
        )
        adjusted_probabilities = self._apply_coach_bias(
            direct_world_model_probabilities,
            policy_intent=policy_intent,
            feasible_actions=context.feasible_actions,
            temperature=context.temperature,
        )
        sampling_uniform = float(rng.random())
        counterfactual_baseline_action = sample_action_from_uniform(
            labels, baseline_sampling_probabilities, sampling_uniform
        )
        choice = sample_action_from_uniform(
            labels, adjusted_probabilities, sampling_uniform
        )
        if planned_kind in {"pass", "shot"}:
            choice = planned_kind
            counterfactual_baseline_action = planned_kind
        return _SampledAction(
            choice=choice,
            counterfactual_baseline_action=counterfactual_baseline_action,
            baseline_probabilities=baseline_sampling_probabilities,
            adjusted_probabilities=adjusted_probabilities,
            sampling_uniform=sampling_uniform,
            policy_intent=policy_intent,
            policy_outcome_baseline=policy_outcome_baseline,
            planned_kind=planned_kind,
        )

    def _record_policy_result(
        self,
        state: MatchAffectiveState,
        sampled: _SampledAction,
        actual_action: str,
    ) -> None:
        if sampled.policy_intent is not None:
            from src.match_engine.world_model.decision_adoption import (
                record_policy_intervention_result,
            )

            record_policy_intervention_result(
                state,
                decision_id=sampled.policy_intent["decision_id"],
                actual_action=actual_action,
                t_sec=float(state.clock_seconds),
                outcome_baseline=sampled.policy_outcome_baseline,
            )
        if self.wm_runtime is None:
            return
        from src.match_engine.world_model.action_adoption import (
            record_action_policy_sample,
        )

        record_action_policy_sample(
            state,
            actual_action=actual_action,
            labels=_ACTION_LABELS,
            base_probabilities=sampled.baseline_probabilities,
            adjusted_probabilities=sampled.adjusted_probabilities,
            sampling_uniform=sampled.sampling_uniform,
            counterfactual_baseline_action=(sampled.counterfactual_baseline_action),
            externally_overridden=sampled.planned_kind in {"pass", "shot"},
            cointervention=bool(
                sampled.policy_intent is not None
                and float(sampled.policy_intent["logit_bias"]) > 0.0
            ),
        )

    @staticmethod
    def _encode_world_model_action(
        state: MatchAffectiveState,
        action: str,
        target: np.ndarray,
    ) -> None:
        if not hasattr(state, "_wm_last_action"):
            return
        from src.match_engine.world_model.action_codec import (
            encode_high_level_action,
        )

        state._wm_last_action = encode_high_level_action(
            action,
            target,
            horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
        )

    def _execute_pass(
        self,
        state: MatchAffectiveState,
        sampled: _SampledAction,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        pass_action, pass_events = self.passing.step(state, mod_home, mod_away, rng)
        if self.continuous_clock is not None:
            self.continuous_clock.schedule_after_pass(state, pass_action)
        if self.subtick_queue is not None:
            self.subtick_queue.schedule_after_pass(state, pass_action)
            self.subtick_queue.reconcile(
                state.clock_seconds + self.cfg.dt_default, state
            )
        self._record_policy_result(state, sampled, "pass")
        return "pass", list(pass_events)

    def _execute_shot(
        self,
        state: MatchAffectiveState,
        context: _ActionContext,
        sampled: _SampledAction,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        self._encode_world_model_action(state, "shot", context.carrier.position)
        now = float(state.clock_seconds)
        if not shot_cooldown_ok(state, context.attacking_home, self.cfg, now):
            return self._execute_pass(state, sampled, mod_home, mod_away, rng)
        outcome = self.shots.resolve_shot(
            state,
            context.carrier,
            context.modulator,
            rng,
            from_schedule=False,
        )
        if getattr(state, "_wm_last_action", None) is not None:
            from src.match_engine.world_model.schema import (
                SHOT_GOAL_INDEX,
                SHOT_ON_TARGET_INDEX,
            )

            state._wm_last_action[13] = float(np.clip(outcome.xg, 0.0, 1.0))
            state._wm_last_action[SHOT_GOAL_INDEX] = 1.0 if outcome.goal else 0.0
            state._wm_last_action[SHOT_ON_TARGET_INDEX] = (
                1.0 if outcome.on_target else 0.0
            )
        mark_shot_taken(state, context.attacking_home, now)
        if outcome.goal:
            state.ball.position = outcome.traj.landed
            state.ball.possessor_id = None
        else:
            defending_team_id = (
                state.away.team_id if context.attacking_home else state.home.team_id
            )
            goalkeeper = self.shots._gk_player(state, defending_team_id)
            turnover_to_defence_after_shot(
                state, context.attacking_home, goalkeeper, rng
            )
        self._record_policy_result(state, sampled, "shot")
        return "shot", list(outcome.events)

    @staticmethod
    def _cross_observation(
        state: MatchAffectiveState, attacking_home: bool
    ) -> np.ndarray | None:
        from src.match_engine.ball_path_logger import (
            ball_log_wm_snapshot_enabled,
        )

        if not ball_log_wm_snapshot_enabled():
            return None
        from src.match_engine.world_model.observation import encode_observation

        return encode_observation(state, attacking_home=attacking_home)

    @staticmethod
    def _apply_cross_winner(
        state: MatchAffectiveState, winner_id: str | None
    ) -> PlayerAffectiveState | None:
        if not winner_id:
            return None
        for player in state.home.players + state.away.players:
            if player.player_id == winner_id:
                state.ball.possessor_id = player.player_id
                state.ball.possession_team_id = player.team_id
                return player
        return None

    def _execute_cross(
        self,
        state: MatchAffectiveState,
        context: _ActionContext,
        sampled: _SampledAction,
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        from src.match_engine.aerial_duel import (
            aerial_goal_events,
            apply_aerial_xg_to_state,
        )
        from src.match_engine.ball_path_logger import record_cross

        self._encode_world_model_action(state, "cross", context.carrier.position)
        observation_pre = self._cross_observation(state, context.attacking_home)
        trajectory, aerial_outcome = self.aerial.resolve_cross(
            state, context.carrier, rng
        )
        state.ball.position = aerial_outcome.landed_xy
        apply_aerial_xg_to_state(
            state,
            aerial_outcome.xg_added,
            context.attacking_home,
            self.cfg,
        )
        events = aerial_goal_events(state, context.carrier, aerial_outcome)
        winner = self._apply_cross_winner(state, aerial_outcome.winner_id)
        if getattr(state, "_wm_last_action", None) is not None:
            state._wm_last_action[6:8] = np.clip(aerial_outcome.landed_xy, 0.0, 1.0)
        observation_post = self._cross_observation(state, context.attacking_home)
        record_cross(
            state,
            carrier=context.carrier,
            landed=aerial_outcome.landed_xy,
            contact=aerial_outcome.contact,
            winner=winner,
            xg_added=aerial_outcome.xg_added,
            goal=aerial_outcome.goal,
            traj_peak=float(getattr(trajectory, "peak_height", 0.0)),
            traj_tof=float(getattr(trajectory, "time_of_flight", 0.0)),
            obs_pre=observation_pre,
            obs_post=observation_post,
        )
        self._record_policy_result(state, sampled, "cross")
        return "cross", list(events)

    def _execute_hold(
        self,
        state: MatchAffectiveState,
        sampled: _SampledAction,
    ) -> Tuple[str, List[MicroEvent]]:
        self._encode_world_model_action(state, "hold", state.ball.position)
        self._record_policy_result(state, sampled, "hold")
        return "hold", []

    def _execute_sampled_action(
        self,
        state: MatchAffectiveState,
        context: _ActionContext,
        sampled: _SampledAction,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        if (
            sampled.choice == "shot"
            and context.distance_to_goal < self.cfg.shot_max_dist
        ):
            return self._execute_shot(state, context, sampled, mod_home, mod_away, rng)
        if sampled.choice == "cross" and context.carrier.role in _WIDE_ROLES:
            return self._execute_cross(state, context, sampled, rng)
        if sampled.choice == "hold":
            return self._execute_hold(state, sampled)
        return self._execute_pass(state, sampled, mod_home, mod_away, rng)

    def step(
        self,
        state: MatchAffectiveState,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[str, List[MicroEvent]]:
        if self.subtick_queue is not None:
            self.subtick_queue.capture_pre_action(state)
        clock_gated, planned_kind = self._consume_scheduled_kind(state)
        if clock_gated:
            return "hold", []
        carrier = self.passing._get_carrier(state)
        if carrier is None:
            self.passing._assign_random_possession(state, rng)
            return "kickoff", []

        context, utils = self._build_action_context(state, carrier, mod_home, mod_away)
        sampled = self._sample_action(state, context, utils, planned_kind, rng)
        return self._execute_sampled_action(
            state, context, sampled, mod_home, mod_away, rng
        )
