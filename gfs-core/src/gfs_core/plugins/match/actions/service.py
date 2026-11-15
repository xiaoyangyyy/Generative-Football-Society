from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.contracts.micro_math import softmax
from gfs_core.contracts.micro_state import PlayerModulators

from .events import apply_micro_event
from .possession import (
    mark_shot_taken,
    shot_cooldown_ok,
    turnover_to_defence_after_shot,
)
from .sampling import mask_infeasible_actions, sample_action
from .wall import find_wall_partner

ACTION_LABELS = ("pass", "shot", "cross", "hold")
WIDE_ROLES = frozenset({"LW", "RW", "LB", "RB"})


@dataclass(frozen=True, slots=True)
class ActionSelection:
    choice: str
    probabilities: np.ndarray
    sampling_uniform: float
    feasible_actions: frozenset[str]
    planned_kind: str | None


class ActionService:
    def __init__(
        self, config: MicroMatchConfig, physics=None, ball_log=None, base_dir=".", spatial=None,
    ) -> None:
        self.config = config
        self.physics = physics
        self.ball_log = ball_log
        self.base_dir = base_dir
        self.spatial = spatial

    @staticmethod
    def create_player_tracker():
        from .stats import PlayerMatchStatsTracker

        return PlayerMatchStatsTracker()

    def create_substitution_service(self):
        from .substitutions import SubstitutionService

        return SubstitutionService(self.spatial)

    def create_passing_engine(self, spatial_intelligence, *, player_tracker=None):
        from .passing import PassingEngine

        engine = PassingEngine(
            self.config,
            spatial_intelligence,
            self.physics,
            base_dir=self.base_dir,
            ball_log=self.ball_log,
        )
        engine.player_tracker = player_tracker
        return engine

    def create_shot_engine(self, spatial_intelligence, *, player_tracker=None):
        from .shots import ShotEngine

        engine = ShotEngine(
            self.config, spatial_intelligence, self.physics, self.ball_log,
        )
        engine.player_tracker = player_tracker
        return engine

    def create_aerial_engine(self):
        from .aerial import AerialDuelEngine

        return AerialDuelEngine(self.config, self.physics)

    def create_executor(self, passing, shots, aerial):
        from .executor import ActionExecutor

        return ActionExecutor(self, passing, shots, aerial)

    apply_micro_event = staticmethod(apply_micro_event)
    shot_cooldown_ok = staticmethod(shot_cooldown_ok)
    mark_shot_taken = staticmethod(mark_shot_taken)
    turnover_to_defence_after_shot = staticmethod(turnover_to_defence_after_shot)

    def select(
        self,
        state,
        carrier,
        mod_home: list[PlayerModulators],
        mod_away: list[PlayerModulators],
        shots,
        rng: np.random.Generator,
        *,
        planned_kind: str | None = None,
    ) -> ActionSelection:
        attacking_home = carrier.team_id == state.home.team_id
        modulators = mod_home if attacking_home else mod_away
        modulator = next(
            (item for item in modulators if item.player_id == carrier.player_id),
            modulators[0] if modulators else PlayerModulators(carrier.player_id),
        )
        distance = shots.distance_to_goal(carrier.position, attacking_home)
        tactics = state.team(carrier.team_id).coach.tactical_current or {}
        low_block = float(tactics.get("low_block", 0.35))
        shot_utility, team_shots = self._shot_utility(
            state,
            carrier,
            modulator,
            shots,
            attacking_home=attacking_home,
            distance=distance,
            tactics=tactics,
            low_block=low_block,
        )
        utilities = np.array([
            self._pass_utility(state, carrier),
            shot_utility,
            self._cross_utility(carrier, distance),
            self._hold_utility(distance, low_block),
        ])
        utilities = np.nan_to_num(utilities, nan=0.0, posinf=5.0, neginf=-5.0)
        feasible = {"pass", "hold"}
        if distance < self.config.shot_max_dist and shot_cooldown_ok(
            state, attacking_home, self.config, float(state.clock_seconds),
        ):
            feasible.add("shot")
        if carrier.role in WIDE_ROLES:
            feasible.add("cross")
        probabilities = softmax(
            utilities,
            tau=max(0.2, self.config.action_tau * modulator.tau_dec),
        )
        probabilities = mask_infeasible_actions(
            ACTION_LABELS, probabilities, feasible,
        )
        uniform = float(rng.random())
        choice = sample_action(ACTION_LABELS, probabilities, uniform)
        if planned_kind in {"pass", "shot"}:
            choice = planned_kind
        return ActionSelection(
            choice,
            probabilities,
            uniform,
            frozenset(feasible),
            planned_kind,
        )

    def _shot_utility(
        self,
        state,
        carrier,
        modulator,
        shots,
        *,
        attacking_home,
        distance,
        tactics,
        low_block,
    ):
        utility = (
            self.config.action_shot_base
            + shots.shot_utility_max(state, carrier, modulator) * _shot_bias(tactics)
            + self.config.action_shot_dist_bonus
            * max(0.0, self.config.shot_max_dist - distance)
            * (1.0 - 0.55 * low_block)
        )
        side = "home_shots" if attacking_home else "away_shots"
        opponent = "away_shots" if attacking_home else "home_shots"
        team_shots = int(shots.stats.get(side, 0))
        opponent_shots = int(shots.stats.get(opponent, 0))
        defensive_block = float(np.clip((low_block - 0.50) / 0.45, 0.0, 1.0))
        if defensive_block > 0.05:
            utility *= float(np.exp(
                -self.config.action_shot_volume_decay * defensive_block * team_shots
            ))
            if team_shots >= 6 and opponent_shots <= max(2, team_shots // 4):
                utility *= float(np.exp(
                    -self.config.action_shot_skew_decay
                    * defensive_block
                    * (team_shots - opponent_shots)
                ))
        if team_shots + 2 <= opponent_shots and distance < self.config.shot_max_dist * 0.85:
            utility += 0.14 * min(3.0, (opponent_shots - team_shots) ** 0.5)
        return utility, team_shots

    def _pass_utility(self, state, carrier) -> float:
        utility = self.config.action_pass_base
        if not self.config.enable_wall_pass:
            return utility
        team = state.team(carrier.team_id)
        if find_wall_partner(carrier, team.players, self.config) is None:
            return utility
        shape = state.spatial.press.shape
        x = int(np.clip(carrier.position[0] * (shape[0] - 1), 0, shape[0] - 1))
        y = int(np.clip(carrier.position[1] * (shape[1] - 1), 0, shape[1] - 1))
        press = float(state.spatial.press[x, y])
        return utility + 0.12 + 0.2 * max(0.0, press - 0.35)

    def _cross_utility(self, carrier, distance) -> float:
        if carrier.role not in WIDE_ROLES:
            return 0.0
        if distance > 0.16:
            return (
                self.config.action_cross_base
                + 0.32 * float(carrier.abilities.curve)
                - 0.15 * distance
            )
        return self.config.action_cross_base * 0.65

    def _hold_utility(self, distance, low_block) -> float:
        return (
            -0.02
            + 0.04 * (1.0 - distance)
            + 0.10 * low_block
            * min(1.0, distance / max(1e-6, self.config.shot_max_dist))
        )


def _shot_bias(tactics) -> float:
    verticality = float(tactics.get("verticality", 0.5))
    risk = float(tactics.get("risk_budget", 0.5))
    low_block = float(tactics.get("low_block", 0.35))
    possession = float(tactics.get("possession_orientation", 0.5))
    counter = float(tactics.get("counter_attack", 0.45))
    attack = 0.52 * verticality + 0.30 * risk + 0.22 * counter * verticality
    defend = 0.38 * low_block + 0.22 * (1.0 - possession) * low_block
    return float(0.88 + attack - defend)
