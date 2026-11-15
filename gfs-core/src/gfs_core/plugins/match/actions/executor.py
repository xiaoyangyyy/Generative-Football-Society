from __future__ import annotations

from .aerial import aerial_goal_events, apply_aerial_xg_to_state
from .possession import mark_shot_taken, shot_cooldown_ok, turnover_to_defence_after_shot
from .service import WIDE_ROLES


class ActionExecutor:
    def __init__(self, service, passing, shots, aerial) -> None:
        self.service = service
        self.passing = passing
        self.shots = shots
        self.aerial = aerial

    def step(self, state, mod_home, mod_away, rng, *, planned_kind=None):
        carrier = self.passing._get_carrier(state)
        if carrier is None:
            self.passing._assign_random_possession(state, rng)
            return "kickoff", []
        selection = self.service.select(
            state,
            carrier,
            mod_home,
            mod_away,
            self.shots,
            rng,
            planned_kind=planned_kind,
        )
        attacking_home = carrier.team_id == state.home.team_id
        distance = self.shots.distance_to_goal(carrier.position, attacking_home)
        if selection.choice == "shot" and distance < self.service.config.shot_max_dist:
            return self._shot(state, carrier, attacking_home, mod_home, mod_away, rng)
        if selection.choice == "cross" and carrier.role in WIDE_ROLES:
            return self._cross(state, carrier, attacking_home, rng)
        if selection.choice == "hold":
            return "hold", []
        action, events = self.passing.step(state, mod_home, mod_away, rng)
        return "pass", list(events)

    def _shot(self, state, carrier, attacking_home, mod_home, mod_away, rng):
        now = float(state.clock_seconds)
        if not shot_cooldown_ok(
            state, attacking_home, self.service.config, now,
        ):
            action, events = self.passing.step(state, mod_home, mod_away, rng)
            return "pass", list(events)
        modulators = mod_home if attacking_home else mod_away
        modulator = next(
            (item for item in modulators if item.player_id == carrier.player_id),
            modulators[0] if modulators else None,
        )
        if modulator is None:
            from gfs_core.contracts.micro_state import PlayerModulators

            modulator = PlayerModulators(carrier.player_id)
        outcome = self.shots.resolve_shot(
            state, carrier, modulator, rng, from_schedule=False,
        )
        mark_shot_taken(state, attacking_home, now)
        if outcome.goal:
            state.ball.position = outcome.traj.landed
            state.ball.possessor_id = None
        else:
            defending = state.away.team_id if attacking_home else state.home.team_id
            goalkeeper = self.shots._gk_player(state, defending)
            turnover_to_defence_after_shot(
                state, attacking_home, goalkeeper, rng,
            )
        return "shot", list(outcome.events)

    def _cross(self, state, carrier, attacking_home, rng):
        trajectory, outcome = self.aerial.resolve_cross(state, carrier, rng)
        state.ball.position = outcome.landed_xy
        apply_aerial_xg_to_state(
            state,
            outcome.xg_added,
            attacking_home,
            self.service.config,
        )
        events = aerial_goal_events(state, carrier, outcome)
        winner = None
        if outcome.winner_id:
            for player in state.home.players + state.away.players:
                if player.player_id == outcome.winner_id:
                    winner = player
                    state.ball.possessor_id = player.player_id
                    state.ball.possession_team_id = player.team_id
                    break
        if self.service.ball_log is not None:
            self.service.ball_log.record_cross(
                state,
                carrier=carrier,
                landed=outcome.landed_xy,
                contact=outcome.contact,
                winner=winner,
                xg_added=outcome.xg_added,
                goal=outcome.goal,
                trajectory=trajectory,
            )
        return "cross", list(events)
