from __future__ import annotations

import numpy as np

from gfs_core.contracts.events import MicroEventType
from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.contracts.trace import TraceTick
from gfs_core.kernel.identity import content_identity

from .replay import (
    checkpoint_payload,
    replay_target,
    request_payload,
    verify_checkpoint,
)
from .state import build_match_state, possession_prior
from .summary import build_summary, initial_tactical_execution


class MatchService:
    def __init__(self, *, agents, spatial, physics, actions, psychology, scoring) -> None:
        self.agents = agents
        self.spatial = spatial
        self.physics = physics
        self.actions = actions
        self.psychology = psychology
        self.scoring = scoring

    def run(
        self,
        home_profile,
        away_profile,
        *,
        goals_home: int,
        goals_away: int,
        xg_home: float,
        xg_away: float,
        referee=None,
        stage_pressure: float = 0.3,
        drama_score: float = 0.5,
        internal_home=None,
        internal_away=None,
        neutral_venue: bool = False,
        config: MicroMatchConfig | None = None,
        seed: int = 42,
        effective_home: float | None = None,
        effective_away: float | None = None,
        stage_name: str = "match",
        match_seconds: float | None = None,
        authoritative_clock: bool = False,
        allow_xg_supplement: bool = False,
        resume_checkpoint=None,
        checkpoint_sink=None,
        checkpoint_interval: int = 0,
        stop_at_tick: int | None = None,
        trace_sink=None,
        trace_match_id: str | None = None,
        memory=None,
        memory_as_of: int | None = None,
        data_identity=None,
    ):
        config = config or MicroMatchConfig()
        rng = np.random.default_rng(seed)
        effective_home = float(
            effective_home if effective_home is not None else home_profile.status_score
        )
        effective_away = float(
            effective_away if effective_away is not None else away_profile.status_score
        )
        possession_home = possession_prior(effective_home, effective_away, config)
        state, signals = build_match_state(
            home_profile,
            away_profile,
            self.agents,
            self.spatial,
            referee=referee,
            stage_pressure=stage_pressure,
            neutral_venue=neutral_venue,
            effective_home=effective_home,
            effective_away=effective_away,
            possession_home=possession_home,
            internal_home=internal_home,
            internal_away=internal_away,
            rng=rng,
        )
        if (
            memory is not None
            and resume_checkpoint is not None
            and memory_as_of is None
            and memory.snapshot()["records"]
        ):
            raise ValueError("memory_as_of is required when replaying with historical memory")
        if memory is not None:
            memory.apply_to_state(state, as_of=memory_as_of)
        affective = self.psychology.create_coupling(config)
        spatial = self.spatial.create_field(config)
        intelligence = self.spatial.create_intelligence(config)
        kinematic = self.physics.create_kinematics(
            config, spatial_intelligence=intelligence,
        )
        tracker = self.actions.create_player_tracker()
        passing = self.actions.create_passing_engine(
            intelligence, player_tracker=tracker,
        )
        shots = self.actions.create_shot_engine(
            intelligence, player_tracker=tracker,
        )
        aerial = self.actions.create_aerial_engine()
        executor = self.actions.create_executor(passing, shots, aerial)
        substitutions = self.actions.create_substitution_service()
        tactical_execution = {
            "home": initial_tactical_execution(state.home),
            "away": initial_tactical_execution(state.away),
        }

        dt = config.dt_default
        duration = float(match_seconds if match_seconds is not None else config.match_seconds)
        ticks = max(1, int(duration / dt))
        request = request_payload(home_profile, away_profile, config, signals, {
            "goals_home": goals_home,
            "goals_away": goals_away,
            "xg_home": xg_home,
            "xg_away": xg_away,
            "referee": referee,
            "stage_pressure": stage_pressure,
            "drama_score": drama_score,
            "neutral_venue": neutral_venue,
            "seed": seed,
            "effective_home": effective_home,
            "effective_away": effective_away,
            "stage_name": stage_name,
            "duration": duration,
            "authoritative_clock": authoritative_clock,
            "allow_xg_supplement": allow_xg_supplement,
            "memory_identity": content_identity(memory.snapshot()) if memory is not None else None,
            "memory_as_of": memory_as_of,
            "data_identity": data_identity,
        })
        target = replay_target(resume_checkpoint, request) if resume_checkpoint else 0
        if target > ticks:
            raise ValueError("micro checkpoint is beyond the configured match duration")
        if stop_at_tick is not None and (
            isinstance(stop_at_tick, bool)
            or not isinstance(stop_at_tick, int)
            or not target <= stop_at_tick <= ticks
        ):
            raise ValueError("micro checkpoint stop tick is invalid")
        if checkpoint_interval < 0:
            raise ValueError("micro checkpoint interval cannot be negative")
        referee_strictness = float((referee or {}).get("strictness", 0.55))
        schedule = self.psychology.build_event_schedule(
            state,
            goals_home=goals_home,
            goals_away=goals_away,
            xg_home=xg_home,
            xg_away=xg_away,
            referee_strictness=referee_strictness,
            drama_score=drama_score,
            internal_conflict_home=signals.conflict_home,
            internal_conflict_away=signals.conflict_away,
            rng=rng,
            match_seconds=duration,
            use_tick_fouls=config.discipline_tick_fouls,
            allow_scheduled_shots=False,
        )
        meso = self.psychology.create_meso(config.meso_window_seconds)
        timeline = []
        event_index = 0
        strictness_sum = 0.0
        possession_ticks = 0
        phi_home_sum = 0.0
        phi_away_sum = 0.0
        substitutions_done = {}
        clock_contract = (
            "authoritative_tick_v2" if authoritative_clock
            else "legacy_affective_offset_v1"
        )

        def runtime_state():
            return {
                "event_index": event_index,
                "strictness_sum": strictness_sum,
                "possession_ticks": possession_ticks,
                "phi_home_sum": phi_home_sum,
                "phi_away_sum": phi_away_sum,
                "substitutions_done": substitutions_done,
                "timeline": timeline,
                "tactical_execution": tactical_execution,
                "tracker_lines": tracker._lines,
                "tracker_substitutions": tracker.substitutions,
                "passing_stats": passing.stats,
                "shot_stats": shots.stats,
                "aerial_stats": aerial.stats,
                "meso_windows": meso._windows,
            }

        def checkpoint(next_tick, status):
            payload = checkpoint_payload(
                request, next_tick, state, runtime_state(), status=status,
            )
            if resume_checkpoint is not None and next_tick == target:
                verify_checkpoint(resume_checkpoint, next_tick, state, runtime_state())
            if checkpoint_sink is not None and next_tick >= target:
                checkpoint_sink(payload)
            return payload

        if target == 0 or stop_at_tick == 0 or (
            checkpoint_sink is not None and checkpoint_interval > 0
        ):
            initial = checkpoint(0, "interrupted" if stop_at_tick == 0 else "running")
            if stop_at_tick == 0:
                return initial

        match_id = trace_match_id or (
            f"{stage_name}:{home_profile.team_name}:{away_profile.team_name}:{seed}"
        )
        if trace_sink is not None:
            trace_sink(TraceTick(match_id, 0, 0.0, state, "start"))

        for tick in range(ticks):
            start = tick * dt
            end = start + dt
            state.clock_seconds = start if authoritative_clock else end
            tick_events, event_index = _events_in_tick(
                schedule, event_index, start, end,
            )
            press_home = float(
                state.home.coach.tactical_current.get("pressing_intensity", 0.5)
            )
            press_away = float(
                state.away.coach.tactical_current.get("pressing_intensity", 0.5)
            )
            self.psychology.step_in_match_pressure(state, dt)
            if config.enable_spatial:
                spatial.step(state, dt)
                intelligence.step(state)
            modifiers, scalars = affective.step(
                state,
                dt,
                events=_events_for_affective(tick_events, config),
                score_diff_home=state.home.score - state.away.score,
                xg_swing_home=xg_home - xg_away,
                coordination_home=signals.coordination_home,
                coordination_away=signals.coordination_away,
                conflict_home=signals.conflict_home,
                conflict_away=signals.conflict_away,
                press_home=press_home,
                press_away=press_away,
            )
            mod_home = modifiers["home"]
            mod_away = modifiers["away"]
            if authoritative_clock:
                state.clock_seconds = end
            if config.discipline_tick_fouls:
                foul_rate = float(scalars.get("lambda_foul", config.lambda_foul_base))
                discipline_events = self.psychology.emit_tick_discipline(
                    state,
                    mod_home,
                    mod_away,
                    cfg=config,
                    lambda_foul=foul_rate,
                    referee_strictness=referee_strictness,
                    drama_score=drama_score,
                    rng=rng,
                )
                tackle_events = self.psychology.emit_tick_tackle(
                    state,
                    mod_home,
                    mod_away,
                    cfg=config,
                    lambda_foul=foul_rate,
                    rng=rng,
                )
                cross_events = self.psychology.emit_tick_cross(
                    state, aerial, cfg=config, rng=rng,
                )
                for event in (*discipline_events, *tackle_events, *cross_events):
                    self.actions.apply_micro_event(state, event, config)
                    tick_events.append(event)
            strictness_sum += scalars["strictness_effective"]
            home_possession = state.ball.possession_team_id == state.home.team_id
            state.home.possession_share = (
                0.92 * state.home.possession_share + (0.08 if home_possession else 0.0)
            )
            state.away.possession_share = 1.0 - state.home.possession_share
            possession_ticks += int(home_possession)
            if config.enable_spatial:
                kinematic.step(state, mod_home, mod_away)
                phi_home_sum += float(np.mean(state.spatial.phi_home))
                phi_away_sum += float(np.mean(state.spatial.phi_away))

            for event in tick_events:
                tracker.apply_event(event)
            action = "hold"
            action_events = []
            if config.enable_phase3:
                if tick_events:
                    _resolve_scheduled_shots(
                        state, tick_events, shots, config, mod_home, mod_away,
                        self.actions, rng,
                    )
                action, action_events = executor.step(
                    state, mod_home, mod_away, rng,
                )
                for event in action_events:
                    tracker.apply_event(event)
                _apply_phase3_events(
                    state, action_events, config, self.actions,
                )
            elif config.enable_passing:
                _, pass_events = passing.step(state, mod_home, mod_away, rng)
                action = "pass"
                action_events = list(pass_events)
                for event in pass_events:
                    self.actions.apply_micro_event(state, event, config)

            tracker.tick_minutes(state, dt)
            timeline.extend(substitutions.maybe_apply(
                state,
                end,
                rng,
                tracker,
                subs_done=substitutions_done,
                disabled_team_ids=set(),
            ))
            _sync_ball_to_possessor(state)
            _record_tick_meso(
                state, affective, meso, xg_home, xg_away,
            )
            timeline.extend(
                f"{int(start // 60)}' {event.event_type.value} ({event.team_id})"
                for event in tick_events
                if event.event_type.value in {
                    "goal_scored", "red_card", "var_controversy",
                }
            )
            if trace_sink is not None:
                trace_sink(TraceTick(
                    match_id,
                    tick + 1,
                    dt,
                    state,
                    action,
                    (*tick_events, *action_events),
                ))
            next_tick = tick + 1
            should_checkpoint = (
                next_tick == target
                or next_tick == stop_at_tick
                or (checkpoint_sink is not None and next_tick == ticks)
                or (
                    checkpoint_sink is not None
                    and checkpoint_interval > 0
                    and next_tick % checkpoint_interval == 0
                )
            )
            if should_checkpoint:
                status = (
                    "completed" if next_tick == ticks
                    else "interrupted" if next_tick == stop_at_tick
                    else "running"
                )
                payload = checkpoint(next_tick, status)
                if next_tick == stop_at_tick:
                    return payload

        packets = meso.flush_packets(state.home.team_id, state.away.team_id)
        emotion_home = affective.team_emotion_mean(state.home)
        emotion_away = affective.team_emotion_mean(state.away)
        physics_home = int(
            shots.stats["home_goals"] + aerial.stats["home_aerial_goals"]
        )
        physics_away = int(
            shots.stats["away_goals"] + aerial.stats["away_aerial_goals"]
        )
        supplement_meta = {}
        if config.physics_goals:
            final_home, final_away = physics_home, physics_away
            if config.score_mode == "hybrid" and allow_xg_supplement:
                final_home, final_away, supplement_meta = self.scoring.supplement_goals(
                    physics_home,
                    physics_away,
                    state.micro_xg_home,
                    state.micro_xg_away,
                    config,
                    rng=np.random.default_rng(seed + 909),
                )
        else:
            final_home, final_away = int(goals_home), int(goals_away)
        state.home.score = final_home
        state.away.score = final_away
        summary = build_summary(
            state=state,
            config=config,
            scoring=self.scoring,
            passing=passing,
            shots=shots,
            aerial=aerial,
            tracker=tracker,
            packets=packets,
            timeline=timeline,
            ticks=ticks,
            strictness_sum=strictness_sum,
            possession_ticks=possession_ticks,
            phi_home_sum=phi_home_sum,
            phi_away_sum=phi_away_sum,
            emotion_home=emotion_home,
            emotion_away=emotion_away,
            effective_home=effective_home,
            effective_away=effective_away,
            final_home=final_home,
            final_away=final_away,
            physics_home=physics_home,
            physics_away=physics_away,
            supplement_meta=supplement_meta,
            tactical_execution=tactical_execution,
            clock_contract=clock_contract,
            duration=duration,
        )
        if self.actions.ball_log is not None:
            summary.ball_log_path = self.actions.ball_log.persist(
                state,
                home=state.home.team_id,
                away=state.away.team_id,
                stage=stage_name,
            )
        return summary


def _events_in_tick(schedule, index, start, end):
    events = []
    while index < len(schedule) and schedule[index].t_sec < end:
        if schedule[index].t_sec >= start:
            events.append(schedule[index])
        index += 1
    return events, index


def _events_for_affective(events, config):
    if not config.enable_phase3:
        return events
    skipped = {MicroEventType.SHOT_ON_TARGET, MicroEventType.SHOT_OFF_TARGET}
    if config.physics_goals:
        skipped |= {MicroEventType.GOAL_SCORED, MicroEventType.GOAL_CONCEDED}
    return [event for event in events if event.event_type not in skipped]


def _apply_phase3_events(state, events, config, actions):
    for event in events:
        if not config.physics_goals and event.event_type in {
            MicroEventType.GOAL_SCORED, MicroEventType.GOAL_CONCEDED,
        }:
            continue
        actions.apply_micro_event(state, event, config)


def _resolve_scheduled_shots(
    state, events, shots, config, mod_home, mod_away, actions, rng,
):
    for event in events:
        if event.event_type not in {
            MicroEventType.SHOT_ON_TARGET, MicroEventType.SHOT_OFF_TARGET,
        }:
            continue
        team = state.team(event.team_id)
        attacking_home = event.team_id == state.home.team_id
        pool = [player for player in team.players if player.on_pitch and player.role != "GK"]
        if not pool:
            continue
        carrier = pool[int(rng.integers(0, len(pool)))]
        if event.player_id:
            for player in pool:
                if player.player_id == event.player_id:
                    carrier = player
                    break
        scheduled_on_target = event.event_type == MicroEventType.SHOT_ON_TARGET
        if scheduled_on_target:
            carrier.position[0] = (
                config.scheduled_shot_box_x_home
                if attacking_home else config.scheduled_shot_box_x_away
            )
            spread = float(config.scheduled_shot_box_y_spread)
            carrier.position[1] = float(np.clip(
                0.5 + (rng.random() - 0.5) * spread, 0.10, 0.90,
            ))
        state.ball.possessor_id = carrier.player_id
        state.ball.position = carrier.position.copy()
        modulators = mod_home if attacking_home else mod_away
        modulator = next(
            (item for item in modulators if item.player_id == carrier.player_id),
            modulators[0],
        )
        kind = "curved" if rng.random() < 0.2 else "driven"
        now = float(state.clock_seconds)
        if not actions.shot_cooldown_ok(state, attacking_home, config, now):
            continue
        outcome = shots.resolve_shot(
            state,
            carrier,
            modulator,
            rng,
            forced_kind=kind,
            scheduled_on_target=scheduled_on_target,
            from_schedule=True,
        )
        actions.mark_shot_taken(state, attacking_home, now)
        if not outcome.goal:
            defending = state.away.team_id if attacking_home else state.home.team_id
            goalkeeper = shots._gk_player(state, defending)
            actions.turnover_to_defence_after_shot(
                state, attacking_home, goalkeeper, rng,
            )
        _apply_phase3_events(state, outcome.events, config, actions)


def _sync_ball_to_possessor(state):
    if not state.ball.possessor_id:
        return
    for player in state.home.players + state.away.players:
        if player.player_id == state.ball.possessor_id:
            state.ball.position = 0.85 * state.ball.position + 0.15 * player.position
            return


def _record_tick_meso(state, affective, meso, xg_home, xg_away):
    emotion_home = affective.team_emotion_mean(state.home)
    emotion_away = affective.team_emotion_mean(state.away)
    for team, xg_for, xg_against, emotion in (
        (state.home, xg_home, xg_away, emotion_home),
        (state.away, xg_away, xg_home, emotion_away),
    ):
        meso.record_tick(
            state,
            team_id=team.team_id,
            xg_for=xg_for,
            xg_against=xg_against,
            controversy=state.referee.controversy_integral,
            psi=state.crowd.psi,
            emotion=emotion,
            tactical=team.coach.tactical_current,
            tactical_base=team.coach.tactical_base,
            icon_shock=_icon_shock(team),
            phi=float(np.mean(
                state.spatial.phi_home if team is state.home else state.spatial.phi_away
            )),
        )


def _icon_shock(team):
    if not team.icon_player_id:
        return 0.0
    for player in team.players:
        if player.player_id == team.icon_player_id:
            emotions = player.emotion_profile()
            return float(
                emotions.get("pride", 0)
                - emotions.get("fear", 0)
                + 0.5 * emotions.get("anger", 0)
            )
    return 0.0
