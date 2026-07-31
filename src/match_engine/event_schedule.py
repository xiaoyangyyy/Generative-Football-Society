"""
Stochastic micro-event schedule conditioned on match score / xG / drama.

Produces a realistic minute-by-minute event stream to drive Phase 1b ODEs
without requiring full spatial simulation yet.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.match_engine.discipline_schedule import (
    baseline_foul_modulators,
    expected_match_fouls,
    pick_foul_committing_team,
    pick_weighted_outfield,
    team_foul_weight,
)
from src.match_engine.micro_events import MicroEvent, MicroEventType
from src.match_engine.squad_factory import pick_random_outfield_player
from src.match_engine.state import MatchAffectiveState


def _finite_xg(value: float, default: float = 1.0) -> float:
    value = float(value)
    return default if not np.isfinite(value) else float(np.clip(value, 0.05, 3.6))


def _build_schedule_times(
    state, *, goals_home, goals_away, xg_home, xg_away,
    referee_strictness, drama_score, match_seconds, use_tick_fouls, rng,
):
    goals_home = int(max(0, goals_home))
    goals_away = int(max(0, goals_away))
    goal_times_home = _spread_times(goals_home, match_seconds, rng)
    goal_times_away = _spread_times(goals_away, match_seconds, rng)
    all_goals = sorted(
        [(time, "home") for time in goal_times_home]
        + [(time, "away") for time in goal_times_away]
    )
    from src.simulation.score_path import scheduled_shots_allowed

    if not scheduled_shots_allowed():
        shots_home, shots_away = goals_home, goals_away
    else:
        from src.match_engine.micro_config import MicroMatchConfig

        config = MicroMatchConfig()
        multiplier = float(getattr(config, "schedule_shots_xg_mult", 1.35))
        poisson_rate = float(getattr(config, "schedule_shots_poisson", 0.6))
        cap = int(getattr(config, "max_scheduled_shots_per_team", 16))
        shots_home = min(
            cap, max(goals_home, int(round(xg_home * multiplier + rng.poisson(poisson_rate)))),
        )
        shots_away = min(
            cap, max(goals_away, int(round(xg_away * multiplier + rng.poisson(poisson_rate)))),
        )
    shot_times_home = _spread_times(shots_home, match_seconds, rng)
    shot_times_away = _spread_times(shots_away, match_seconds, rng)

    mod_home, mod_away = baseline_foul_modulators(state)
    foul_weight_home = team_foul_weight(state.home, mod_home)
    foul_weight_away = team_foul_weight(state.away, mod_away)
    foul_times, yellow_times, red_times = [], [], []
    if not use_tick_fouls:
        foul_count = expected_match_fouls(
            foul_weight_home, foul_weight_away, referee_strictness, rng,
        )
        foul_times = _spread_times(foul_count, match_seconds, rng)
        card_pressure = 0.5 * (foul_weight_home + foul_weight_away)
        yellow_count = int(min(6, max(0, rng.poisson(
            2.0 + referee_strictness + 0.35 * (card_pressure - 1.0)
        ))))
        yellow_times = _spread_times(yellow_count, match_seconds, rng)
        red_count = 1 if rng.random() < 0.008 + 0.02 * drama_score else 0
        red_times = _spread_times(red_count, match_seconds, rng)
    return {
        "all_goals": all_goals,
        "shots_home": shot_times_home, "shots_away": shot_times_away,
        "fouls": foul_times, "yellows": yellow_times, "reds": red_times,
        "mod_home": mod_home, "mod_away": mod_away,
        "foul_weight_home": foul_weight_home,
        "foul_weight_away": foul_weight_away,
    }


def _append_atmosphere_events(
    events, state, *, drama_score, conflict_home, conflict_away,
    match_seconds, rng,
):
    if rng.random() < 0.12 + 0.2 * drama_score:
        events.append(MicroEvent(
            t_sec=float(rng.uniform(0.25, 0.85) * match_seconds),
            event_type=MicroEventType.VAR_CONTROVERSY,
            team_id=state.home.team_id,
            intensity=0.6 + 0.4 * drama_score,
            opponent_team_id=state.away.team_id,
        ))
    for team, conflict in (
        (state.home, conflict_home), (state.away, conflict_away),
    ):
        if rng.random() < 0.15 + conflict:
            events.append(MicroEvent(
                t_sec=float(rng.uniform(0.3, 0.7) * match_seconds),
                event_type=MicroEventType.ICON_PROTEST,
                team_id=team.team_id,
                intensity=0.4 + conflict,
            ))


def _append_goal_events(events, state, goal_times, rng):
    for time, side in goal_times:
        scoring_team = state.home if side == "home" else state.away
        conceding_team = state.away if side == "home" else state.home
        scorer = pick_random_outfield_player(scoring_team, rng)
        events.extend([
            MicroEvent(
                t_sec=time, event_type=MicroEventType.GOAL_SCORED,
                team_id=scoring_team.team_id, player_id=scorer.player_id,
                opponent_team_id=conceding_team.team_id,
                intensity=1.0 + 0.1 * rng.random(),
            ),
            MicroEvent(
                t_sec=time + 1.0, event_type=MicroEventType.GOAL_CONCEDED,
                team_id=conceding_team.team_id,
                opponent_team_id=scoring_team.team_id, intensity=0.9,
            ),
        ])
        # Preserve the existing home-only assist scheduling behaviour.
        if side == "home" and rng.random() < 0.55:
            assister = pick_random_outfield_player(scoring_team, rng)
            if assister.player_id != scorer.player_id:
                events.append(MicroEvent(
                    t_sec=max(0, time - 3.0), event_type=MicroEventType.ASSIST,
                    team_id=scoring_team.team_id, player_id=assister.player_id,
                    intensity=0.85,
                ))


def _append_shot_events(events, state, times, *, xg, rng):
    for time in times:
        player = pick_random_outfield_player(state, rng)
        on_target = rng.random() < 0.38 + xg * 0.05
        events.append(MicroEvent(
            t_sec=time,
            event_type=(
                MicroEventType.SHOT_ON_TARGET
                if on_target else MicroEventType.SHOT_OFF_TARGET
            ),
            team_id=state.team_id, player_id=player.player_id,
            intensity=0.7 + 0.3 * rng.random(),
        ))


def _append_discipline_events(events, state, schedule, rng):
    weight_home = schedule["foul_weight_home"]
    weight_away = schedule["foul_weight_away"]
    for time in schedule["fouls"]:
        team = pick_foul_committing_team(state, weight_home, weight_away, rng)
        opponent = state.away if team is state.home else state.home
        modulators = (
            schedule["mod_home"] if team is state.home else schedule["mod_away"]
        )
        committer = pick_weighted_outfield(team, rng, modulators)
        victim = pick_random_outfield_player(opponent, rng)
        events.extend([
            MicroEvent(
                t_sec=time, event_type=MicroEventType.FOUL_COMMITTED,
                team_id=team.team_id, player_id=committer.player_id,
                intensity=0.5 + 0.3 * rng.random(),
            ),
            MicroEvent(
                t_sec=time, event_type=MicroEventType.FOUL_SUFFERED,
                team_id=opponent.team_id, player_id=victim.player_id,
                intensity=0.55,
            ),
        ])
        if rng.random() < 0.12:
            events.append(MicroEvent(
                t_sec=time, event_type=MicroEventType.TACKLE_WON,
                team_id=opponent.team_id, player_id=victim.player_id,
                intensity=0.5,
            ))
    for time in schedule["yellows"]:
        team = pick_foul_committing_team(state, weight_home, weight_away, rng)
        modulators = (
            schedule["mod_home"] if team is state.home else schedule["mod_away"]
        )
        player = pick_weighted_outfield(team, rng, modulators)
        events.append(MicroEvent(
            t_sec=time, event_type=MicroEventType.YELLOW_CARD,
            team_id=team.team_id, player_id=player.player_id, intensity=0.8,
        ))
    for time in schedule["reds"]:
        team = state.home if rng.random() < 0.5 else state.away
        player = pick_random_outfield_player(team, rng)
        events.append(MicroEvent(
            t_sec=time, event_type=MicroEventType.RED_CARD,
            team_id=team.team_id, player_id=player.player_id, intensity=1.0,
        ))


def _append_background_events(events, state, match_seconds, rng):
    event_count = int(40 + rng.poisson(15))
    for _ in range(event_count):
        time = float(rng.uniform(0, match_seconds))
        team = state.home if rng.random() < 0.5 else state.away
        player = pick_random_outfield_player(team, rng)
        if rng.random() < 0.7:
            events.append(MicroEvent(
                t_sec=time, event_type=MicroEventType.KEY_PASS,
                team_id=team.team_id, player_id=player.player_id,
                intensity=0.25 + 0.2 * rng.random(),
            ))
        else:
            events.append(MicroEvent(
                t_sec=time, event_type=MicroEventType.DISPOSSESSED,
                team_id=team.team_id, player_id=player.player_id,
                intensity=0.35,
            ))


def build_event_schedule(
    state: MatchAffectiveState,
    *,
    goals_home: int,
    goals_away: int,
    xg_home: float,
    xg_away: float,
    referee_strictness: float = 0.55,
    drama_score: float = 0.5,
    internal_conflict_home: float = 0.12,
    internal_conflict_away: float = 0.12,
    rng: Optional[np.random.Generator] = None,
    match_seconds: Optional[float] = None,
    use_tick_fouls: bool = True,
) -> List[MicroEvent]:
    rng = rng or np.random.default_rng()
    match_sec = float(match_seconds if match_seconds is not None else 90.0 * 60.0)
    events: List[MicroEvent] = []
    xg_home = _finite_xg(xg_home)
    xg_away = _finite_xg(xg_away)
    schedule = _build_schedule_times(
        state, goals_home=goals_home, goals_away=goals_away,
        xg_home=xg_home, xg_away=xg_away,
        referee_strictness=referee_strictness, drama_score=drama_score,
        match_seconds=match_sec, use_tick_fouls=use_tick_fouls, rng=rng,
    )
    _append_atmosphere_events(
        events, state, drama_score=drama_score,
        conflict_home=internal_conflict_home,
        conflict_away=internal_conflict_away,
        match_seconds=match_sec, rng=rng,
    )
    _append_goal_events(events, state, schedule["all_goals"], rng)
    _append_shot_events(events, state.home, schedule["shots_home"], xg=xg_home, rng=rng)
    _append_shot_events(events, state.away, schedule["shots_away"], xg=xg_away, rng=rng)
    _append_discipline_events(events, state, schedule, rng)
    _append_background_events(events, state, match_sec, rng)

    events.sort(key=lambda e: e.t_sec)
    return events


def _spread_times(n: int, match_sec: float, rng: np.random.Generator) -> List[float]:
    if n <= 0:
        return []
    # avoid clustering all goals at kickoff: beta-like spread
    u = np.sort(rng.uniform(0.04, 0.96, size=n))
    return [float(x * match_sec) for x in u]
