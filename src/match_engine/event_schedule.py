"""
Stochastic micro-event schedule conditioned on match score / xG / drama.

Produces a realistic minute-by-minute event stream to drive Phase 1b ODEs
without requiring full spatial simulation yet.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

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
from src.match_engine.state import MatchAffectiveState, TeamAffectiveState


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

    def _finite_xg(x: float, default: float = 1.0) -> float:
        v = float(x)
        return default if not np.isfinite(v) else float(np.clip(v, 0.05, 3.6))

    xg_home = _finite_xg(xg_home)
    xg_away = _finite_xg(xg_away)

    gh, ga = int(max(0, goals_home)), int(max(0, goals_away))
    goal_times_h = _spread_times(gh, match_sec, rng)
    goal_times_a = _spread_times(ga, match_sec, rng)
    all_goal_times = sorted([(t, "home") for t in goal_times_h] + [(t, "away") for t in goal_times_a])

    # Scheduled shots optional; default OFF on physics-official path (Phase 4).
    from src.simulation.score_path import scheduled_shots_allowed

    use_calendar = scheduled_shots_allowed()

    if not use_calendar:
        n_shots_h, n_shots_a = gh, ga
    else:
        from src.match_engine.micro_config import MicroMatchConfig

        mc = MicroMatchConfig()
        xg_mult = float(getattr(mc, "schedule_shots_xg_mult", 1.35))
        pois = float(getattr(mc, "schedule_shots_poisson", 0.6))
        cap = int(getattr(mc, "max_scheduled_shots_per_team", 16))
        n_shots_h = min(cap, max(gh, int(round(xg_home * xg_mult + rng.poisson(pois)))))
        n_shots_a = min(cap, max(ga, int(round(xg_away * xg_mult + rng.poisson(pois)))))
    shot_times_h = _spread_times(n_shots_h, match_sec, rng)
    shot_times_a = _spread_times(n_shots_a, match_sec, rng)

    mod_home, mod_away = baseline_foul_modulators(state)
    w_foul_home = team_foul_weight(state.home, mod_home)
    w_foul_away = team_foul_weight(state.away, mod_away)

    foul_times: List[float] = []
    yellow_times: List[float] = []
    red_times: List[float] = []
    if not use_tick_fouls:
        n_fouls = expected_match_fouls(w_foul_home, w_foul_away, referee_strictness, rng)
        foul_times = _spread_times(n_fouls, match_sec, rng)
        card_pressure = 0.5 * (w_foul_home + w_foul_away)
        n_yellow = int(min(6, max(0, rng.poisson(2.0 + referee_strictness + 0.35 * (card_pressure - 1.0)))))
        yellow_times = _spread_times(n_yellow, match_sec, rng)
        red_prob = 0.008 + 0.02 * drama_score
        n_red = 1 if rng.random() < red_prob else 0
        red_times = _spread_times(n_red, match_sec, rng)

    var_prob = 0.12 + 0.2 * drama_score
    if rng.random() < var_prob:
        events.append(
            MicroEvent(
                t_sec=float(rng.uniform(0.25, 0.85) * match_sec),
                event_type=MicroEventType.VAR_CONTROVERSY,
                team_id=state.home.team_id,
                intensity=0.6 + 0.4 * drama_score,
                opponent_team_id=state.away.team_id,
            )
        )

    if rng.random() < 0.15 + internal_conflict_home:
        events.append(
            MicroEvent(
                t_sec=float(rng.uniform(0.3, 0.7) * match_sec),
                event_type=MicroEventType.ICON_PROTEST,
                team_id=state.home.team_id,
                intensity=0.4 + internal_conflict_home,
            )
        )
    if rng.random() < 0.15 + internal_conflict_away:
        events.append(
            MicroEvent(
                t_sec=float(rng.uniform(0.3, 0.7) * match_sec),
                event_type=MicroEventType.ICON_PROTEST,
                team_id=state.away.team_id,
                intensity=0.4 + internal_conflict_away,
            )
        )

    for t, side in all_goal_times:
        if side == "home":
            scorer = pick_random_outfield_player(state.home, rng)
            events.append(
                MicroEvent(
                    t_sec=t,
                    event_type=MicroEventType.GOAL_SCORED,
                    team_id=state.home.team_id,
                    player_id=scorer.player_id,
                    opponent_team_id=state.away.team_id,
                    intensity=1.0 + 0.1 * rng.random(),
                )
            )
            events.append(
                MicroEvent(
                    t_sec=t + 1.0,
                    event_type=MicroEventType.GOAL_CONCEDED,
                    team_id=state.away.team_id,
                    opponent_team_id=state.home.team_id,
                    intensity=0.9,
                )
            )
            if rng.random() < 0.55:
                assister = pick_random_outfield_player(state.home, rng)
                if assister.player_id != scorer.player_id:
                    events.append(
                        MicroEvent(
                            t_sec=max(0, t - 3.0),
                            event_type=MicroEventType.ASSIST,
                            team_id=state.home.team_id,
                            player_id=assister.player_id,
                            intensity=0.85,
                        )
                    )
        else:
            scorer = pick_random_outfield_player(state.away, rng)
            events.append(
                MicroEvent(
                    t_sec=t,
                    event_type=MicroEventType.GOAL_SCORED,
                    team_id=state.away.team_id,
                    player_id=scorer.player_id,
                    opponent_team_id=state.home.team_id,
                    intensity=1.0 + 0.1 * rng.random(),
                )
            )
            events.append(
                MicroEvent(
                    t_sec=t + 1.0,
                    event_type=MicroEventType.GOAL_CONCEDED,
                    team_id=state.home.team_id,
                    opponent_team_id=state.away.team_id,
                    intensity=0.9,
                )
            )

    for t in shot_times_h:
        player = pick_random_outfield_player(state.home, rng)
        on = rng.random() < 0.38 + xg_home * 0.05
        events.append(
            MicroEvent(
                t_sec=t,
                event_type=MicroEventType.SHOT_ON_TARGET if on else MicroEventType.SHOT_OFF_TARGET,
                team_id=state.home.team_id,
                player_id=player.player_id,
                intensity=0.7 + 0.3 * rng.random(),
            )
        )
    for t in shot_times_a:
        player = pick_random_outfield_player(state.away, rng)
        on = rng.random() < 0.38 + xg_away * 0.05
        events.append(
            MicroEvent(
                t_sec=t,
                event_type=MicroEventType.SHOT_ON_TARGET if on else MicroEventType.SHOT_OFF_TARGET,
                team_id=state.away.team_id,
                player_id=player.player_id,
                intensity=0.7 + 0.3 * rng.random(),
            )
        )

    for t in foul_times:
        side = pick_foul_committing_team(state, w_foul_home, w_foul_away, rng)
        opp = state.away if side is state.home else state.home
        side_mod = mod_home if side is state.home else mod_away
        committer = pick_weighted_outfield(side, rng, side_mod)
        victim = pick_random_outfield_player(opp, rng)
        events.append(
            MicroEvent(
                t_sec=t,
                event_type=MicroEventType.FOUL_COMMITTED,
                team_id=side.team_id,
                player_id=committer.player_id,
                intensity=0.5 + 0.3 * rng.random(),
            )
        )
        events.append(
            MicroEvent(
                t_sec=t,
                event_type=MicroEventType.FOUL_SUFFERED,
                team_id=opp.team_id,
                player_id=victim.player_id,
                intensity=0.55,
            )
        )
        if rng.random() < 0.12:
            events.append(
                MicroEvent(
                    t_sec=t,
                    event_type=MicroEventType.TACKLE_WON,
                    team_id=opp.team_id,
                    player_id=victim.player_id,
                    intensity=0.5,
                )
            )

    for t in yellow_times:
        side = pick_foul_committing_team(state, w_foul_home, w_foul_away, rng)
        side_mod = mod_home if side is state.home else mod_away
        p = pick_weighted_outfield(side, rng, side_mod)
        events.append(
            MicroEvent(
                t_sec=t,
                event_type=MicroEventType.YELLOW_CARD,
                team_id=side.team_id,
                player_id=p.player_id,
                intensity=0.8,
            )
        )

    for t in red_times:
        side = state.home if rng.random() < 0.5 else state.away
        p = pick_random_outfield_player(side, rng)
        events.append(
            MicroEvent(
                t_sec=t,
                event_type=MicroEventType.RED_CARD,
                team_id=side.team_id,
                player_id=p.player_id,
                intensity=1.0,
            )
        )

    # background passes / dispossessions
    n_bg = int(40 + rng.poisson(15))
    for _ in range(n_bg):
        t = float(rng.uniform(0, match_sec))
        side = state.home if rng.random() < 0.5 else state.away
        p = pick_random_outfield_player(side, rng)
        if rng.random() < 0.7:
            events.append(
                MicroEvent(
                    t_sec=t,
                    event_type=MicroEventType.KEY_PASS,
                    team_id=side.team_id,
                    player_id=p.player_id,
                    intensity=0.25 + 0.2 * rng.random(),
                )
            )
        else:
            events.append(
                MicroEvent(
                    t_sec=t,
                    event_type=MicroEventType.DISPOSSESSED,
                    team_id=side.team_id,
                    player_id=p.player_id,
                    intensity=0.35,
                )
            )

    events.sort(key=lambda e: e.t_sec)
    return events


def _spread_times(n: int, match_sec: float, rng: np.random.Generator) -> List[float]:
    if n <= 0:
        return []
    # avoid clustering all goals at kickoff: beta-like spread
    u = np.sort(rng.uniform(0.04, 0.96, size=n))
    return [float(x * match_sec) for x in u]
