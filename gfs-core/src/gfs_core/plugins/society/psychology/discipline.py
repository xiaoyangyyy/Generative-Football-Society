"""Foul / card helpers — press style + live foul_impulse (continuous, tick + calendar)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from math import expm1

from gfs_core.contracts.affective import AffectiveConfig
from gfs_core.contracts.events import MicroEvent, MicroEventType
from gfs_core.contracts.micro_state import (
    MatchAffectiveState,
    PlayerAffectiveState,
    PlayerModulators,
    TeamAffectiveState,
)

from .selection import pick_random_outfield_player

if TYPE_CHECKING:
    from gfs_core.contracts.micro_config import MicroMatchConfig


def baseline_foul_modulators(
    state: MatchAffectiveState,
    cfg: Optional[AffectiveConfig] = None,
) -> Tuple[Dict[str, PlayerModulators], Dict[str, PlayerModulators]]:
    """Pre-match foul_impulse from squad emotion + abilities (before tick impulses)."""
    from .coupling import AffectiveSpatialCoupling

    aff = AffectiveSpatialCoupling(cfg or AffectiveConfig())
    home = {p.player_id: aff.compute_modulators(p, state) for p in state.home.players}
    away = {p.player_id: aff.compute_modulators(p, state) for p in state.away.players}
    return home, away


def team_press_factor(tac: Dict[str, float]) -> float:
    """Continuous pressing intensity → foul-rate multiplier anchor (~0.7–1.35)."""
    p = float(tac.get("pressing_intensity", 0.5))
    cp = float(tac.get("counterpress", 0.5))
    hp = float(tac.get("high_press", 0.5))
    mop = float(tac.get("man_oriented_press", 0.5))
    return float(
        np.clip(0.52 + 0.28 * p + 0.14 * cp + 0.10 * hp + 0.08 * mop, 0.55, 1.40)
    )


def team_possession_foul_relief(tac: Dict[str, float]) -> float:
    """Possession-oriented sides commit fewer fouls (continuous relief)."""
    po = float(tac.get("possession_orientation", 0.5))
    bus = float(tac.get("build_up_short", 0.5))
    return float(np.clip(1.0 - 0.20 * po - 0.08 * bus, 0.62, 1.0))


def team_impulse_factor(
    team: TeamAffectiveState,
    mod_map: Dict[str, PlayerModulators],
) -> float:
    vals: List[float] = []
    for p in team.players:
        if not p.on_pitch or p.role == "GK":
            continue
        m = mod_map.get(p.player_id)
        if m is None:
            continue
        vals.append(max(0.0, float(m.foul_impulse)))
    if not vals:
        return 1.0
    mean_i = float(np.mean(vals))
    return float(np.clip(0.78 + 0.42 * mean_i, 0.72, 1.35))


def team_foul_weight(
    team: TeamAffectiveState,
    mod_map: Dict[str, PlayerModulators],
) -> float:
    tac = dict(team.coach.tactical_current or team.coach.tactical_base or {})
    return (
        team_press_factor(tac)
        * team_possession_foul_relief(tac)
        * team_impulse_factor(team, mod_map)
    )


def expected_match_fouls(
    w_home: float,
    w_away: float,
    referee_strictness: float,
    rng: np.random.Generator,
) -> int:
    """
    Total foul events (one committing team per event).
    Count is referee-calibrated; w_home/w_away steer allocation (press vs possession).
    """
    del w_home, w_away
    ref = float(np.clip(referee_strictness, 0.0, 1.0))
    base = 14.0 + 11.0 * ref
    n = int(round(base + rng.poisson(3)))
    return int(np.clip(n, 10, 36))


def pick_weighted_outfield(
    team: TeamAffectiveState,
    rng: np.random.Generator,
    mod_map: Dict[str, PlayerModulators],
) -> PlayerAffectiveState:

    pool = [p for p in team.players if p.on_pitch and p.role != "GK"]
    if not pool:
        return pick_random_outfield_player(team, rng)
    w = np.array(
        [
            max(
                0.05,
                float(
                    mod_map.get(p.player_id, PlayerModulators(p.player_id)).foul_impulse
                )
                + 0.08,
            )
            for p in pool
        ],
        dtype=float,
    )
    w /= w.sum()
    return pool[int(rng.choice(len(pool), p=w))]


def pick_foul_committing_team(
    state: MatchAffectiveState,
    w_home: float,
    w_away: float,
    rng: np.random.Generator,
) -> TeamAffectiveState:
    wh = max(0.05, float(w_home))
    wa = max(0.05, float(w_away))
    return state.home if rng.random() < wh / (wh + wa) else state.away


def mod_list_to_map(mod_list: List[PlayerModulators]) -> Dict[str, PlayerModulators]:
    return {m.player_id: m for m in mod_list}


def foul_weights_from_mods(
    state: MatchAffectiveState,
    mod_home: List[PlayerModulators],
    mod_away: List[PlayerModulators],
) -> Tuple[float, float]:
    return (
        team_foul_weight(state.home, mod_list_to_map(mod_home)),
        team_foul_weight(state.away, mod_list_to_map(mod_away)),
    )


def tick_foul_probability(
    *,
    lambda_foul: float,
    lambda_foul_base: float,
    match_seconds: float,
    dt: float,
    referee_strictness: float,
    foul_target_base: float,
    foul_target_ref_slope: float,
) -> float:
    """Poisson event probability for an interval of dt seconds."""
    if dt <= 0 or match_seconds <= 0:
        raise ValueError("foul interval and match duration must be positive")
    ref = float(np.clip(referee_strictness, 0.0, 1.0))
    target_total = foul_target_base + foul_target_ref_slope * ref
    rate = target_total / match_seconds
    lam_scale = float(lambda_foul) / max(1e-6, float(lambda_foul_base))
    return float(-expm1(-rate * lam_scale * dt))


def emit_tick_discipline(
    state: MatchAffectiveState,
    mod_home: List[PlayerModulators],
    mod_away: List[PlayerModulators],
    *,
    cfg: "MicroMatchConfig",
    lambda_foul: float,
    referee_strictness: float,
    drama_score: float,
    rng: np.random.Generator,
    player_tracker: Optional[object] = None,
) -> List[MicroEvent]:
    """
    Continuous in-play fouls/cards from lambda_foul, pressing style, and live foul_impulse.
    """
    if not getattr(cfg, "discipline_tick_fouls", True):
        return []

    match_sec = float(getattr(cfg, "match_seconds", 90.0 * 60.0))
    dt = float(cfg.dt_default)
    p = tick_foul_probability(
        lambda_foul=lambda_foul,
        lambda_foul_base=cfg.lambda_foul_base,
        match_seconds=match_sec,
        dt=dt,
        referee_strictness=referee_strictness,
        foul_target_base=float(getattr(cfg, "discipline_foul_target_base", 14.0)),
        foul_target_ref_slope=float(
            getattr(cfg, "discipline_foul_target_ref_slope", 11.0)
        ),
    )
    if rng.random() >= p:
        return []

    w_h, w_a = foul_weights_from_mods(state, mod_home, mod_away)
    side = pick_foul_committing_team(state, w_h, w_a, rng)
    opp = state.away if side is state.home else state.home
    side_mod = mod_list_to_map(mod_home if side is state.home else mod_away)


    committer = pick_weighted_outfield(side, rng, side_mod)
    victim = pick_random_outfield_player(opp, rng)
    events: List[MicroEvent] = [
        MicroEvent(
            t_sec=float(state.clock_seconds),
            event_type=MicroEventType.FOUL_COMMITTED,
            team_id=side.team_id,
            player_id=committer.player_id,
            intensity=0.5
            + 0.25
            * float(
                side_mod.get(
                    committer.player_id, PlayerModulators(committer.player_id)
                ).foul_impulse
            ),
        ),
        MicroEvent(
            t_sec=float(state.clock_seconds),
            event_type=MicroEventType.FOUL_SUFFERED,
            team_id=opp.team_id,
            player_id=victim.player_id,
            intensity=0.55,
        ),
    ]
    if rng.random() < 0.12:
        events.append(
            MicroEvent(
                t_sec=float(state.clock_seconds),
                event_type=MicroEventType.TACKLE_WON,
                team_id=opp.team_id,
                player_id=victim.player_id,
                intensity=0.5,
            )
        )

    fi = float(
        side_mod.get(
            committer.player_id, PlayerModulators(committer.player_id)
        ).foul_impulse
    )
    p_yellow = (
        float(getattr(cfg, "discipline_yellow_on_foul_base", 0.055))
        + float(getattr(cfg, "discipline_yellow_on_foul_ref", 0.075))
        * referee_strictness
        + 0.04 * fi
    )
    if rng.random() < p_yellow:
        events.append(
            MicroEvent(
                t_sec=float(state.clock_seconds),
                event_type=MicroEventType.YELLOW_CARD,
                team_id=side.team_id,
                player_id=committer.player_id,
                intensity=0.8,
            )
        )

    p_red = (
        float(getattr(cfg, "discipline_red_on_foul_base", 0.004))
        + float(getattr(cfg, "discipline_red_drama_gain", 0.012)) * drama_score
    )
    if rng.random() < p_red:
        events.append(
            MicroEvent(
                t_sec=float(state.clock_seconds),
                event_type=MicroEventType.RED_CARD,
                team_id=side.team_id,
                player_id=committer.player_id,
                intensity=1.0,
            )
        )

    if player_tracker is not None:
        for ev in events:
            player_tracker.apply_event(ev)
    return events


def emit_tick_tackle(
    state: MatchAffectiveState,
    mod_home: List[PlayerModulators],
    mod_away: List[PlayerModulators],
    *,
    cfg: "MicroMatchConfig",
    lambda_foul: float,
    rng: np.random.Generator,
) -> List[MicroEvent]:
    """Standalone pressing tackles (continuous, no hard thresholds)."""
    dt = float(cfg.dt_default)
    match_sec = float(getattr(cfg, "match_seconds", 90.0 * 60.0))
    tac_h = state.home.coach.tactical_current or {}
    tac_a = state.away.coach.tactical_current or {}
    press = 0.5 * (team_press_factor(tac_h) + team_press_factor(tac_a))
    lam_scale = float(lambda_foul) / max(1e-6, float(cfg.lambda_foul_base))
    target_tackles = 11.0
    p = float(-expm1(-(target_tackles / match_sec) * press * lam_scale * 0.55 * dt))
    if rng.random() >= p:
        return []
    side = state.home if rng.random() < 0.5 else state.away

    tackler = pick_random_outfield_player(side, rng)
    return [
        MicroEvent(
            t_sec=float(state.clock_seconds),
            event_type=MicroEventType.TACKLE_WON,
            team_id=side.team_id,
            player_id=tackler.player_id,
            intensity=0.55,
        )
    ]


def emit_tick_cross(
    state: MatchAffectiveState,
    aerial_engine,
    *,
    cfg: "MicroMatchConfig",
    rng: np.random.Generator,
) -> list[MicroEvent]:
    """Wing crosses at rate calibrated to StatsBomb (~11/team/match)."""
    if not state.ball.possession_team_id:
        return []
    dt = float(cfg.dt_default)
    match_sec = float(getattr(cfg, "match_seconds", 90.0 * 60.0))
    ticks = max(1.0, match_sec / dt)
    team = state.team(state.ball.possession_team_id)
    tac = dict(team.coach.tactical_current or {})
    wing = float(tac.get("cross_frequency", 0.5)) * float(tac.get("wing_focus", 0.5))
    target = float(getattr(cfg, "cross_tick_target", 20.0))
    p = float(np.clip((target / ticks) * (0.68 + 0.52 * wing), 0.0, 0.16))
    if rng.random() >= p:
        return []
    wide = [
        p for p in team.players if p.on_pitch and p.role in ("LW", "RW", "LB", "RB")
    ]
    if not wide:
        return []
    crosser = wide[int(rng.integers(0, len(wide)))]
    _, outcome = aerial_engine.resolve_cross(state, crosser, rng)
    attacking_home = crosser.team_id == state.home.team_id
    _apply_aerial_xg(state, outcome.xg_added, attacking_home, cfg)
    return _aerial_goal_events(state, crosser, outcome)


def _apply_aerial_xg(state, value, attacking_home, cfg):
    value = max(0.0, float(value))
    if value <= 0.0:
        return
    cap = float(getattr(cfg, "micro_xg_match_cap", 2.2))
    if attacking_home:
        room = max(0.0, cap - state.micro_xg_home)
        state.micro_xg_home += min(value, room)
    else:
        room = max(0.0, cap - state.micro_xg_away)
        state.micro_xg_away += min(value, room)


def _aerial_goal_events(state, crosser, outcome):
    if not outcome.goal or not outcome.winner_id:
        return []
    attacking_home = crosser.team_id == state.home.team_id
    opponent_id = state.away.team_id if attacking_home else state.home.team_id
    return [
        MicroEvent(
            t_sec=float(state.clock_seconds),
            event_type=MicroEventType.GOAL_SCORED,
            team_id=crosser.team_id,
            player_id=outcome.winner_id,
            opponent_team_id=opponent_id,
            intensity=1.0,
            meta={"source": "aerial_header"},
        ),
        MicroEvent(
            t_sec=float(state.clock_seconds) + 0.5,
            event_type=MicroEventType.GOAL_CONCEDED,
            team_id=opponent_id,
            opponent_team_id=crosser.team_id,
            intensity=0.9,
            meta={"source": "aerial_header"},
        ),
    ]
