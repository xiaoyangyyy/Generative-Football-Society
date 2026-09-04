"""
Run full-match Phase 1b affective simulation driven by score/xG-calibrated events.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.affective_coupling import AffectiveSpatialCoupling
from src.match_engine.config import AffectiveConfig
from src.match_engine.event_schedule import build_event_schedule
from src.match_engine.internal_signals import normalize_internal_match_signals
from src.match_engine.macro_bridge import apply_affective_endstate_to_agents, build_match_affective_state
from src.match_engine.meso_aggregator import MesoAggregator, icon_emotion_shock
from src.match_engine.state import AffectiveMatchSummary

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def run_match_affective_simulation(
    home_agent: "SocietyAgent",
    away_agent: "SocietyAgent",
    *,
    goals_home: int,
    goals_away: int,
    xg_home: float,
    xg_away: float,
    referee: Optional[Dict[str, Any]] = None,
    stage_pressure: float = 0.3,
    drama_score: float = 0.5,
    internal_home: Optional[Dict[str, float]] = None,
    internal_away: Optional[Dict[str, float]] = None,
    neutral_venue: bool = False,
    config: Optional[AffectiveConfig] = None,
    seed: int = 42,
    writeback_agents: bool = True,
    blend: float = 0.35,
    base_dir: str | os.PathLike[str] | None = None,
) -> AffectiveMatchSummary:
    cfg = config or AffectiveConfig()
    rng = np.random.default_rng(seed)
    engine = AffectiveSpatialCoupling(cfg)

    state = build_match_affective_state(
        home_agent,
        away_agent,
        referee=referee,
        stage_pressure=stage_pressure,
        neutral_venue=neutral_venue,
        rng=rng,
        base_dir=base_dir,
    )
    internal_signals = normalize_internal_match_signals(
        internal_home, internal_away,
    )
    coord_h = internal_signals.coordination_home
    coord_a = internal_signals.coordination_away
    conflict_h = internal_signals.conflict_home
    conflict_a = internal_signals.conflict_away

    ref_strict = float((referee or {}).get("strictness", 0.55))
    schedule = build_event_schedule(
        state,
        goals_home=goals_home,
        goals_away=goals_away,
        xg_home=xg_home,
        xg_away=xg_away,
        referee_strictness=ref_strict,
        drama_score=drama_score,
        internal_conflict_home=conflict_h,
        internal_conflict_away=conflict_a,
        rng=rng,
        use_tick_fouls=bool(getattr(cfg, "discipline_tick_fouls", True)),
    )

    dt = cfg.dt_default
    n_ticks = int(cfg.match_seconds / dt)
    ev_idx = 0
    meso = MesoAggregator(cfg.meso_window_seconds)
    timeline: List[str] = []
    strictness_sum = 0.0

    for tick in range(n_ticks):
        t0 = tick * dt
        t1 = t0 + dt
        tick_events = []
        while ev_idx < len(schedule) and schedule[ev_idx].t_sec < t1:
            if schedule[ev_idx].t_sec >= t0:
                tick_events.append(schedule[ev_idx])
            ev_idx += 1

        score_diff_home = state.home.score - state.away.score
        xg_swing_home = xg_home - xg_away
        press_h = float(np.clip(home_agent.tactical_controls.get("pressing_intensity", 0.5), 0, 1))
        press_a = float(np.clip(away_agent.tactical_controls.get("pressing_intensity", 0.5), 0, 1))

        _, scalars = engine.step(
            state,
            dt,
            events=tick_events,
            score_diff_home=score_diff_home,
            xg_swing_home=xg_swing_home,
            coordination_home=coord_h,
            coordination_away=coord_a,
            conflict_home=conflict_h,
            conflict_away=conflict_a,
            press_home=press_h,
            press_away=press_a,
        )
        strictness_sum += scalars["strictness_effective"]

        emo_h = engine.team_emotion_mean(state.home)
        emo_a = engine.team_emotion_mean(state.away)
        for team, tid, xgf, xga, emo in (
            (state.home, state.home.team_id, xg_home, xg_away, emo_h),
            (state.away, state.away.team_id, xg_away, xg_home, emo_a),
        ):
            meso.record_tick(
                state,
                team_id=tid,
                xg_for=xgf,
                xg_against=xga,
                controversy=state.referee.controversy_integral,
                psi=state.crowd.psi,
                emotion=emo,
                tactical=team.coach.tactical_current,
                tactical_base=team.coach.tactical_base,
                icon_shock=icon_emotion_shock(team),
            )

        for ev in tick_events:
            if ev.event_type.value in ("goal_scored", "red_card", "var_controversy"):
                timeline.append(f"{int(t0//60)}' {ev.event_type.value} ({ev.team_id})")

    packets = meso.flush_packets(state.home.team_id, state.away.team_id)
    emo_home = engine.team_emotion_mean(state.home)
    emo_away = engine.team_emotion_mean(state.away)

    def tactical_drift(team) -> float:
        base = team.coach.tactical_base
        cur = team.coach.tactical_current
        return float(sum(abs(cur.get(k, 0.5) - base.get(k, 0.5)) for k in base))

    state.home.score = int(goals_home)
    state.away.score = int(goals_away)

    summary = AffectiveMatchSummary(
        home_team=state.home.team_id,
        away_team=state.away.team_id,
        ticks=n_ticks,
        final_psi=state.crowd.psi,
        home_coach_stress=state.home.coach.stress(),
        away_coach_stress=state.away.coach.stress(),
        ref_strictness_mean=strictness_sum / max(1, n_ticks),
        home_emotion_mean=emo_home,
        away_emotion_mean=emo_away,
        icon_shock_home=icon_emotion_shock(state.home),
        icon_shock_away=icon_emotion_shock(state.away),
        tactical_drift_home=tactical_drift(state.home),
        tactical_drift_away=tactical_drift(state.away),
        controversy_integral=state.referee.controversy_integral,
        meso_packets=[p.__dict__ if hasattr(p, "__dict__") else p for p in packets],
        timeline_snippet=timeline[:12],
    )

    if writeback_agents:
        apply_affective_endstate_to_agents(
            home_agent, away_agent, state, emo_home, emo_away, blend=blend
        )

    return summary
