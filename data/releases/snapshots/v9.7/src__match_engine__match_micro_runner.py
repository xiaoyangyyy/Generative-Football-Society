"""
Full micro match: Phase 1b + 2a + 2b + 3 (affective, spatial, passing, shots/aerial, micro xG→λ).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.action_engine import ActionEngine
from src.match_engine.affective_coupling import AffectiveSpatialCoupling
from src.match_engine.event_schedule import build_event_schedule
from src.match_engine.goal_generator import lambdas_from_micro, simulate_match_score_from_micro
from src.match_engine.kinematic_position import KinematicPositionLayer
from src.match_engine.macro_bridge import apply_affective_endstate_to_agents, build_match_affective_state
from src.match_engine.meso_aggregator import MesoAggregator, icon_emotion_shock
from src.match_engine.possession_helpers import (
    mark_shot_taken,
    shot_cooldown_ok,
    turnover_to_defence_after_shot,
)
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEvent, MicroEventType, apply_micro_event
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.aerial_duel import AerialDuelEngine
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.spatial_field import SpatialFieldEngine
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.squad_factory import init_away_positions
from src.match_engine.tactical_engine import TacticalMicroEngine
from src.match_engine.player_match_stats import PlayerMatchStatsTracker
from src.match_engine.substitution_engine import maybe_apply_substitutions
from src.match_engine.state import MicroMatchSummary

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def _resolve_cognitive_layer(
    cfg: MicroMatchConfig,
    seed: int,
    home_agent: "SocietyAgent",
    away_agent: "SocietyAgent",
):
    from src.match_engine.cognitive.config import CognitiveMatchConfig

    cog_cfg = CognitiveMatchConfig.from_env()
    if not cog_cfg.enabled:
        return None, None
    from src.match_engine.cognitive.bus import MatchCognitiveBus
    from src.match_engine.cognitive.executor import CognitiveExecutor

    rng_cog = np.random.default_rng(seed + 31337)
    bus = MatchCognitiveBus(cog_cfg, rng_cog)
    llm = None
    try:
        from src.simulation.llm_engine import SimulationLLM

        llm = SimulationLLM()
    except Exception:
        llm = None
    executor = CognitiveExecutor(cog_cfg, llm, use_llm=bool(llm))
    return bus, executor


def _process_cognitive_tick(
    bus,
    executor,
    state,
    tick_events,
    t1: float,
    dt: float,
    xg_swing_home: float,
    home_agent,
    away_agent,
    processed_subs: set,
    player_tracker,
) -> None:
    if bus is None or executor is None:
        return
    from src.match_engine.cognitive.assistant_dynamics import step_assistant_dynamics

    for ev in tick_events:
        bus.ingest_micro_event(ev, state)
    bus.ingest_clock(t1, state, xg_swing_home=xg_swing_home)
    step_assistant_dynamics(state, dt)
    for sub in player_tracker.substitutions:
        key = (sub.get("off"), sub.get("on"), sub.get("minute"))
        if key in processed_subs:
            continue
        processed_subs.add(key)
        bus.ingest_substitution(
            t1,
            state,
            str(sub.get("team_id", "")),
            str(sub.get("off", "")),
            str(sub.get("on", "")),
        )
    pending = bus.drain_pending()
    if pending:
        executor.process_batch(
            pending,
            state,
            home_agent=home_agent,
            away_agent=away_agent,
        )


def _possession_prior_from_strength(
    eff_home: float,
    eff_away: float,
    cfg: MicroMatchConfig,
    *,
    neutral_venue: bool = False,
) -> float:
    # Possession prior reflects squad strength only. Home advantage is NOT here —
    # it flows through crowd ψ → player morale/emotion (see affective_coupling).
    del neutral_venue
    total = max(24.0, eff_home + eff_away)
    share = eff_home / total
    beta = float(cfg.possession_strength_beta)
    return float(np.clip(0.5 + beta * (share - 0.5), 0.36, 0.64))


def _init_micro_state(
    state,
    cfg: MicroMatchConfig,
    rng: np.random.Generator,
    *,
    possession_home: float | None = None,
) -> None:
    state.home.attacks_high_x = True
    state.away.attacks_high_x = False
    init_away_positions(state.away)
    poss_h = float(possession_home if possession_home is not None else cfg.possession_home_prior)
    state.home.possession_share = poss_h
    state.away.possession_share = 1.0 - poss_h
    dm = next((p for p in state.home.players if p.role == "CM"), state.home.players[0])
    state.ball.position = dm.position.copy()
    state.ball.possessor_id = dm.player_id
    state.ball.possession_team_id = state.home.team_id
    state.ball.velocity = np.zeros(2)
    state.ball.height = 0.0
    state.ball.omega = 0.0


def _events_for_affective(tick_events: List, cfg: MicroMatchConfig) -> List:
    """Strip events handled by Phase 3 physics or end-of-match Poisson."""
    if not cfg.enable_phase3:
        return tick_events
    skip = {MicroEventType.SHOT_ON_TARGET, MicroEventType.SHOT_OFF_TARGET}
    if cfg.use_micro_goals:
        skip |= {MicroEventType.GOAL_SCORED, MicroEventType.GOAL_CONCEDED}
    return [e for e in tick_events if e.event_type not in skip]


def _apply_phase3_events(state, events: List, cfg: MicroMatchConfig) -> None:
    for ev in events:
        if not cfg.use_micro_goals and ev.event_type in (
            MicroEventType.GOAL_SCORED,
            MicroEventType.GOAL_CONCEDED,
        ):
            continue
        apply_micro_event(state, ev, cfg)


def _resolve_scheduled_shots(state, tick_events, shots: ShotEngine, cfg: MicroMatchConfig, mod_home, mod_away, rng):
    """Physics-resolve shot events from the calendar (replacing heuristic xG bump)."""
    for ev in tick_events:
        if ev.event_type not in (MicroEventType.SHOT_ON_TARGET, MicroEventType.SHOT_OFF_TARGET):
            continue
        team = state.team(ev.team_id)
        attacking_home = ev.team_id == state.home.team_id
        pool = [p for p in team.players if p.on_pitch and p.role != "GK"]
        if not pool:
            continue
        carrier = pool[int(rng.integers(0, len(pool)))]
        if ev.player_id:
            for p in pool:
                if p.player_id == ev.player_id:
                    carrier = p
                    break
        scheduled_on_target = ev.event_type == MicroEventType.SHOT_ON_TARGET
        if scheduled_on_target:
            gx = cfg.scheduled_shot_box_x_home if attacking_home else cfg.scheduled_shot_box_x_away
            spread = float(cfg.scheduled_shot_box_y_spread)
            carrier.position[0] = float(gx)
            carrier.position[1] = float(np.clip(0.5 + (rng.random() - 0.5) * spread, 0.10, 0.90))
        state.ball.possessor_id = carrier.player_id
        state.ball.position = carrier.position.copy()
        mod_list = mod_home if attacking_home else mod_away
        mod_map = {m.player_id: m for m in mod_list}
        mod_c = mod_map.get(carrier.player_id, mod_list[0])
        kind = "curved" if rng.random() < 0.2 else "driven"
        now = float(state.clock_seconds)
        if not shot_cooldown_ok(state, attacking_home, cfg, now):
            continue
        out = shots.resolve_shot(
            state,
            carrier,
            mod_c,
            rng,
            forced_kind=kind,
            scheduled_on_target=scheduled_on_target,
            from_schedule=True,
        )
        mark_shot_taken(state, attacking_home, now)
        if not out.goal:
            def_id = state.away.team_id if attacking_home else state.home.team_id
            gk = shots._gk_player(state, def_id)
            turnover_to_defence_after_shot(state, attacking_home, gk, rng)
        _apply_phase3_events(state, out.events, cfg)


def run_match_micro_simulation(
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
    config: Optional[MicroMatchConfig] = None,
    seed: int = 42,
    writeback_agents: bool = True,
    blend: float = 0.35,
    eff_status_home: Optional[float] = None,
    eff_status_away: Optional[float] = None,
    stage_name: str = "match",
    match_seconds: Optional[float] = None,
    tactical_override_home: Optional[Dict[str, float]] = None,
    tactical_override_away: Optional[Dict[str, float]] = None,
) -> MicroMatchSummary:
    cfg = config or MicroMatchConfig()
    rng = np.random.default_rng(seed)
    eff_h = float(eff_status_home if eff_status_home is not None else home_agent.status_score)
    eff_a = float(eff_status_away if eff_status_away is not None else away_agent.status_score)
    poss_home = _possession_prior_from_strength(eff_h, eff_a, cfg, neutral_venue=neutral_venue)

    affective = AffectiveSpatialCoupling(cfg)
    spatial = SpatialFieldEngine(cfg)
    sie = SpatialIntelligenceEngine(cfg)
    kinematic = KinematicPositionLayer(cfg, sie=sie)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    wm_runtime = None
    from src.match_engine.world_model.config import world_model_enabled
    if world_model_enabled():
        from src.match_engine.world_model.inference import WorldModelRuntime

        wm_runtime = WorldModelRuntime.load_default(base_dir)

    passing = PassingEngine(cfg, sie, wm_runtime=wm_runtime, base_dir=base_dir)
    shots = ShotEngine(cfg, sie)
    player_tracker = PlayerMatchStatsTracker()
    passing.player_tracker = player_tracker
    shots.player_tracker = player_tracker
    aerial = AerialDuelEngine(cfg)
    continuous_clock = None
    if cfg.enable_continuous_event_clock:
        from src.match_engine.continuous_micro_clock import ContinuousMicroEventClock

        artifact = Path(cfg.continuous_clock_artifact)
        if not artifact.is_absolute():
            artifact = Path(base_dir) / artifact
        continuous_clock = ContinuousMicroEventClock(
            artifact, cfg.continuous_clock_provider, cfg.continuous_clock_max_delay_s
        )
    subtick_queue = None
    if cfg.enable_subtick_reception_queue:
        if cfg.enable_continuous_event_clock:
            raise ValueError("legacy continuous gate and sub-tick reception queue are mutually exclusive")
        from src.match_engine.subtick_reception_queue import SubtickReceptionQueue

        temporal_artifact = Path(cfg.continuous_clock_artifact)
        pass_artifact = Path(cfg.subtick_pass_artifact)
        if not temporal_artifact.is_absolute():
            temporal_artifact = Path(base_dir) / temporal_artifact
        if not pass_artifact.is_absolute():
            pass_artifact = Path(base_dir) / pass_artifact
        subtick_queue = SubtickReceptionQueue(temporal_artifact, pass_artifact, cfg.subtick_reception_provider, cfg.subtick_response_blend)
    actions = ActionEngine(
        cfg, passing, shots, aerial, wm_runtime=wm_runtime, continuous_clock=continuous_clock,
        subtick_queue=subtick_queue,
    )
    tactical_eng = TacticalMicroEngine(cfg)

    state = build_match_affective_state(
        home_agent,
        away_agent,
        referee=referee,
        stage_pressure=stage_pressure,
        neutral_venue=neutral_venue,
        eff_status_home=eff_h,
        eff_status_away=eff_a,
        rng=rng,
    )
    _init_micro_state(state, cfg, rng, possession_home=poss_home)
    tactical_eng.bootstrap(state, home_agent, away_agent)
    if tactical_override_home:
        from src.match_engine.tactical_profile import apply_vector_to_team_coach

        apply_vector_to_team_coach(state.home, tactical_override_home)
    if tactical_override_away:
        from src.match_engine.tactical_profile import apply_vector_to_team_coach

        apply_vector_to_team_coach(state.away, tactical_override_away)

    ih = internal_home or {}
    ia = internal_away or {}
    coord_h = float(ih.get("coordination", 0.6))
    coord_a = float(ia.get("coordination", 0.6))
    conflict_h = float(ih.get("conflict_heat", 0.12))
    conflict_a = float(ih.get("conflict_heat", 0.12))
    ref_strict = float((referee or {}).get("strictness", 0.55))

    dt = cfg.dt_default
    duration = float(match_seconds if match_seconds is not None else cfg.match_seconds)
    n_ticks = max(1, int(duration / dt))

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
        match_seconds=duration,
        use_tick_fouls=bool(getattr(cfg, "discipline_tick_fouls", True)),
    )
    ev_idx = 0
    meso = MesoAggregator(cfg.meso_window_seconds)
    timeline: List[str] = []
    strictness_sum = 0.0
    poss_home_ticks = 0
    phi_sum_h = 0.0
    phi_sum_a = 0.0
    subs_done: dict = {}
    cognitive_bus, cognitive_executor = _resolve_cognitive_layer(cfg, seed, home_agent, away_agent)
    processed_subs: set = set()

    wm_recorder = getattr(home_agent, "_wm_recorder", None) or getattr(away_agent, "_wm_recorder", None)
    state._wm_recorder = wm_recorder
    wm_cfg = wm_runtime.cfg if wm_runtime is not None else None

    for tick in range(n_ticks):
        t0 = tick * dt
        t1 = t0 + dt
        state.clock_seconds = t1
        state._wm_horizon_s = dt
        if wm_recorder is not None or wm_runtime is not None:
            from src.match_engine.world_model.observation import encode_observation

            att = state.ball.possession_team_id == state.home.team_id
            state._wm_obs_pre = encode_observation(state, attacking_home=att, cfg=wm_cfg)
            state._wm_last_action = None
        tick_events = []
        while ev_idx < len(schedule) and schedule[ev_idx].t_sec < t1:
            if schedule[ev_idx].t_sec >= t0:
                tick_events.append(schedule[ev_idx])
            ev_idx += 1

        score_diff_home = state.home.score - state.away.score
        xg_swing_home = xg_home - xg_away
        press_h = float(state.home.coach.tactical_current.get("pressing_intensity", 0.5))
        press_a = float(state.away.coach.tactical_current.get("pressing_intensity", 0.5))

        tactical_eng.step_in_match_pressure(state, dt)

        if cfg.enable_spatial:
            spatial.step(state, dt)
            sie.step(state)

        aff_events = _events_for_affective(tick_events, cfg)
        mod_dict, scalars = affective.step(
            state,
            dt,
            events=aff_events,
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
        mod_home = mod_dict["home"]
        mod_away = mod_dict["away"]

        if getattr(cfg, "discipline_tick_fouls", True):
            from src.match_engine.discipline_schedule import emit_tick_discipline

            from src.match_engine.discipline_schedule import emit_tick_cross, emit_tick_tackle

            lam_f = float(scalars.get("lambda_foul", cfg.lambda_foul_base))
            for dev in emit_tick_discipline(
                state,
                mod_home,
                mod_away,
                cfg=cfg,
                lambda_foul=lam_f,
                referee_strictness=ref_strict,
                drama_score=drama_score,
                rng=rng,
                player_tracker=None,
            ):
                apply_micro_event(state, dev, cfg)
                tick_events.append(dev)
            for tev in emit_tick_tackle(state, mod_home, mod_away, cfg=cfg, lambda_foul=lam_f, rng=rng):
                apply_micro_event(state, tev, cfg)
                tick_events.append(tev)
            emit_tick_cross(state, aerial, cfg=cfg, rng=rng)

        if state.ball.possession_team_id == state.home.team_id:
            poss_home_ticks += 1
            state.home.possession_share = 0.92 * state.home.possession_share + 0.08
        else:
            state.home.possession_share = 0.92 * state.home.possession_share
        state.away.possession_share = 1.0 - state.home.possession_share

        if cfg.enable_spatial:
            kinematic.step(state, mod_home, mod_away)
            phi_sum_h += float(np.mean(state.spatial.phi_home))
            phi_sum_a += float(np.mean(state.spatial.phi_away))

        for ev in tick_events:
            player_tracker.apply_event(ev)

        if cfg.enable_phase3:
            if tick_events:
                _resolve_scheduled_shots(state, tick_events, shots, cfg, mod_home, mod_away, rng)
            _, act_events = actions.step(state, mod_home, mod_away, rng)
            for ev in act_events:
                player_tracker.apply_event(ev)
            _apply_phase3_events(state, act_events, cfg)
        elif cfg.enable_passing:
            _, pass_events = passing.step(state, mod_home, mod_away, rng)
            for pe in pass_events:
                apply_micro_event(state, pe, cfg)

        player_tracker.tick_minutes(state, dt)
        sub_notes = maybe_apply_substitutions(state, t1, rng, player_tracker, subs_done=subs_done)
        for sn in sub_notes:
            timeline.append(sn)

        _process_cognitive_tick(
            cognitive_bus,
            cognitive_executor,
            state,
            tick_events,
            t1,
            dt,
            xg_swing_home,
            home_agent,
            away_agent,
            processed_subs,
            player_tracker,
        )

        if wm_recorder is not None and getattr(state, "_wm_obs_pre", None) is not None:
            act = getattr(state, "_wm_last_action", None)
            if act is not None:
                from src.match_engine.world_model.observation import encode_observation

                att = state.ball.possession_team_id == state.home.team_id
                obs_next = encode_observation(state, attacking_home=att, cfg=wm_cfg)
                wm_recorder.write(
                    state._wm_obs_pre,
                    act,
                    obs_next,
                    meta={
                        "t_sec": t1,
                        "tick": tick,
                        "outcome_supervised": bool(act[0] > 0.5 or act[1] > 0.5),
                    },
                )

        if state.ball.possessor_id:
            for p in state.home.players + state.away.players:
                if p.player_id == state.ball.possessor_id:
                    state.ball.position = 0.85 * state.ball.position + 0.15 * p.position
                    break

        emo_h = affective.team_emotion_mean(state.home)
        emo_a = affective.team_emotion_mean(state.away)
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
    emo_home = affective.team_emotion_mean(state.home)
    emo_away = affective.team_emotion_mean(state.away)

    def tactical_drift(team) -> float:
        base = team.coach.tactical_base
        cur = team.coach.tactical_current
        return float(sum(abs(cur.get(k, 0.5) - base.get(k, 0.5)) for k in base))

    lam_h, lam_a = lambdas_from_micro(state.micro_xg_home, state.micro_xg_away, eff_h, eff_a, cfg)

    physics_gh = int(shots.stats["home_goals"]) + int(aerial.stats.get("home_aerial_goals", 0))
    physics_ga = int(shots.stats["away_goals"]) + int(aerial.stats.get("away_aerial_goals", 0))
    xg_supplement_meta: dict = {}
    if cfg.use_micro_goals:
        final_gh, final_ga = physics_gh, physics_ga
        from src.simulation.score_path import xg_supplement_allowed

        if xg_supplement_allowed():
            from src.match_engine.goal_generator import supplement_goals_from_micro_xg

            final_gh, final_ga, xg_supplement_meta = supplement_goals_from_micro_xg(
                physics_gh,
                physics_ga,
                state.micro_xg_home,
                state.micro_xg_away,
                cfg,
                rng=np.random.default_rng(seed + 909),
            )
    else:
        final_gh, final_ga = int(goals_home), int(goals_away)

    state.home.score = final_gh
    state.away.score = final_ga

    ha, aa = passing.stats["home_attempts"], passing.stats["away_attempts"]
    hc, ac = passing.stats["home_completed"], passing.stats["away_completed"]
    discipline = player_tracker.team_discipline_totals(state)
    home_disc = discipline.get(state.home.team_id, {})
    away_disc = discipline.get(state.away.team_id, {})

    summary = MicroMatchSummary(
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
        possession_home=poss_home_ticks / max(1, n_ticks),
        passes_home=ha,
        passes_away=aa,
        pass_completion_home=hc / max(1, ha),
        pass_completion_away=ac / max(1, aa),
        through_balls_home=passing.stats["home_through"],
        through_balls_away=passing.stats["away_through"],
        long_passes_home=passing.stats["home_long"],
        long_passes_away=passing.stats["away_long"],
        curved_passes_home=passing.stats["home_curved"],
        curved_passes_away=passing.stats["away_curved"],
        pass_curve_omega_mean=(
            passing.stats["curve_omega_sum"] / max(1, passing.stats["curve_pass_count"])
        ),
        outside_foot_passes_home=passing.stats["home_outside"],
        outside_foot_passes_away=passing.stats["away_outside"],
        ground_passes_home=passing.stats["home_ground"],
        ground_passes_away=passing.stats["away_ground"],
        pass_intercepts_home=passing.stats["home_intercepts"],
        pass_intercepts_away=passing.stats["away_intercepts"],
        wall_passes_home=passing.stats["home_wall"],
        wall_passes_away=passing.stats["away_wall"],
        wall_combos_home=passing.stats["home_wall_ok"],
        wall_combos_away=passing.stats["away_wall_ok"],
        phi_integral_home=phi_sum_h / max(1, n_ticks),
        phi_integral_away=phi_sum_a / max(1, n_ticks),
        micro_xg_home=state.micro_xg_home,
        micro_xg_away=state.micro_xg_away,
        goals_micro_home=final_gh,
        goals_micro_away=final_ga,
        goals_physics_home=physics_gh,
        goals_physics_away=physics_ga,
        xg_supplement_applied=bool(xg_supplement_meta.get("supplemented")),
        xg_supplement_meta=xg_supplement_meta,
        shots_home=shots.stats["home_shots"],
        shots_away=shots.stats["away_shots"],
        shots_scheduled_home=shots.stats["home_shots_scheduled"],
        shots_scheduled_away=shots.stats["away_shots_scheduled"],
        shots_on_target_home=shots.stats["home_on_target"],
        shots_on_target_away=shots.stats["away_on_target"],
        fouls_committed_home=int(home_disc.get("fouls_committed", 0)),
        fouls_committed_away=int(away_disc.get("fouls_committed", 0)),
        yellow_cards_home=int(home_disc.get("yellow_cards", 0)),
        yellow_cards_away=int(away_disc.get("yellow_cards", 0)),
        red_cards_home=int(home_disc.get("red_cards", 0)),
        red_cards_away=int(away_disc.get("red_cards", 0)),
        tackles_home=int(home_disc.get("tackles", 0)),
        tackles_away=int(away_disc.get("tackles", 0)),
        curved_shots=shots.stats["curved"],
        knuckle_shots=shots.stats["knuckle"],
        headers_attempted=int(
            aerial.stats.get("header_attempts", aerial.stats.get("headers", 0))
            + shots.stats.get("home_headers", 0)
            + shots.stats.get("away_headers", 0)
        ),
        crosses_attempted=aerial.stats["crosses"],
        lambda_home=lam_h,
        lambda_away=lam_a,
        player_stats=player_tracker.export_by_team(state),
        substitutions=player_tracker.substitutions,
        cognitive_triggers=(
            [t.to_dict() for t in cognitive_bus.fired_log] if cognitive_bus is not None else []
        ),
        cognitive_plans=(
            cognitive_executor.export_log() if cognitive_executor is not None else []
        ),
        cognitive_tier_usage=(
            cognitive_bus.tier_usage() if cognitive_bus is not None else {}
        ),
        meso_packets=[p.__dict__ if hasattr(p, "__dict__") else p for p in packets],
        timeline_snippet=timeline[:12],
        continuous_clock=continuous_clock.diagnostics() if continuous_clock is not None else {"enabled": False},
        subtick_reception_queue=subtick_queue.diagnostics() if subtick_queue is not None else {"enabled": False},
    )

    if writeback_agents:
        apply_affective_endstate_to_agents(home_agent, away_agent, state, emo_home, emo_away, blend=blend)

    from src.match_engine.ball_path_logger import persist_ball_path_log

    summary.ball_log_path = persist_ball_path_log(
        state,
        home_team=state.home.team_id,
        away_team=state.away.team_id,
        stage_name=stage_name,
        base_dir=base_dir,
        score_home=final_gh,
        score_away=final_ga,
    )

    return summary
