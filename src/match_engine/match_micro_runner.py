"""
Full micro match: Phase 1b + 2a + 2b + 3 (affective, spatial, passing, shots/aerial, micro xG→λ).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from src.simulation.runtime import environment_snapshot, env_bool

from src.match_engine.action_engine import ActionEngine
from src.match_engine.affective_coupling import AffectiveSpatialCoupling
from src.match_engine.event_schedule import build_event_schedule
from src.match_engine.goal_generator import lambdas_from_micro
from src.match_engine.internal_signals import normalize_internal_match_signals
from src.match_engine.kinematic_position import KinematicPositionLayer
from src.match_engine.macro_bridge import apply_affective_endstate_to_agents, build_match_affective_state
from src.match_engine.meso_aggregator import MesoAggregator, icon_emotion_shock
from src.match_engine.possession_helpers import (
    mark_shot_taken,
    shot_cooldown_ok,
    turnover_to_defence_after_shot,
)
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEventType, apply_micro_event
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
    world_model_runtime=None,
    base_dir: str = "",
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
    except Exception as exc:
        if cog_cfg.require_llm:
            raise RuntimeError(
                "cognitive strict mode requires a live LLM provider"
            ) from exc
        llm = None
    environment_signature = _policy_environment_signature(cfg, cog_cfg)
    executor = CognitiveExecutor(
        cog_cfg, llm, use_llm=bool(llm),
        world_model_runtime=world_model_runtime,
        policy_environment_signature=environment_signature,
        outcome_residual_memory=(
            _load_policy_residual_memory(
                base_dir, world_model_runtime, cfg, cog_cfg,
            )
            if world_model_runtime is not None else None
        ),
        opponent_meta_belief_memory=(
            _load_opponent_meta_belief_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
            )
            if world_model_runtime is not None else None
        ),
        opponent_response_memory=(
            _load_opponent_response_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
            )
            if world_model_runtime is not None else None
        ),
        llm_critic_memory=(
            _load_llm_critic_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
                str(getattr(llm, "model", "rule_fallback")),
            )
            if world_model_runtime is not None else None
        ),
        active_probe_memory=(
            _load_active_probe_discovery_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
            )
            if world_model_runtime is not None else None
        ),
        predictive_mechanism_memory=(
            _load_predictive_mechanism_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
            )
            if world_model_runtime is not None else None
        ),
        predictive_mechanism_chain_memory=(
            _load_predictive_mechanism_chain_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
            )
            if world_model_runtime is not None else None
        ),
        predictive_mechanism_chain_fusion_memory=(
            _load_predictive_mechanism_chain_fusion_memory(
                base_dir, world_model_runtime, environment_signature,
                str(getattr(llm, "model", "rule_fallback")),
            )
            if world_model_runtime is not None else None
        ),
        mechanism_stress_memory=(
            _load_mechanism_stress_memory(
                base_dir,
                world_model_runtime,
                environment_signature,
            )
            if world_model_runtime is not None else None
        ),
    )
    return bus, executor


def _policy_environment_signature(cfg, cog_cfg):
    from src.match_engine.world_model.residual_memory import (
        policy_environment_signature,
    )

    return policy_environment_signature(cfg, cog_cfg)


def _load_policy_residual_memory(
    base_dir, world_model_runtime, cfg, cog_cfg,
):
    from src.match_engine.world_model.residual_memory import (
        load_contextual_residual_memory,
    )

    return load_contextual_residual_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=_policy_environment_signature(cfg, cog_cfg),
        min_samples=cog_cfg.world_model_residual_min_samples,
    )


def _load_opponent_meta_belief_memory(
    base_dir, world_model_runtime, environment_signature,
):
    from src.match_engine.world_model.opponent_meta_memory import (
        load_opponent_meta_belief_memory,
    )

    return load_opponent_meta_belief_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
    )


def _load_opponent_response_memory(
    base_dir, world_model_runtime, environment_signature,
):
    from src.match_engine.world_model.opponent_response import (
        load_opponent_response_memory,
    )

    return load_opponent_response_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
    )


def _load_llm_critic_memory(
    base_dir, world_model_runtime, environment_signature, llm_model,
):
    from src.match_engine.world_model.llm_critic import (
        llm_critic_signature,
    )
    from src.match_engine.world_model.llm_critic_memory import (
        load_llm_critic_memory,
    )

    return load_llm_critic_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
        critic_signature=llm_critic_signature(str(llm_model)),
    )


def _load_active_probe_discovery_memory(
    base_dir, world_model_runtime, environment_signature,
):
    from src.match_engine.world_model.active_probe_memory import (
        load_active_probe_discovery_memory,
    )

    return load_active_probe_discovery_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
    )


def _load_predictive_mechanism_memory(
    base_dir, world_model_runtime, environment_signature,
):
    from src.match_engine.world_model.predictive_mechanism_memory import (
        load_predictive_mechanism_memory,
    )

    return load_predictive_mechanism_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
    )


def _load_mechanism_stress_memory(
    base_dir, world_model_runtime, environment_signature,
):
    from src.match_engine.world_model.mechanism_stress_memory import (
        load_mechanism_stress_memory,
    )

    return load_mechanism_stress_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
    )


def _load_predictive_mechanism_chain_memory(
    base_dir, world_model_runtime, environment_signature,
):
    from src.match_engine.world_model.predictive_mechanism_chain_memory import (
        load_predictive_mechanism_chain_memory,
    )

    return load_predictive_mechanism_chain_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
    )


def _load_predictive_mechanism_chain_fusion_memory(
    base_dir, world_model_runtime, environment_signature, model_name,
):
    from src.match_engine.world_model.predictive_mechanism_chain import (
        predictive_mechanism_chain_llm_signature,
    )
    from src.match_engine.world_model.predictive_mechanism_chain_fusion_memory import (
        load_predictive_mechanism_chain_fusion_memory,
    )

    return load_predictive_mechanism_chain_fusion_memory(
        base_dir,
        checkpoint_signature=str(getattr(
            world_model_runtime, "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(environment_signature),
        llm_signature=predictive_mechanism_chain_llm_signature(model_name),
    )


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


def _resolve_final_micro_score(
    state, shots, aerial, cfg, *, goals_home, goals_away, seed,
):
    physics_home = (
        int(shots.stats["home_goals"])
        + int(aerial.stats.get("home_aerial_goals", 0))
    )
    physics_away = (
        int(shots.stats["away_goals"])
        + int(aerial.stats.get("away_aerial_goals", 0))
    )
    supplement_meta: dict = {}
    if not cfg.use_micro_goals:
        final_home, final_away = int(goals_home), int(goals_away)
    else:
        final_home, final_away = physics_home, physics_away
        from src.simulation.score_path import xg_supplement_allowed

        if xg_supplement_allowed():
            from src.match_engine.goal_generator import supplement_goals_from_micro_xg

            final_home, final_away, supplement_meta = supplement_goals_from_micro_xg(
                physics_home, physics_away,
                state.micro_xg_home, state.micro_xg_away, cfg,
                rng=np.random.default_rng(seed + 909),
            )
    state.home.score, state.away.score = final_home, final_away
    return final_home, final_away, physics_home, physics_away, supplement_meta


def _tactical_drift(team) -> float:
    base = team.coach.tactical_base
    current = team.coach.tactical_current
    return float(
        sum(abs(current.get(key, 0.5) - base.get(key, 0.5)) for key in base)
    )


def _tactical_vector_snapshot(values) -> Dict[str, float]:
    from src.match_engine.tactical_catalog import TACTICAL_KEYS

    source = values if isinstance(values, dict) else {}
    return {
        key: round(float(source.get(key, 0.5)), 6)
        for key in TACTICAL_KEYS
    }


def _initial_tactical_execution(agent, team) -> Dict[str, Any]:
    memory = getattr(agent, "semantic_memory", None)
    intervention = (
        memory.get("tactical_intervention")
        if isinstance(memory, dict) else None
    )
    intervention = intervention if isinstance(intervention, dict) else {}
    locked = bool(getattr(agent, "_tactical_preset_locked", False))
    native_archetype = str(
        getattr(getattr(agent, "coach_profile", None), "preferred_preset", "")
        or getattr(agent, "style_archetype", "") or "balanced"
    )
    return {
        "schema_version": 1,
        "team": str(team.team_id),
        "applied_tactic": str(
            intervention.get("preset") if locked else "team_identity"
        ),
        "native_archetype": native_archetype,
        "binding_kind": "locked_preset" if locked else "native_team_vector",
        "source": str(
            intervention.get("source")
            or ("unknown_locked_source" if locked else "agent_team_identity")
        ),
        "preset_locked": locked,
        "initial_vector": _tactical_vector_snapshot(team.coach.tactical_base),
    }


def _final_tactical_execution(initial, state) -> Dict[str, Any]:
    result = {}
    for side, team in (("home", state.home), ("away", state.away)):
        row = dict(initial[side])
        final_vector = _tactical_vector_snapshot(team.coach.tactical_current)
        initial_vector = row["initial_vector"]
        row["final_vector"] = final_vector
        row["changed_controls"] = [
            key for key in initial_vector
            if abs(final_vector[key] - initial_vector[key]) > 1e-6
        ]
        row["final_delta_l1"] = round(sum(
            abs(final_vector[key] - initial_vector[key])
            for key in initial_vector
        ), 6)
        result[side] = row
    return result


def _build_micro_match_summary(
    *,
    state, cfg, passing, shots, aerial, player_tracker,
    cognitive_bus, cognitive_executor, continuous_clock, subtick_queue,
    wm_runtime, manager_runtimes, tactical_execution, branch_anchor,
    clock_contract, logical_duration,
    packets, timeline, n_ticks, strictness_sum, poss_home_ticks,
    phi_sum_h, phi_sum_a, emo_home, emo_away, eff_h, eff_a,
    final_gh, final_ga, physics_gh, physics_ga, xg_supplement_meta,
):
    lambda_home, lambda_away = lambdas_from_micro(
        state.micro_xg_home, state.micro_xg_away, eff_h, eff_a, cfg,
    )
    home_attempts = passing.stats["home_attempts"]
    away_attempts = passing.stats["away_attempts"]
    home_completed = passing.stats["home_completed"]
    away_completed = passing.stats["away_completed"]
    discipline = player_tracker.team_discipline_totals(state)
    home_disc = discipline.get(state.home.team_id, {})
    away_disc = discipline.get(state.away.team_id, {})
    from src.match_engine.world_model.decision_adoption import (
        decision_adoption_diagnostics,
    )
    from src.match_engine.world_model.action_adoption import (
        direct_action_adoption_diagnostics,
    )

    return MicroMatchSummary(
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
        tactical_drift_home=_tactical_drift(state.home),
        tactical_drift_away=_tactical_drift(state.away),
        controversy_integral=state.referee.controversy_integral,
        possession_home=poss_home_ticks / max(1, n_ticks),
        passes_home=home_attempts,
        passes_away=away_attempts,
        pass_completion_home=home_completed / max(1, home_attempts),
        pass_completion_away=away_completed / max(1, away_attempts),
        through_balls_home=passing.stats["home_through"],
        through_balls_away=passing.stats["away_through"],
        long_passes_home=passing.stats["home_long"],
        long_passes_away=passing.stats["away_long"],
        curved_passes_home=passing.stats["home_curved"],
        curved_passes_away=passing.stats["away_curved"],
        pass_curve_omega_mean=(
            passing.stats["curve_omega_sum"]
            / max(1, passing.stats["curve_pass_count"])
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
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        player_stats=player_tracker.export_by_team(state),
        substitutions=player_tracker.substitutions,
        cognitive_triggers=(
            [trigger.to_dict() for trigger in cognitive_bus.fired_log]
            if cognitive_bus is not None else []
        ),
        cognitive_plans=(
            cognitive_executor.export_log() if cognitive_executor is not None else []
        ),
        cognitive_tier_usage=(
            cognitive_bus.tier_usage() if cognitive_bus is not None else {}
        ),
        meso_packets=[
            packet.__dict__ if hasattr(packet, "__dict__") else packet
            for packet in packets
        ],
        timeline_snippet=timeline[:12],
        continuous_clock=(
            continuous_clock.diagnostics()
            if continuous_clock is not None else {"enabled": False}
        ),
        subtick_reception_queue=(
            subtick_queue.diagnostics()
            if subtick_queue is not None else {"enabled": False}
        ),
        world_model_online_calibration=(
            wm_runtime.online_calibration_diagnostics()
            if wm_runtime is not None else {"available": False}
        ),
        world_model_decision_adoption=decision_adoption_diagnostics(state),
        world_model_action_adoption=direct_action_adoption_diagnostics(state),
        world_model_branch_anchor=branch_anchor,
        simulation_clock={
            "schema_version": 1,
            "contract": clock_contract,
            "authoritative_tick_clock": (
                clock_contract == "authoritative_tick_v2"
            ),
            "final_logical_sec": float(logical_duration),
            "final_state_clock_sec": float(state.clock_seconds),
        },
        world_model_runtime={
            "loaded": wm_runtime is not None,
            "checkpoint_signature": (
                getattr(wm_runtime, "checkpoint_signature", None)
                if wm_runtime is not None else None
            ),
            "shot_probability_source": (
                getattr(wm_runtime, "shot_probability_source", None)
                if wm_runtime is not None else None
            ),
            "pass_quality": (
                getattr(wm_runtime, "pass_quality", None)
                if wm_runtime is not None else None
            ),
            "shot_quality": (
                getattr(wm_runtime, "shot_quality", None)
                if wm_runtime is not None else None
            ),
            "control_scope": (
                environment_snapshot().get(
                    "MATCH_WM_CONTROL_SCOPE", "both",
                ).strip().lower()
                if wm_runtime is not None else "none"
            ),
            "outcome_aligned_policy": (
                env_bool(
                    environment_snapshot(),
                    "MATCH_WM_OUTCOME_ALIGNED_POLICY",
                    False,
                )
                if wm_runtime is not None else False
            ),
        },
        in_match_management={
            side: runtime.diagnostics()
            for side, runtime in manager_runtimes.items()
        },
        tactical_execution=_final_tactical_execution(tactical_execution, state),
        squad_provenance={
            "schema_version": 1,
            "home": dict(state.home.squad_provenance),
            "away": dict(state.away.squad_provenance),
            "fallback_used": bool(
                state.home.squad_provenance.get("fallback_used")
                or state.away.squad_provenance.get("fallback_used")
            ),
        },
    )


def _load_world_model_runtime(base_dir):
    from src.match_engine.world_model.config import (
        world_model_enabled, world_model_required,
    )

    if not world_model_enabled():
        return None
    from src.match_engine.world_model.inference import WorldModelRuntime

    return WorldModelRuntime.load_default(
        base_dir, required=world_model_required(),
    )


def _build_temporal_adapters(cfg, base_dir):
    continuous_clock = None
    if cfg.enable_continuous_event_clock:
        from src.match_engine.continuous_micro_clock import ContinuousMicroEventClock

        artifact = Path(cfg.continuous_clock_artifact)
        if not artifact.is_absolute():
            artifact = Path(base_dir) / artifact
        continuous_clock = ContinuousMicroEventClock(
            artifact,
            cfg.continuous_clock_provider,
            cfg.continuous_clock_max_delay_s,
        )

    subtick_queue = None
    if cfg.enable_subtick_reception_queue:
        if cfg.enable_continuous_event_clock:
            raise ValueError(
                "legacy continuous gate and sub-tick reception queue are mutually exclusive"
            )
        from src.match_engine.subtick_reception_queue import SubtickReceptionQueue

        temporal_artifact = Path(cfg.continuous_clock_artifact)
        pass_artifact = Path(cfg.subtick_pass_artifact)
        if not temporal_artifact.is_absolute():
            temporal_artifact = Path(base_dir) / temporal_artifact
        if not pass_artifact.is_absolute():
            pass_artifact = Path(base_dir) / pass_artifact
        subtick_queue = SubtickReceptionQueue(
            temporal_artifact,
            pass_artifact,
            cfg.subtick_reception_provider,
            cfg.subtick_response_blend,
        )
    return continuous_clock, subtick_queue


def _build_micro_engines(cfg, base_dir):
    affective = AffectiveSpatialCoupling(cfg)
    spatial = SpatialFieldEngine(cfg)
    spatial_intelligence = SpatialIntelligenceEngine(cfg)
    kinematic = KinematicPositionLayer(cfg, sie=spatial_intelligence)
    world_model = _load_world_model_runtime(base_dir)
    passing = PassingEngine(
        cfg, spatial_intelligence, wm_runtime=world_model, base_dir=base_dir,
    )
    shots = ShotEngine(cfg, spatial_intelligence)
    player_tracker = PlayerMatchStatsTracker()
    passing.player_tracker = player_tracker
    shots.player_tracker = player_tracker
    aerial = AerialDuelEngine(cfg)
    continuous_clock, subtick_queue = _build_temporal_adapters(cfg, base_dir)
    actions = ActionEngine(
        cfg, passing, shots, aerial, wm_runtime=world_model,
        continuous_clock=continuous_clock, subtick_queue=subtick_queue,
    )
    return (
        affective, spatial, spatial_intelligence, kinematic, world_model,
        passing, shots, player_tracker, aerial, continuous_clock,
        subtick_queue, actions, TacticalMicroEngine(cfg),
    )


def _initialize_micro_match_state(
    *, home_agent, away_agent, referee, stage_pressure, neutral_venue,
    eff_h, eff_a, poss_home, cfg, rng, tactical_engine,
    tactical_override_home, tactical_override_away,
    internal_home, internal_away, match_seconds,
    goals_home, goals_away, xg_home, xg_away, drama_score, base_dir,
):
    state = build_match_affective_state(
        home_agent, away_agent, referee=referee,
        stage_pressure=stage_pressure, neutral_venue=neutral_venue,
        eff_status_home=eff_h, eff_status_away=eff_a, rng=rng,
        base_dir=base_dir,
    )
    _init_micro_state(state, cfg, rng, possession_home=poss_home)
    tactical_engine.bootstrap(state, home_agent, away_agent)
    if tactical_override_home or tactical_override_away:
        from src.match_engine.tactical_profile import apply_vector_to_team_coach

        if tactical_override_home:
            apply_vector_to_team_coach(state.home, tactical_override_home)
        if tactical_override_away:
            apply_vector_to_team_coach(state.away, tactical_override_away)

    internal_signals = normalize_internal_match_signals(
        internal_home, internal_away,
    )
    coordination_home = internal_signals.coordination_home
    coordination_away = internal_signals.coordination_away
    conflict_home = internal_signals.conflict_home
    conflict_away = internal_signals.conflict_away
    referee_strictness = float((referee or {}).get("strictness", 0.55))
    dt = cfg.dt_default
    duration = float(match_seconds if match_seconds is not None else cfg.match_seconds)
    n_ticks = max(1, int(duration / dt))
    schedule = build_event_schedule(
        state, goals_home=goals_home, goals_away=goals_away,
        xg_home=xg_home, xg_away=xg_away,
        referee_strictness=referee_strictness, drama_score=drama_score,
        internal_conflict_home=conflict_home,
        internal_conflict_away=conflict_away, rng=rng,
        match_seconds=duration,
        use_tick_fouls=bool(getattr(cfg, "discipline_tick_fouls", True)),
    )
    return {
        "state": state, "coord_h": coordination_home,
        "coord_a": coordination_away, "conflict_h": conflict_home,
        "conflict_a": conflict_away, "ref_strict": referee_strictness,
        "dt": dt, "duration": duration, "n_ticks": n_ticks,
        "schedule": schedule,
    }


def _events_in_tick(schedule, event_index, t0, t1):
    events = []
    while event_index < len(schedule) and schedule[event_index].t_sec < t1:
        if schedule[event_index].t_sec >= t0:
            events.append(schedule[event_index])
        event_index += 1
    return events, event_index


def _begin_world_model_tick(state, *, dt, wm_recorder, wm_runtime, wm_cfg):
    state._wm_horizon_s = dt
    if wm_recorder is None and wm_runtime is None:
        return
    from src.match_engine.world_model.observation import encode_observation

    attacking_home = state.ball.possession_team_id == state.home.team_id
    state._wm_actor_team_id_pre = state.ball.possession_team_id
    state._wm_obs_pre = encode_observation(
        state, attacking_home=attacking_home, cfg=wm_cfg,
    )
    waiting_for_option_followup = any(
        record.get("event_option_resolved_t_sec") is not None
        and record.get("event_option_next_action") is None
        and not record.get("event_option_followup_expired")
        and str(record.get("team_id")) == str(state._wm_actor_team_id_pre)
        for record in (
            getattr(state, "_wm_coach_decision_adoption", None) or []
        )
    )
    if waiting_for_option_followup:
        from src.match_engine.world_model.policy_outcomes import (
            capture_policy_outcome_baseline,
        )

        state._wm_outcome_baseline_pre = capture_policy_outcome_baseline(
            state, team_id=str(state._wm_actor_team_id_pre),
        )
    else:
        state._wm_outcome_baseline_pre = None
    state._wm_last_action = None


def _step_affective_discipline(
    *, state, cfg, dt, tick_events, affective, tactical_engine,
    spatial, spatial_intelligence, coordination_home, coordination_away,
    conflict_home, conflict_away, xg_swing_home, referee_strictness,
    drama_score, aerial, rng,
):
    press_home = float(
        state.home.coach.tactical_current.get("pressing_intensity", 0.5)
    )
    press_away = float(
        state.away.coach.tactical_current.get("pressing_intensity", 0.5)
    )
    tactical_engine.step_in_match_pressure(state, dt)
    if cfg.enable_spatial:
        spatial.step(state, dt)
        spatial_intelligence.step(state)
    modifiers, scalars = affective.step(
        state, dt, events=_events_for_affective(tick_events, cfg),
        score_diff_home=state.home.score - state.away.score,
        xg_swing_home=xg_swing_home,
        coordination_home=coordination_home,
        coordination_away=coordination_away,
        conflict_home=conflict_home, conflict_away=conflict_away,
        press_home=press_home, press_away=press_away,
    )
    mod_home, mod_away = modifiers["home"], modifiers["away"]
    if getattr(cfg, "discipline_tick_fouls", True):
        from src.match_engine.discipline_schedule import (
            emit_tick_cross,
            emit_tick_discipline,
            emit_tick_tackle,
        )

        foul_rate = float(scalars.get("lambda_foul", cfg.lambda_foul_base))
        discipline_events = emit_tick_discipline(
            state, mod_home, mod_away, cfg=cfg, lambda_foul=foul_rate,
            referee_strictness=referee_strictness,
            drama_score=drama_score, rng=rng, player_tracker=None,
        )
        for event in discipline_events:
            apply_micro_event(state, event, cfg)
            tick_events.append(event)
        tackle_events = emit_tick_tackle(
            state, mod_home, mod_away, cfg=cfg,
            lambda_foul=foul_rate, rng=rng,
        )
        for event in tackle_events:
            apply_micro_event(state, event, cfg)
            tick_events.append(event)
        cross_events = emit_tick_cross(state, aerial, cfg=cfg, rng=rng)
        for event in cross_events:
            apply_micro_event(state, event, cfg)
            tick_events.append(event)
    return mod_home, mod_away, scalars


def _step_possession_spatial(
    state, cfg, kinematic, mod_home, mod_away,
):
    home_possession = state.ball.possession_team_id == state.home.team_id
    if home_possession:
        state.home.possession_share = 0.92 * state.home.possession_share + 0.08
    else:
        state.home.possession_share = 0.92 * state.home.possession_share
    state.away.possession_share = 1.0 - state.home.possession_share
    phi_home = phi_away = 0.0
    if cfg.enable_spatial:
        kinematic.step(state, mod_home, mod_away)
        phi_home = float(np.mean(state.spatial.phi_home))
        phi_away = float(np.mean(state.spatial.phi_away))
    return int(home_possession), phi_home, phi_away


def _execute_tick_actions(
    *, state, cfg, tick_events, player_tracker, shots, actions,
    passing, mod_home, mod_away, rng,
):
    for event in tick_events:
        player_tracker.apply_event(event)
    if cfg.enable_phase3:
        if tick_events:
            _resolve_scheduled_shots(
                state, tick_events, shots, cfg, mod_home, mod_away, rng,
            )
        _, action_events = actions.step(state, mod_home, mod_away, rng)
        for event in action_events:
            player_tracker.apply_event(event)
        _apply_phase3_events(state, action_events, cfg)
    elif cfg.enable_passing:
        _, pass_events = passing.step(state, mod_home, mod_away, rng)
        for event in pass_events:
            apply_micro_event(state, event, cfg)


def _finish_world_model_tick(
    state, *, wm_recorder, wm_runtime, wm_cfg, t1, tick,
):
    if (
        wm_recorder is None and wm_runtime is None
    ) or getattr(state, "_wm_obs_pre", None) is None:
        return
    action = getattr(state, "_wm_last_action", None)
    if action is None:
        return
    from src.match_engine.world_model.observation import encode_observation

    attacking_home = state.ball.possession_team_id == state.home.team_id
    observation_next = encode_observation(
        state, attacking_home=attacking_home, cfg=wm_cfg,
    )
    from src.match_engine.world_model.action_codec import decode_action_kind
    from src.match_engine.world_model.decision_adoption import observe_executed_action
    from src.match_engine.world_model.policy_outcomes import (
        observe_policy_intervention_outcomes,
    )

    observe_executed_action(
        state,
        team_id=str(getattr(state, "_wm_actor_team_id_pre", "")),
        action_kind=decode_action_kind(action),
        t_sec=t1,
        outcome_baseline=getattr(
            state, "_wm_outcome_baseline_pre", None,
        ),
    )
    observe_policy_intervention_outcomes(
        state,
        t_sec=t1,
    )
    from src.match_engine.world_model.action_adoption import (
        resolve_action_policy_outcome,
    )

    resolve_action_policy_outcome(
        state,
        state._wm_obs_pre,
        observation_next,
        attacking_home=(
            str(getattr(state, "_wm_actor_team_id_pre", ""))
            == str(state.home.team_id)
        ),
    )
    calibration = None
    if wm_runtime is not None:
        try:
            calibration = wm_runtime.observe_transition(
                state._wm_obs_pre, action, observation_next,
            )
            state._wm_online_last = calibration
        except (RuntimeError, TypeError, ValueError) as exc:
            state._wm_online_error = type(exc).__name__
    if wm_recorder is not None:
        wm_recorder.write(
            state._wm_obs_pre, action, observation_next,
            meta={
                "t_sec": t1, "tick": tick,
                "outcome_supervised": bool(
                    action[0] > 0.5 or action[1] > 0.5
                ),
                "online_calibration": calibration,
            },
        )


def _record_tick_meso(state, affective, meso, *, xg_home, xg_away):
    emotion_home = affective.team_emotion_mean(state.home)
    emotion_away = affective.team_emotion_mean(state.away)
    for team, team_id, xg_for, xg_against, emotion in (
        (state.home, state.home.team_id, xg_home, xg_away, emotion_home),
        (state.away, state.away.team_id, xg_away, xg_home, emotion_away),
    ):
        meso.record_tick(
            state, team_id=team_id, xg_for=xg_for, xg_against=xg_against,
            controversy=state.referee.controversy_integral,
            psi=state.crowd.psi, emotion=emotion,
            tactical=team.coach.tactical_current,
            tactical_base=team.coach.tactical_base,
            icon_shock=icon_emotion_shock(team),
        )


def _sync_ball_to_possessor(state):
    if not state.ball.possessor_id:
        return
    for player in state.home.players + state.away.players:
        if player.player_id == state.ball.possessor_id:
            state.ball.position = (
                0.85 * state.ball.position + 0.15 * player.position
            )
            return


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
    in_match_plan_home: Optional[Dict[str, Any]] = None,
    in_match_plan_away: Optional[Dict[str, Any]] = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> MicroMatchSummary:
    cfg = config or MicroMatchConfig()
    rng = np.random.default_rng(seed)
    eff_h = float(eff_status_home if eff_status_home is not None else home_agent.status_score)
    eff_a = float(eff_status_away if eff_status_away is not None else away_agent.status_score)
    poss_home = _possession_prior_from_strength(eff_h, eff_a, cfg, neutral_venue=neutral_venue)

    resolved_base_dir = os.path.abspath(
        os.fspath(base_dir) if base_dir is not None
        else os.path.join(os.path.dirname(__file__), "..", "..")
    )
    (
        affective, spatial, sie, kinematic, wm_runtime, passing, shots,
        player_tracker, aerial, continuous_clock, subtick_queue, actions,
        tactical_eng,
    ) = _build_micro_engines(cfg, resolved_base_dir)

    initialized = _initialize_micro_match_state(
        home_agent=home_agent, away_agent=away_agent, referee=referee,
        stage_pressure=stage_pressure, neutral_venue=neutral_venue,
        eff_h=eff_h, eff_a=eff_a, poss_home=poss_home, cfg=cfg, rng=rng,
        tactical_engine=tactical_eng,
        tactical_override_home=tactical_override_home,
        tactical_override_away=tactical_override_away,
        internal_home=internal_home, internal_away=internal_away,
        match_seconds=match_seconds, goals_home=goals_home,
        goals_away=goals_away, xg_home=xg_home, xg_away=xg_away,
        drama_score=drama_score, base_dir=resolved_base_dir,
    )
    state = initialized["state"]
    tactical_execution = {
        "home": _initial_tactical_execution(home_agent, state.home),
        "away": _initial_tactical_execution(away_agent, state.away),
    }
    manager_runtimes = {}
    from src.match_engine.manager_plan import InMatchPlan, InMatchPlanRuntime

    for side, payload, expected_team in (
        ("home", in_match_plan_home, state.home.team_id),
        ("away", in_match_plan_away, state.away.team_id),
    ):
        if payload is None:
            continue
        manager_plan = InMatchPlan.from_payload(payload)
        if manager_plan.team != expected_team:
            raise ValueError(f"{side} in-match plan team identity mismatch")
        manager_runtimes[side] = InMatchPlanRuntime(manager_plan)
    manager_substitution_teams = {
        runtime.plan.team for runtime in manager_runtimes.values()
        if runtime.plan.controls_substitutions
    }
    state._wm_policy_experiment_seed = int(seed)
    coord_h, coord_a = initialized["coord_h"], initialized["coord_a"]
    conflict_h, conflict_a = initialized["conflict_h"], initialized["conflict_a"]
    ref_strict = initialized["ref_strict"]
    dt, duration, n_ticks = (
        initialized["dt"], initialized["duration"], initialized["n_ticks"],
    )
    schedule = initialized["schedule"]
    ev_idx = 0
    meso = MesoAggregator(cfg.meso_window_seconds)
    timeline: List[str] = []
    strictness_sum = 0.0
    poss_home_ticks = 0
    phi_sum_h = 0.0
    phi_sum_a = 0.0
    subs_done: dict = {}
    cognitive_bus, cognitive_executor = _resolve_cognitive_layer(
        cfg, seed, home_agent, away_agent,
        world_model_runtime=wm_runtime,
        base_dir=resolved_base_dir,
    )
    processed_subs: set = set()

    wm_recorder = getattr(home_agent, "_wm_recorder", None) or getattr(away_agent, "_wm_recorder", None)
    state._wm_recorder = wm_recorder
    wm_cfg = wm_runtime.cfg if wm_runtime is not None else None
    from src.match_engine.world_model.config import world_model_branch_at_sec

    branch_at_sec = world_model_branch_at_sec()
    authoritative_clock = env_bool(
        environment_snapshot(), "MATCH_AUTHORITATIVE_TICK_CLOCK", False,
    )
    clock_contract = (
        "authoritative_tick_v2" if authoritative_clock
        else "legacy_affective_offset_v1"
    )
    branch_anchor: Dict[str, Any] = {
        "schema_version": 1,
        "available": False,
        "reason": "no_branch_requested",
        "resume_capability": "deterministic_replay_only",
    }

    for tick in range(n_ticks):
        t0 = tick * dt
        t1 = t0 + dt
        state.clock_seconds = t0 if authoritative_clock else t1
        _begin_world_model_tick(
            state, dt=dt, wm_recorder=wm_recorder,
            wm_runtime=wm_runtime, wm_cfg=wm_cfg,
        )
        tick_events, ev_idx = _events_in_tick(schedule, ev_idx, t0, t1)
        xg_swing_home = xg_home - xg_away
        mod_home, mod_away, scalars = _step_affective_discipline(
            state=state, cfg=cfg, dt=dt, tick_events=tick_events,
            affective=affective, tactical_engine=tactical_eng,
            spatial=spatial, spatial_intelligence=sie,
            coordination_home=coord_h, coordination_away=coord_a,
            conflict_home=conflict_h, conflict_away=conflict_a,
            xg_swing_home=xg_swing_home, referee_strictness=ref_strict,
            drama_score=drama_score, aerial=aerial, rng=rng,
        )
        if authoritative_clock:
            state.clock_seconds = t1
        for runtime in manager_runtimes.values():
            timeline.extend(runtime.step(
                state, clock_sec=t1, tracker=player_tracker,
                subs_done=subs_done,
            ))
        strictness_sum += scalars["strictness_effective"]
        possession_tick, phi_home, phi_away = _step_possession_spatial(
            state, cfg, kinematic, mod_home, mod_away,
        )
        poss_home_ticks += possession_tick
        phi_sum_h += phi_home
        phi_sum_a += phi_away
        if (
            branch_at_sec is not None
            and not branch_anchor.get("available")
            and t1 >= branch_at_sec
        ):
            from src.match_engine.branch_anchor import capture_branch_anchor

            branch_anchor = capture_branch_anchor(
                state, rng, requested_sec=branch_at_sec, actual_sec=t1,
                clock_contract=clock_contract, tick=tick,
                event_cursor=ev_idx, passing=passing, shots=shots,
                aerial=aerial, player_tracker=player_tracker,
                continuous_clock=continuous_clock, subtick_queue=subtick_queue,
                manager_runtimes=manager_runtimes,
            )
        from src.match_engine.world_model.config import world_model_plan_clock

        with world_model_plan_clock(t1):
            _execute_tick_actions(
                state=state, cfg=cfg, tick_events=tick_events,
                player_tracker=player_tracker, shots=shots, actions=actions,
                passing=passing, mod_home=mod_home, mod_away=mod_away, rng=rng,
            )
        player_tracker.tick_minutes(state, dt)
        sub_notes = maybe_apply_substitutions(
            state, t1, rng, player_tracker, subs_done=subs_done,
            disabled_team_ids=manager_substitution_teams,
        )
        timeline.extend(sub_notes)
        _process_cognitive_tick(
            cognitive_bus, cognitive_executor, state, tick_events, t1, dt,
            xg_swing_home, home_agent, away_agent, processed_subs,
            player_tracker,
        )
        _finish_world_model_tick(
            state, wm_recorder=wm_recorder, wm_runtime=wm_runtime,
            wm_cfg=wm_cfg, t1=t1, tick=tick,
        )
        _sync_ball_to_possessor(state)
        _record_tick_meso(
            state, affective, meso, xg_home=xg_home, xg_away=xg_away,
        )
        timeline.extend(
            f"{int(t0 // 60)}' {event.event_type.value} ({event.team_id})"
            for event in tick_events
            if event.event_type.value
            in ("goal_scored", "red_card", "var_controversy")
        )

    from src.match_engine.world_model.decision_adoption import (
        finalize_decision_adoption,
    )

    finalize_decision_adoption(state, t_sec=duration)
    packets = meso.flush_packets(state.home.team_id, state.away.team_id)
    emo_home = affective.team_emotion_mean(state.home)
    emo_away = affective.team_emotion_mean(state.away)
    final_gh, final_ga, physics_gh, physics_ga, xg_supplement_meta = (
        _resolve_final_micro_score(
            state, shots, aerial, cfg,
            goals_home=goals_home, goals_away=goals_away, seed=seed,
        )
    )
    summary = _build_micro_match_summary(
        state=state, cfg=cfg, passing=passing, shots=shots, aerial=aerial,
        player_tracker=player_tracker, cognitive_bus=cognitive_bus,
        cognitive_executor=cognitive_executor,
        continuous_clock=continuous_clock, subtick_queue=subtick_queue,
        wm_runtime=wm_runtime, manager_runtimes=manager_runtimes,
        tactical_execution=tactical_execution, branch_anchor=branch_anchor,
        clock_contract=clock_contract, logical_duration=duration,
        packets=packets, timeline=timeline, n_ticks=n_ticks,
        strictness_sum=strictness_sum, poss_home_ticks=poss_home_ticks,
        phi_sum_h=phi_sum_h, phi_sum_a=phi_sum_a,
        emo_home=emo_home, emo_away=emo_away, eff_h=eff_h, eff_a=eff_a,
        final_gh=final_gh, final_ga=final_ga,
        physics_gh=physics_gh, physics_ga=physics_ga,
        xg_supplement_meta=xg_supplement_meta,
    )

    if writeback_agents:
        apply_affective_endstate_to_agents(home_agent, away_agent, state, emo_home, emo_away, blend=blend)

    from src.match_engine.ball_path_logger import persist_ball_path_log

    summary.ball_log_path = persist_ball_path_log(
        state,
        home_team=state.home.team_id,
        away_team=state.away.team_id,
        stage_name=stage_name,
        base_dir=resolved_base_dir,
        score_home=final_gh,
        score_away=final_ga,
    )

    return summary
