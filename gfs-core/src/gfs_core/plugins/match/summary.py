from __future__ import annotations

from gfs_core.contracts.micro_state import MicroMatchSummary

TACTICAL_KEYS = (
    "pressing_intensity", "risk_budget", "line_height", "rotation_aggressiveness",
    "possession_orientation", "verticality", "width_play", "tempo",
    "counter_attack", "high_press", "low_block", "wing_focus",
    "cross_frequency", "through_ball_bias", "long_ball_bias", "build_up_short",
    "offside_trap", "counterpress", "man_oriented_press", "target_man",
    "overlap_fullbacks", "compactness",
)


def initial_tactical_execution(team) -> dict:
    return {
        "schema_version": 1,
        "team": team.team_id,
        "applied_tactic": "team_identity",
        "binding_kind": "native_team_vector",
        "source": "agent_team_identity",
        "initial_vector": _vector(team.coach.tactical_base),
    }


def build_summary(
    *,
    state,
    config,
    scoring,
    passing,
    shots,
    aerial,
    tracker,
    packets,
    timeline,
    ticks,
    strictness_sum,
    possession_ticks,
    phi_home_sum,
    phi_away_sum,
    emotion_home,
    emotion_away,
    effective_home,
    effective_away,
    final_home,
    final_away,
    physics_home,
    physics_away,
    supplement_meta,
    tactical_execution,
    clock_contract,
    duration,
) -> MicroMatchSummary:
    lambda_home, lambda_away = scoring.lambdas_from_micro(
        state.micro_xg_home,
        state.micro_xg_away,
        effective_home,
        effective_away,
        config,
    )
    home_attempts = passing.stats["home_attempts"]
    away_attempts = passing.stats["away_attempts"]
    discipline = tracker.team_discipline_totals(state)
    home_discipline = discipline.get(state.home.team_id, {})
    away_discipline = discipline.get(state.away.team_id, {})
    return MicroMatchSummary(
        home_team=state.home.team_id,
        away_team=state.away.team_id,
        ticks=ticks,
        final_psi=state.crowd.psi,
        home_coach_stress=state.home.coach.stress(),
        away_coach_stress=state.away.coach.stress(),
        ref_strictness_mean=strictness_sum / max(1, ticks),
        home_emotion_mean=emotion_home,
        away_emotion_mean=emotion_away,
        icon_shock_home=_icon_shock(state.home),
        icon_shock_away=_icon_shock(state.away),
        tactical_drift_home=_drift(state.home),
        tactical_drift_away=_drift(state.away),
        controversy_integral=state.referee.controversy_integral,
        possession_home=possession_ticks / max(1, ticks),
        passes_home=home_attempts,
        passes_away=away_attempts,
        pass_completion_home=passing.stats["home_completed"] / max(1, home_attempts),
        pass_completion_away=passing.stats["away_completed"] / max(1, away_attempts),
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
        phi_integral_home=phi_home_sum / max(1, ticks),
        phi_integral_away=phi_away_sum / max(1, ticks),
        micro_xg_home=state.micro_xg_home,
        micro_xg_away=state.micro_xg_away,
        goals_micro_home=final_home,
        goals_micro_away=final_away,
        goals_physics_home=physics_home,
        goals_physics_away=physics_away,
        xg_supplement_applied=bool(supplement_meta.get("supplemented")),
        xg_supplement_meta=supplement_meta,
        shots_home=shots.stats["home_shots"],
        shots_away=shots.stats["away_shots"],
        shots_scheduled_home=shots.stats["home_shots_scheduled"],
        shots_scheduled_away=shots.stats["away_shots_scheduled"],
        shots_on_target_home=shots.stats["home_on_target"],
        shots_on_target_away=shots.stats["away_on_target"],
        fouls_committed_home=int(home_discipline.get("fouls_committed", 0)),
        fouls_committed_away=int(away_discipline.get("fouls_committed", 0)),
        yellow_cards_home=int(home_discipline.get("yellow_cards", 0)),
        yellow_cards_away=int(away_discipline.get("yellow_cards", 0)),
        red_cards_home=int(home_discipline.get("red_cards", 0)),
        red_cards_away=int(away_discipline.get("red_cards", 0)),
        tackles_home=int(home_discipline.get("tackles", 0)),
        tackles_away=int(away_discipline.get("tackles", 0)),
        curved_shots=shots.stats["curved"],
        knuckle_shots=shots.stats["knuckle"],
        headers_attempted=int(
            aerial.stats.get("header_attempts", aerial.stats.get("headers", 0))
            + shots.stats["home_headers"]
            + shots.stats["away_headers"]
        ),
        crosses_attempted=aerial.stats["crosses"],
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        player_stats=tracker.export_by_team(state),
        substitutions=tracker.substitutions,
        meso_packets=[
            packet.__dict__ if hasattr(packet, "__dict__") else packet
            for packet in packets
        ],
        timeline_snippet=timeline[:12],
        simulation_clock={
            "schema_version": 1,
            "contract": clock_contract,
            "authoritative_tick_clock": clock_contract == "authoritative_tick_v2",
            "final_logical_sec": float(duration),
            "final_state_clock_sec": float(state.clock_seconds),
        },
        score_mode=config.score_mode,
        score_provenance={
            "official_from_physics": config.physics_goals,
            "physics_goals": [physics_home, physics_away],
            "official_goals": [final_home, final_away],
            "supplement": supplement_meta,
            "supplement_recorded": bool(supplement_meta),
        },
        in_match_management={},
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


def _drift(team) -> float:
    base = team.coach.tactical_base
    current = team.coach.tactical_current
    return float(sum(abs(current.get(key, 0.5) - base.get(key, 0.5)) for key in base))


def _vector(values) -> dict:
    source = values if isinstance(values, dict) else {}
    return {
        key: round(float(source.get(key, 0.5)), 6)
        for key in TACTICAL_KEYS
    }


def _final_tactical_execution(initial, state) -> dict:
    result = {}
    for side, team in (("home", state.home), ("away", state.away)):
        row = dict(initial[side])
        initial_vector = row["initial_vector"]
        final_vector = _vector(team.coach.tactical_current)
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


def _icon_shock(team) -> float:
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
