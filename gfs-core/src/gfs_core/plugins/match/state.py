from __future__ import annotations

import numpy as np

from gfs_core.contracts.micro_state import (
    AssistantRefereeState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
)
from gfs_core.contracts.signals import normalize_internal_signals


def possession_prior(home: float, away: float, config) -> float:
    share = home / max(24.0, home + away)
    return float(np.clip(
        0.5 + config.possession_strength_beta * (share - 0.5), 0.36, 0.64,
    ))


def build_match_state(
    home_profile,
    away_profile,
    agents,
    spatial,
    *,
    referee,
    stage_pressure,
    neutral_venue,
    effective_home,
    effective_away,
    possession_home,
    internal_home,
    internal_away,
    rng,
):
    home = agents.build(home_profile, rng, match_eff_status=effective_home)
    away = agents.build(away_profile, rng, match_eff_status=effective_away)
    home.attacks_high_x = True
    away.attacks_high_x = False
    for player in away.players:
        player.position = spatial.mirror_for_away(player.position)
    home.coach.team_id = home.team_id
    away.coach.team_id = away.team_id
    home.formation_key = spatial.normalize_formation_key(home_profile.formation)
    away.formation_key = spatial.normalize_formation_key(away_profile.formation)

    referee = referee or {}
    referee_state = RefereeAffectiveState(
        profile_name=str(referee.get("profile_name", "balanced")),
        strictness_base=float(referee.get("strictness", 0.55)),
        bias_home=float(referee.get("bias_t1", 0.0)),
        z_calm=float(np.arctanh(0.55)),
        controversy_integral=float(
            home_profile.referee_grievance + away_profile.referee_grievance
        ) * 0.5,
        assistants=[
            AssistantRefereeState(
                side="left", offside_strictness=0.52, trust_with_center=0.62,
            ),
            AssistantRefereeState(
                side="right", offside_strictness=0.48, trust_with_center=0.58,
            ),
        ],
    )
    crowd = CrowdState(
        psi=0.0 if neutral_venue else 0.14,
        home_team_id=home.team_id,
        is_neutral_venue=neutral_venue,
    )
    if not neutral_venue:
        for player in home.players:
            player.fan_affinity = float(np.clip(player.fan_affinity + 0.20, 0.0, 1.0))
            player.morale_logit = float(player.morale_logit + 0.15)
    state = MatchAffectiveState(
        home,
        away,
        referee_state,
        crowd,
        stage_pressure=float(np.clip(stage_pressure, 0.0, 1.0)),
    )
    for player in away.players:
        player.position = spatial.mirror_for_away(player.position)
    state.home.possession_share = possession_home
    state.away.possession_share = 1.0 - possession_home
    midfielder = next(
        (player for player in home.players if player.role == "CM"),
        home.players[0],
    )
    state.ball.position = midfielder.position.copy()
    state.ball.possessor_id = midfielder.player_id
    state.ball.possession_team_id = home.team_id
    state.ball.velocity = np.zeros(2)
    state.ball.height = 0.0
    state.ball.omega = 0.0
    return state, normalize_internal_signals(internal_home, internal_away)
