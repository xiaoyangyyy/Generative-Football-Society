"""Micro-event impulses for player / coach / referee / crowd (continuous, no thresholds)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.match_engine.config import AffectiveConfig
from src.match_engine.state import MatchAffectiveState


class MicroEventType(str, Enum):
    GOAL_SCORED = "goal_scored"
    GOAL_CONCEDED = "goal_conceded"
    ASSIST = "assist"
    KEY_PASS = "key_pass"
    TACKLE_WON = "tackle_won"
    DISPOSSESSED = "dispossessed"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    FOUL_COMMITTED = "foul_committed"
    FOUL_SUFFERED = "foul_suffered"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SAVE = "save"
    VAR_CONTROVERSY = "var_controversy"
    ICON_PROTEST = "icon_protest"
    CROWD_WAVE = "crowd_wave"


@dataclass
class MicroEvent:
    t_sec: float
    event_type: MicroEventType
    team_id: str
    intensity: float = 1.0
    player_id: Optional[str] = None
    opponent_team_id: Optional[str] = None
    meta: Optional[dict] = None


# index in z_emo: pride, anger, fear, determination
_P, _A, _F, _D = 0, 1, 2, 3


def _impulse_vec(pride=0.0, anger=0.0, fear=0.0, determination=0.0) -> np.ndarray:
    return np.array([pride, anger, fear, determination], dtype=float)


def _bump_player(team, player_id, vector, intensity, scale=1.0):
    for player in team.players:
        if player.player_id == player_id:
            player.z_emo = player.z_emo + vector * intensity * scale
            return
    if team.icon_player_id and team.icon_player_id != player_id:
        _bump_player(
            team, team.icon_player_id, vector * 0.6, intensity, scale,
        )


def _bump_team(team, vector, intensity, scale=1.0, exclude=None):
    for player in team.players:
        if not player.on_pitch or (exclude and player.player_id == exclude):
            continue
        weight = 1.15 if player.is_icon else 1.0
        player.z_emo = player.z_emo + vector * intensity * scale * weight


def _bump_opponent(opponent, vector, intensity, scale=0.85):
    if opponent is None:
        return
    for player in opponent.players:
        if player.on_pitch:
            player.z_emo = player.z_emo + vector * intensity * scale


def _apply_goal_event(state, event, cfg, team, opponent, intensity):
    if event.event_type == MicroEventType.GOAL_SCORED:
        vector = _impulse_vec(cfg.impulse_goal_pride, 0, 0, 0.25)
        if event.player_id:
            _bump_player(team, event.player_id, vector, intensity, 1.2)
        _bump_team(team, vector, intensity, 0.55, exclude=event.player_id)
        _bump_opponent(
            opponent,
            _impulse_vec(
                0, cfg.impulse_concede_anger * 0.5,
                cfg.impulse_concede_fear, 0,
            ),
            intensity,
        )
        team.score += 1
        team.coach.z_trust += 0.12 * intensity
        team.coach.z_stress -= 0.08 * intensity
        return cfg.psi_goal_pulse * intensity
    _bump_team(
        team,
        _impulse_vec(
            0, cfg.impulse_concede_anger * 0.6,
            cfg.impulse_concede_fear, 0.15,
        ),
        intensity, 0.9,
    )
    team.coach.z_stress += 0.18 * intensity
    team.coach.z_trust -= 0.10 * intensity
    return cfg.psi_goal_pulse * 0.35 * intensity


def _apply_player_event(event, cfg, team, intensity):
    event_type = event.event_type
    if event_type == MicroEventType.ASSIST:
        vector = _impulse_vec(cfg.impulse_assist_pride, 0, 0, 0.12)
        if event.player_id:
            _bump_player(team, event.player_id, vector, intensity, 1.1)
        _bump_team(team, vector * 0.4, intensity, 0.35)
    elif event_type == MicroEventType.KEY_PASS and event.player_id:
        _bump_player(
            team, event.player_id,
            _impulse_vec(cfg.impulse_key_pass_pride, 0, 0, 0.08),
            intensity,
        )
    elif event_type == MicroEventType.TACKLE_WON and event.player_id:
        _bump_player(
            team, event.player_id,
            _impulse_vec(cfg.impulse_tackle_won_pride, 0, 0, 0.05),
            intensity,
        )
    elif event_type == MicroEventType.DISPOSSESSED and event.player_id:
        _bump_player(
            team, event.player_id,
            _impulse_vec(
                0, cfg.impulse_dispossessed_anger,
                cfg.impulse_dispossessed_fear, 0,
            ),
            intensity,
        )
    elif event_type == MicroEventType.SHOT_ON_TARGET:
        if event.player_id:
            _bump_player(
                team, event.player_id, _impulse_vec(0.1, 0, 0, 0.12),
                intensity,
            )
        return cfg.psi_shot_pulse * intensity
    elif event_type == MicroEventType.SHOT_OFF_TARGET and event.player_id:
        _bump_player(
            team, event.player_id,
            _impulse_vec(
                0, 0.08, cfg.impulse_shot_miss_fear,
                cfg.impulse_shot_miss_determination,
            ),
            intensity,
        )
    elif event_type == MicroEventType.FOUL_COMMITTED and event.player_id:
        _bump_player(
            team, event.player_id,
            _impulse_vec(0, cfg.impulse_foul_committed_anger, 0.05, 0),
            intensity,
        )
    elif event_type == MicroEventType.FOUL_SUFFERED and event.player_id:
        player = next(
            (item for item in team.players if item.player_id == event.player_id),
            None,
        )
        if player:
            player.z_emo += _impulse_vec(
                0, cfg.impulse_foul_suffered_anger, 0.08, 0,
            ) * intensity
            player.norm_violation_accum += 0.15 * intensity
    elif event_type == MicroEventType.SAVE:
        if event.player_id:
            _bump_player(
                team, event.player_id,
                _impulse_vec(cfg.impulse_save_pride, 0, -0.05, 0.1),
                intensity,
            )
        return cfg.psi_save_pulse * intensity
    return 0.0


def _apply_discipline_event(state, event, cfg, team, intensity):
    if event.event_type == MicroEventType.YELLOW_CARD:
        if event.player_id:
            _bump_player(
                team, event.player_id,
                _impulse_vec(
                    0, cfg.impulse_yellow_anger, cfg.impulse_yellow_fear, 0,
                ),
                intensity, 1.1,
            )
        state.referee.card_load += 0.25 * intensity
        state.referee.controversy_integral += 0.2 * intensity
        return cfg.psi_red_pulse * 0.45 * intensity
    if event.player_id:
        player = next(
            (item for item in team.players if item.player_id == event.player_id),
            None,
        )
        if player:
            player.on_pitch = False
            player.z_emo += _impulse_vec(
                0, cfg.impulse_red_anger, cfg.impulse_red_fear, 0,
            ) * intensity
    _bump_team(
        team, _impulse_vec(0, 0.15, 0.35, -0.1), intensity, 0.7,
    )
    state.referee.card_load += 0.55 * intensity
    state.referee.controversy_integral += 0.45 * intensity
    return cfg.psi_red_pulse * intensity


def _apply_atmosphere_event(state, event, cfg, team, opponent, intensity):
    if event.event_type == MicroEventType.VAR_CONTROVERSY:
        _bump_team(
            team,
            _impulse_vec(0, cfg.impulse_var_controversy_anger, 0.12, 0),
            intensity, 0.75,
        )
        if opponent:
            for player in opponent.players:
                if player.on_pitch:
                    player.z_emo += _impulse_vec(
                        0, cfg.impulse_var_controversy_anger * 0.5,
                        0.08, 0,
                    ) * intensity
        state.referee.controversy_integral += cfg.impulse_var_norm * intensity
        state.referee.z_calm -= 0.2 * intensity
        return cfg.psi_var_pulse * intensity
    if event.event_type == MicroEventType.ICON_PROTEST:
        team.coach.z_stress += 0.25 * intensity
        team.coach.z_rage += 0.15 * intensity
        team.coach.volatility_accum += 0.08 * intensity
        if team.icon_player_id:
            _bump_player(
                team, team.icon_player_id,
                _impulse_vec(0, 0.35, 0, 0.2), intensity,
            )
        return 0.0
    return 0.2 * intensity


def apply_micro_event(
    state: MatchAffectiveState,
    event: MicroEvent,
    cfg: AffectiveConfig,
) -> float:
    """
    Apply event to z_emo / coach / ref / crowd.
    Returns crowd pulse contribution G(t) for psi equation.
    """
    team = state.team(event.team_id)
    opponent = (
        state.team(event.opponent_team_id)
        if event.opponent_team_id else None
    )
    intensity = float(max(0.05, event.intensity))
    if event.event_type in (
        MicroEventType.GOAL_SCORED, MicroEventType.GOAL_CONCEDED,
    ):
        return _apply_goal_event(
            state, event, cfg, team, opponent, intensity,
        )
    if event.event_type in (
        MicroEventType.YELLOW_CARD, MicroEventType.RED_CARD,
    ):
        return _apply_discipline_event(state, event, cfg, team, intensity)
    if event.event_type in (
        MicroEventType.VAR_CONTROVERSY,
        MicroEventType.ICON_PROTEST,
        MicroEventType.CROWD_WAVE,
    ):
        return _apply_atmosphere_event(
            state, event, cfg, team, opponent, intensity,
        )
    return _apply_player_event(event, cfg, team, intensity)
