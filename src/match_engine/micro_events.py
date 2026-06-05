"""Micro-event impulses for player / coach / referee / crowd (continuous, no thresholds)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.match_engine.config import AffectiveConfig
from src.match_engine.state import EMOTION_KEYS, MatchAffectiveState, TeamAffectiveState


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
    opp_id = event.opponent_team_id
    opp = state.team(opp_id) if opp_id else None
    inten = float(max(0.05, event.intensity))
    g_psi = 0.0

    def bump_player(player_id: str, vec: np.ndarray, scale: float = 1.0):
        for p in team.players:
            if p.player_id == player_id:
                p.z_emo = p.z_emo + vec * inten * scale
                return
        # fallback: captain/icon
        if team.icon_player_id:
            bump_player(team.icon_player_id, vec * 0.6, scale)

    def bump_team(vec: np.ndarray, scale: float = 1.0, exclude: Optional[str] = None):
        for p in team.players:
            if not p.on_pitch:
                continue
            if exclude and p.player_id == exclude:
                continue
            w = 1.0
            if p.is_icon:
                w = 1.15
            p.z_emo = p.z_emo + vec * inten * scale * w

    def bump_opp(vec: np.ndarray, scale: float = 0.85):
        if opp is None:
            return
        for p in opp.players:
            if p.on_pitch:
                p.z_emo = p.z_emo + vec * inten * scale

    et = event.event_type

    if et == MicroEventType.GOAL_SCORED:
        vec = _impulse_vec(cfg.impulse_goal_pride, 0, 0, 0.25)
        if event.player_id:
            bump_player(event.player_id, vec, 1.2)
        bump_team(vec, 0.55, exclude=event.player_id)
        bump_opp(_impulse_vec(0, cfg.impulse_concede_anger * 0.5, cfg.impulse_concede_fear, 0))
        team.score += 1
        g_psi = cfg.psi_goal_pulse * inten
        team.coach.z_trust += 0.12 * inten
        team.coach.z_stress -= 0.08 * inten

    elif et == MicroEventType.GOAL_CONCEDED:
        bump_team(
            _impulse_vec(0, cfg.impulse_concede_anger * 0.6, cfg.impulse_concede_fear, 0.15),
            0.9,
        )
        team.coach.z_stress += 0.18 * inten
        team.coach.z_trust -= 0.10 * inten
        g_psi = cfg.psi_goal_pulse * 0.35 * inten

    elif et == MicroEventType.ASSIST:
        vec = _impulse_vec(cfg.impulse_assist_pride, 0, 0, 0.12)
        if event.player_id:
            bump_player(event.player_id, vec, 1.1)
        bump_team(vec * 0.4, 0.35)

    elif et == MicroEventType.KEY_PASS:
        vec = _impulse_vec(cfg.impulse_key_pass_pride, 0, 0, 0.08)
        if event.player_id:
            bump_player(event.player_id, vec)

    elif et == MicroEventType.TACKLE_WON:
        if event.player_id:
            bump_player(
                event.player_id,
                _impulse_vec(cfg.impulse_tackle_won_pride, 0, 0, 0.05),
            )

    elif et == MicroEventType.DISPOSSESSED:
        if event.player_id:
            bump_player(
                event.player_id,
                _impulse_vec(0, cfg.impulse_dispossessed_anger, cfg.impulse_dispossessed_fear, 0),
            )

    elif et == MicroEventType.SHOT_ON_TARGET:
        g_psi = cfg.psi_shot_pulse * inten
        if event.player_id:
            bump_player(event.player_id, _impulse_vec(0.1, 0, 0, 0.12))

    elif et == MicroEventType.SHOT_OFF_TARGET:
        if event.player_id:
            bump_player(
                event.player_id,
                _impulse_vec(0, 0.08, cfg.impulse_shot_miss_fear, cfg.impulse_shot_miss_determination),
            )

    elif et == MicroEventType.FOUL_COMMITTED:
        if event.player_id:
            bump_player(
                event.player_id,
                _impulse_vec(0, cfg.impulse_foul_committed_anger, 0.05, 0),
            )

    elif et == MicroEventType.FOUL_SUFFERED:
        if event.player_id:
            p = next((x for x in team.players if x.player_id == event.player_id), None)
            if p:
                p.z_emo += _impulse_vec(0, cfg.impulse_foul_suffered_anger, 0.08, 0) * inten
                p.norm_violation_accum += 0.15 * inten

    elif et == MicroEventType.YELLOW_CARD:
        if event.player_id:
            bump_player(
                event.player_id,
                _impulse_vec(0, cfg.impulse_yellow_anger, cfg.impulse_yellow_fear, 0),
                1.1,
            )
        state.referee.card_load += 0.25 * inten
        g_psi = cfg.psi_red_pulse * 0.45 * inten
        state.referee.controversy_integral += 0.2 * inten

    elif et == MicroEventType.RED_CARD:
        if event.player_id:
            for p in team.players:
                if p.player_id == event.player_id:
                    p.on_pitch = False
                    p.z_emo += _impulse_vec(0, cfg.impulse_red_anger, cfg.impulse_red_fear, 0) * inten
                    break
        bump_team(_impulse_vec(0, 0.15, 0.35, -0.1), 0.7)
        state.referee.card_load += 0.55 * inten
        g_psi = cfg.psi_red_pulse * inten
        state.referee.controversy_integral += 0.45 * inten

    elif et == MicroEventType.SAVE:
        if event.player_id:
            bump_player(
                event.player_id,
                _impulse_vec(cfg.impulse_save_pride, 0, -0.05, 0.1),
                1.0,
            )
        g_psi = cfg.psi_save_pulse * inten

    elif et == MicroEventType.VAR_CONTROVERSY:
        bump_team(
            _impulse_vec(0, cfg.impulse_var_controversy_anger, 0.12, 0),
            0.75,
        )
        if opp:
            for p in opp.players:
                if p.on_pitch:
                    p.z_emo += _impulse_vec(0, cfg.impulse_var_controversy_anger * 0.5, 0.08, 0) * inten
        state.referee.controversy_integral += cfg.impulse_var_norm * inten
        g_psi = cfg.psi_var_pulse * inten
        state.referee.z_calm -= 0.2 * inten

    elif et == MicroEventType.ICON_PROTEST:
        team.coach.z_stress += 0.25 * inten
        team.coach.z_rage += 0.15 * inten
        team.coach.volatility_accum += 0.08 * inten
        if team.icon_player_id:
            bump_player(
                team.icon_player_id,
                _impulse_vec(0, 0.35, 0, 0.2),
            )

    elif et == MicroEventType.CROWD_WAVE:
        g_psi = 0.2 * inten

    return g_psi
