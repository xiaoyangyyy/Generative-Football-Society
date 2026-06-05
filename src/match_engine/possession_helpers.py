"""Ball turnover helpers after shots (open play)."""

from __future__ import annotations

import numpy as np

from src.match_engine.state import MatchAffectiveState


def turnover_to_defence_after_shot(
    state: MatchAffectiveState,
    attacking_home: bool,
    gk_player,
    rng: np.random.Generator,
) -> None:
    """Give defending GK the ball after a miss/save (prevents per-tick shot spam)."""
    state.ball.possessor_id = gk_player.player_id
    state.ball.possession_team_id = gk_player.team_id
    gx = 0.06 if attacking_home else 0.94
    state.ball.position = np.array(
        [gx, float(np.clip(0.5 + (rng.random() - 0.5) * 0.35, 0.12, 0.88))],
        dtype=float,
    )
    state.ball.velocity = np.zeros(2)
    state.ball.height = 0.0
    state.ball.omega = 0.0


def shot_cooldown_ok(state: MatchAffectiveState, attacking_home: bool, cfg, now: float) -> bool:
    cd = float(getattr(cfg, "action_shot_cooldown_sec", 14.0))
    if attacking_home:
        return (now - state.last_shot_clock_home) >= cd
    return (now - state.last_shot_clock_away) >= cd


def mark_shot_taken(state: MatchAffectiveState, attacking_home: bool, now: float) -> None:
    if attacking_home:
        state.last_shot_clock_home = now
    else:
        state.last_shot_clock_away = now
