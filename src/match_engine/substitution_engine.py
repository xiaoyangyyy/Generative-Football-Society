"""In-match substitutions from full squad (starters + bench)."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.formation import interpolate_anchors
from src.match_engine.math_utils import sigmoid
from src.match_engine.player_match_stats import PlayerMatchStatsTracker

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState, TeamAffectiveState

# Match minutes (approx) when coach may use subs
DEFAULT_SUB_WINDOWS = (58.0, 70.0, 82.0)


def _stamina_score(p: "PlayerAffectiveState") -> float:
    return float(sigmoid(float(p.stamina_logit)))


def _pick_sub_off(on_pitch: List["PlayerAffectiveState"], rng: np.random.Generator) -> Optional["PlayerAffectiveState"]:
    field = [p for p in on_pitch if p.role != "GK"]
    if not field:
        return None
    weights = np.array([1.0 - _stamina_score(p) + 0.05 * p.cognitive_load for p in field], dtype=float)
    weights = np.maximum(weights, 0.01)
    weights /= weights.sum()
    return field[int(rng.choice(len(field), p=weights))]


def _pick_sub_on(
    bench: List["PlayerAffectiveState"],
    off_role: str,
    rng: np.random.Generator,
) -> Optional["PlayerAffectiveState"]:
    if not bench:
        return None
    same = [p for p in bench if p.role == off_role]
    pool = same if same else bench
    scores = np.array([_stamina_score(p) + 0.1 * float(p.abilities.pace) for p in pool], dtype=float)
    scores /= scores.sum()
    return pool[int(rng.choice(len(pool), p=scores))]


def maybe_apply_substitutions(
    state: "MatchAffectiveState",
    clock_sec: float,
    rng: np.random.Generator,
    tracker: PlayerMatchStatsTracker,
    *,
    sub_windows: tuple = DEFAULT_SUB_WINDOWS,
    max_subs_per_team: int = 5,
    subs_done: Optional[dict] = None,
) -> List[str]:
    """
    At configured minutes, swap tired on-pitch player for bench.
    Returns timeline snippets.
    """
    subs_done = subs_done if subs_done is not None else {}
    minute = clock_sec / 60.0
    notes: List[str] = []

    for team in (state.home, state.away):
        tid = team.team_id
        count = subs_done.get(tid, 0)
        if count >= max_subs_per_team:
            continue
        window_idx = count
        if window_idx >= len(sub_windows):
            continue
        if minute < sub_windows[window_idx] - 0.5 or minute > sub_windows[window_idx] + 2.5:
            continue

        on_pitch = [p for p in team.players if p.on_pitch]
        bench = [
            p
            for p in team.players
            if not p.on_pitch and float(getattr(p, "availability", 1.0)) > 0.2
        ]
        if len(bench) < 1 or len(on_pitch) < 11:
            continue

        off_p = _pick_sub_off(on_pitch, rng)
        if off_p is None:
            continue
        on_p = _pick_sub_on(bench, off_p.role, rng)
        if on_p is None:
            continue

        _swap_players(team, off_p, on_p, attacks_high_x=team.attacks_high_x, formation_key=team.formation_key)
        tracker.record_substitution(tid, off_p.player_id, on_p.player_id, minute)
        subs_done[tid] = count + 1
        notes.append(f"{int(minute)}' SUB {on_p.name} on for {off_p.name} ({tid})")

    return notes


def _swap_players(
    team: "TeamAffectiveState",
    off_p: "PlayerAffectiveState",
    on_p: "PlayerAffectiveState",
    *,
    attacks_high_x: bool,
    formation_key: str,
) -> None:
    pos = off_p.position.copy()
    off_p.on_pitch = False
    on_p.on_pitch = True
    on_p.position = pos
    on_p.role = off_p.role
    off_p.position = interpolate_anchors(
        on_p.role if on_p.role in ("GK",) else "CM",
        0.02,
        line_height=0.5,
        width=0.5,
        attacks_high_x=False,
        formation_key=formation_key,
    )
