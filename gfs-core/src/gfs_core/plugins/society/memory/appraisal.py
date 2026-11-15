from __future__ import annotations

import hashlib

from gfs_core.contracts.society import MemoryEvent


def player_appraisals(summary, team: str, match_id: str, match_index: int) -> list[MemoryEvent]:
    is_home = summary.home_team == team
    emotion = summary.home_emotion_mean if is_home else summary.away_emotion_mean
    events = []
    for player_id, row in summary.player_stats.get(team, {}).items():
        score = _bounded(
            0.5
            + 0.15 * float(row.get("goals", 0))
            + 0.08 * float(row.get("assists", 0))
            + 0.04 * float(row.get("tackles", 0))
            + 0.04 * float(row.get("pass_cmp", 0.0))
            - 0.04 * float(row.get("yellow_cards", 0))
            - 0.20 * float(row.get("red_cards", 0)),
            0.05,
            0.95,
        )
        valence = 2 * score - 1
        context = {
            "player_id": player_id,
            "minutes": float(row.get("minutes", 0.0)),
            "performance_score": score,
            "statistics": dict(row),
        }
        events.append(MemoryEvent(
            event_id=_event_id(match_id, "player_performance", player_id),
            match_id=match_id,
            team=team,
            subject=f"player:{player_id}",
            kind="player_performance",
            summary=f"player {player_id}: performance_score={score:.3f}",
            match_index=match_index,
            importance=0.55 + 0.2 * abs(valence),
            valence=valence,
            arousal=0.4 + 0.4 * abs(valence),
            emotion=emotion,
            source=f"micro:{match_id}",
            tags=("player",),
            context=context,
        ))
        events.append(MemoryEvent(
            event_id=_event_id(match_id, "coach_player_relation", player_id),
            match_id=match_id,
            team=team,
            subject=f"player:{player_id}:coach:{team}",
            kind="coach_player_relation",
            summary=f"coach-player trust appraisal for {player_id}",
            match_index=match_index,
            importance=0.45,
            valence=valence,
            arousal=0.3,
            emotion=emotion,
            source=f"micro:{match_id}",
            tags=("relationship", "coach_player"),
            context={
                **context,
                "actor_id": player_id,
                "target_id": f"coach:{team}",
                "relation": "coach_player",
                "trust_delta": 0.04 * (score - 0.5),
            },
        ))
    return events


def _event_id(match_id: str, kind: str, player_id: str) -> str:
    return hashlib.sha256(f"{match_id}\x1f{kind}\x1f{player_id}".encode()).hexdigest()


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
