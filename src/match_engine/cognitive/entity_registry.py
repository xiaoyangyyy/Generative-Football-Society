"""Resolve entity IDs and FACTS_LEDGER for cognitive LLM calls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.match_engine.cognitive.events import (
    ENTITY_TIER_ASSISTANT,
    ENTITY_TIER_COACH,
    ENTITY_TIER_CROWD,
    ENTITY_TIER_PLAYER,
    ENTITY_TIER_REFEREE,
    CognitiveTriggerEvent,
)
from src.match_engine.micro_events import MicroEvent, MicroEventType

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState


def build_facts_ledger(state: "MatchAffectiveState", t_sec: float) -> Dict[str, Any]:
    return {
        "clock_min": round(t_sec / 60.0, 1),
        "score": f"{state.home.score}-{state.away.score}",
        "home_team": state.home.team_id,
        "away_team": state.away.team_id,
        "micro_xg": f"{state.micro_xg_home:.2f}-{state.micro_xg_away:.2f}",
        "psi": round(float(state.crowd.psi), 3),
        "ref_strictness": round(float(state.referee.strictness_base), 3),
        "ref_controversy": round(float(state.referee.controversy_integral), 3),
        "home_possession": round(float(state.home.possession_share), 3),
    }


def coach_entity_id(team_id: str) -> str:
    return f"coach:{team_id}"


def referee_entity_id() -> str:
    return "referee:center"


def assistant_entity_id(side: str) -> str:
    return f"assistant:{side}"


def crowd_entity_id() -> str:
    return "crowd:stadium"


def triggers_for_micro_event(
    ev: MicroEvent,
    state: "MatchAffectiveState",
    salience: float,
    facts: Dict[str, Any],
) -> List[CognitiveTriggerEvent]:
    out: List[CognitiveTriggerEvent] = []
    tid = ev.team_id
    et = ev.event_type

    if et == MicroEventType.GOAL_SCORED:
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "goal",
                ENTITY_TIER_COACH,
                coach_entity_id(tid),
                team_id=tid,
                salience=salience,
                facts={**facts, "event": "goal_scored", "team": tid, "player_id": ev.player_id},
            )
        )
        if ev.player_id:
            out.append(
                CognitiveTriggerEvent(
                    ev.t_sec,
                    "goal",
                    ENTITY_TIER_PLAYER,
                    ev.player_id,
                    team_id=tid,
                    salience=salience * 0.95,
                    facts={**facts, "event": "goal_scored", "player_id": ev.player_id},
                )
            )
        opp = state.away.team_id if tid == state.home.team_id else state.home.team_id
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "goal_conceded",
                ENTITY_TIER_COACH,
                coach_entity_id(opp),
                team_id=opp,
                salience=salience * 0.85,
                facts={**facts, "event": "goal_conceded", "team": opp},
            )
        )
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "goal",
                ENTITY_TIER_CROWD,
                crowd_entity_id(),
                salience=salience * 0.7,
                facts={**facts, "event": "crowd_goal"},
            )
        )

    elif et == MicroEventType.RED_CARD:
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "red_card",
                ENTITY_TIER_REFEREE,
                referee_entity_id(),
                salience=salience,
                facts={**facts, "event": "red_card", "team": tid, "player_id": ev.player_id},
            )
        )
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "red_card",
                ENTITY_TIER_COACH,
                coach_entity_id(tid),
                team_id=tid,
                salience=salience * 0.9,
                facts={**facts, "event": "red_card", "team": tid},
            )
        )
        if ev.player_id:
            out.append(
                CognitiveTriggerEvent(
                    ev.t_sec,
                    "red_card",
                    ENTITY_TIER_PLAYER,
                    ev.player_id,
                    team_id=tid,
                    salience=salience,
                    facts={**facts, "player_id": ev.player_id},
                )
            )

    elif et == MicroEventType.VAR_CONTROVERSY:
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "var",
                ENTITY_TIER_REFEREE,
                referee_entity_id(),
                salience=salience,
                facts={**facts, "event": "var"},
            )
        )
        for side in ("left", "right"):
            out.append(
                CognitiveTriggerEvent(
                    ev.t_sec,
                    "var",
                    ENTITY_TIER_ASSISTANT,
                    assistant_entity_id(side),
                    salience=salience * 0.75,
                    facts={**facts, "event": "var", "assistant": side},
                )
            )

    elif et == MicroEventType.YELLOW_CARD:
        if ev.player_id:
            out.append(
                CognitiveTriggerEvent(
                    ev.t_sec,
                    "yellow_card",
                    ENTITY_TIER_PLAYER,
                    ev.player_id,
                    team_id=tid,
                    salience=salience * 0.85,
                    facts={**facts, "player_id": ev.player_id},
                )
            )

    elif et == MicroEventType.ICON_PROTEST:
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "icon_protest",
                ENTITY_TIER_COACH,
                coach_entity_id(tid),
                team_id=tid,
                salience=salience * 0.9,
                facts={**facts, "event": "icon_protest"},
            )
        )

    elif et == MicroEventType.CROWD_WAVE:
        out.append(
            CognitiveTriggerEvent(
                ev.t_sec,
                "crowd_surge",
                ENTITY_TIER_CROWD,
                crowd_entity_id(),
                salience=salience,
                facts={**facts, "event": "crowd_wave"},
            )
        )

    return out


def half_time_triggers(
    t_sec: float,
    state: "MatchAffectiveState",
    salience: float,
    facts: Dict[str, Any],
) -> List[CognitiveTriggerEvent]:
    out = []
    for team in (state.home, state.away):
        out.append(
            CognitiveTriggerEvent(
                t_sec,
                "half_time",
                ENTITY_TIER_COACH,
                coach_entity_id(team.team_id),
                team_id=team.team_id,
                salience=salience,
                facts={**facts, "event": "half_time"},
            )
        )
    return out


def xg_swing_triggers(
    t_sec: float,
    state: "MatchAffectiveState",
    xg_swing: float,
    facts: Dict[str, Any],
) -> List[CognitiveTriggerEvent]:
    if abs(xg_swing) < 0.85:
        return []
    sal = float(min(2.0, abs(xg_swing)))
    trailing = state.home if xg_swing < 0 else state.away
    return [
        CognitiveTriggerEvent(
            t_sec,
            "xg_swing",
            ENTITY_TIER_COACH,
            coach_entity_id(trailing.team_id),
            team_id=trailing.team_id,
            salience=sal,
            facts={**facts, "event": "xg_swing", "xg_swing": round(xg_swing, 2)},
        )
    ]


def players_tier_b_for_sub(
    state: "MatchAffectiveState",
    off_id: str,
    on_id: str,
    team_id: str,
    t_sec: float,
    facts: Dict[str, Any],
) -> List[CognitiveTriggerEvent]:
    sal = 1.1
    return [
        CognitiveTriggerEvent(
            t_sec,
            "substitution",
            ENTITY_TIER_PLAYER,
            off_id,
            team_id=team_id,
            salience=sal,
            facts={**facts, "event": "sub_off", "player_id": off_id},
        ),
        CognitiveTriggerEvent(
            t_sec,
            "substitution",
            ENTITY_TIER_PLAYER,
            on_id,
            team_id=team_id,
            salience=sal * 0.85,
            facts={**facts, "event": "sub_on", "player_id": on_id},
        ),
    ]
