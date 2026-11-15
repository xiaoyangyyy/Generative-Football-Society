from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    player_id: str | None = None
    opponent_team_id: str | None = None
    meta: dict | None = None
