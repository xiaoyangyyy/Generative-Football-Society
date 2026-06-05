"""Per-player micro match statistics → cross-match carryover."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.math_utils import sigmoid
from src.match_engine.micro_events import MicroEvent, MicroEventType

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState


@dataclass
class PlayerStatLine:
    minutes: float = 0.0
    pass_att: int = 0
    pass_cmp: int = 0
    key_passes: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    yellow_cards: int = 0
    red_cards: int = 0
    fouls_committed: int = 0
    tackles: int = 0
    subbed_on: bool = False
    subbed_off: bool = False
    injury_exit: bool = False

    def to_carryover_dict(self) -> Dict[str, float]:
        att = max(1, self.pass_att)
        return {
            "minutes": self.minutes,
            "pass_att": float(self.pass_att),
            "pass_cmp": float(self.pass_cmp / att),
            "key_passes": float(self.key_passes),
            "shots": float(self.shots),
            "goals": float(self.goals),
            "assists": float(self.assists),
            "xg": float(self.xg),
            "xg_share": float(self.xg / max(0.15, self.minutes / 90.0 * 0.35)),
            "yellow_cards": float(self.yellow_cards),
            "red_cards": float(self.red_cards),
        }


class PlayerMatchStatsTracker:
    def __init__(self) -> None:
        self._lines: Dict[str, PlayerStatLine] = {}
        self.substitutions: List[Dict[str, Any]] = []

    def ensure(self, player_id: str) -> PlayerStatLine:
        if player_id not in self._lines:
            self._lines[player_id] = PlayerStatLine()
        return self._lines[player_id]

    def record_pass(
        self,
        carrier_id: str,
        recv_id: Optional[str],
        kind: str,
        completed: bool,
    ) -> None:
        ln = self.ensure(carrier_id)
        ln.pass_att += 1
        if completed:
            ln.pass_cmp += 1
        if completed and kind in ("through", "long") and recv_id:
            ln.key_passes += 1

    def record_shot(
        self,
        player_id: str,
        *,
        xg: float,
        goal: bool,
        on_target: bool,
    ) -> None:
        ln = self.ensure(player_id)
        ln.shots += 1
        ln.xg += float(xg)
        if on_target:
            ln.shots_on_target += 1
        if goal:
            ln.goals += 1

    def record_assist(self, player_id: str) -> None:
        self.ensure(player_id).assists += 1

    def record_card(self, player_id: str, *, yellow: bool = False, red: bool = False) -> None:
        ln = self.ensure(player_id)
        if yellow:
            ln.yellow_cards += 1
        if red:
            ln.red_cards += 1

    def record_foul(self, player_id: str) -> None:
        self.ensure(player_id).fouls_committed += 1

    def record_tackle(self, player_id: str) -> None:
        self.ensure(player_id).tackles += 1

    def record_substitution(
        self,
        team_id: str,
        off_id: str,
        on_id: str,
        minute: float,
    ) -> None:
        self.ensure(on_id).subbed_on = True
        self.ensure(off_id).subbed_off = True
        self.substitutions.append(
            {"team_id": team_id, "off": off_id, "on": on_id, "minute": round(minute, 1)}
        )

    def apply_event(self, event: MicroEvent) -> None:
        pid = event.player_id
        if not pid:
            return
        et = event.event_type
        if et == MicroEventType.YELLOW_CARD:
            self.record_card(pid, yellow=True)
        elif et == MicroEventType.RED_CARD:
            self.record_card(pid, red=True)
        elif et == MicroEventType.GOAL_SCORED:
            self.ensure(pid).goals += 1
        elif et == MicroEventType.ASSIST:
            self.record_assist(pid)
        elif et == MicroEventType.FOUL_COMMITTED:
            self.record_foul(pid)
        elif et == MicroEventType.TACKLE_WON:
            self.record_tackle(pid)

    def tick_minutes(self, state: "MatchAffectiveState", dt_sec: float) -> None:
        dt_min = dt_sec / 60.0
        for team in (state.home, state.away):
            for p in team.players:
                if p.on_pitch:
                    self.ensure(p.player_id).minutes += dt_min

    def export_by_team(self, state: "MatchAffectiveState") -> Dict[str, Dict[str, Dict[str, float]]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for team in (state.home, state.away):
            team_map: Dict[str, Dict[str, float]] = {}
            for p in team.players:
                ln = self._lines.get(p.player_id)
                if ln is None:
                    continue
                if ln.minutes < 0.5 and ln.pass_att == 0 and ln.shots == 0:
                    continue
                team_map[p.player_id] = ln.to_carryover_dict()
            out[team.team_id] = team_map
        return out

    def team_discipline_totals(self, state: "MatchAffectiveState") -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for team in (state.home, state.away):
            fouls = yellow = red = 0
            for p in team.players:
                ln = self._lines.get(p.player_id)
                if ln is None:
                    continue
                fouls += int(ln.fouls_committed)
                yellow += int(ln.yellow_cards)
                red += int(ln.red_cards)
            tackles = 0
            for p in team.players:
                ln = self._lines.get(p.player_id)
                if ln is not None:
                    tackles += int(ln.tackles)
            out[team.team_id] = {
                "fouls_committed": fouls,
                "yellow_cards": yellow,
                "red_cards": red,
                "tackles": tackles,
            }
        return out
