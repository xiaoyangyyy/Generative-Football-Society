"""Frozen, player-level lineup contracts for continuity-enabled matches."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from src.match_engine.formation import roles_for_formation
from src.match_engine.tactical_catalog import normalize_formation_key
from src.simulation.cross_match_state import (
    TeamSquadCarryover, apply_carryover_to_roster_for_agent,
)
from src.simulation.squad_registry import load_effective_roster


LINEUP_SOURCES = {"manual", "automatic"}
MAX_BENCH_PLAYERS = 12


def _player_ids(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"lineup {field} must be an array")
    ids = tuple(str(item).strip() for item in value)
    if any(not item or len(item) > 96 for item in ids):
        raise ValueError(f"lineup {field} contains an invalid player id")
    return ids


@dataclass(frozen=True)
class LineupSelection:
    starters: tuple[str, ...]
    bench: tuple[str, ...] = ()
    source: str = "manual"
    roster_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "starters", _player_ids(self.starters, "starters"))
        object.__setattr__(self, "bench", _player_ids(self.bench, "bench"))
        if len(self.starters) != 11:
            raise ValueError("lineup requires exactly 11 starters")
        if len(self.bench) > MAX_BENCH_PLAYERS:
            raise ValueError(f"lineup bench cannot exceed {MAX_BENCH_PLAYERS} players")
        combined = self.starters + self.bench
        if len(set(combined)) != len(combined):
            raise ValueError("lineup players must be unique")
        if self.source not in LINEUP_SOURCES:
            raise ValueError("unsupported lineup source")
        if self.roster_fingerprint and (
            len(self.roster_fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in self.roster_fingerprint)
        ):
            raise ValueError("invalid lineup roster fingerprint")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LineupSelection":
        if not isinstance(payload, Mapping):
            raise ValueError("lineup must be an object")
        return cls(
            starters=payload.get("starters", ()),
            bench=payload.get("bench", ()),
            source=payload.get("source", "manual"),
            roster_fingerprint=str(payload.get("roster_fingerprint", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "starters": list(self.starters),
            "bench": list(self.bench),
            "source": self.source,
            "roster_fingerprint": self.roster_fingerprint,
        }


def roster_fingerprint(roster: Mapping[str, Any]) -> str:
    players = []
    for item in roster.get("players") or []:
        players.append({
            "player_id": str(item.get("player_id", "")),
            "role": str(item.get("role", "")),
            "availability": round(float(item.get("availability", 1.0)), 8),
            "condition": {
                str(key): round(float(value), 8)
                for key, value in sorted((item.get("condition") or {}).items())
            },
            "form_ema": round(float(item.get("form_ema", 0.55)), 8),
        })
    payload = {
        "team_id": str(roster.get("team_id", "")),
        "formation": str(roster.get("formation", "")),
        "players": sorted(players, key=lambda row: row["player_id"]),
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_carryover(root: Path, team: str) -> TeamSquadCarryover:
    path = root / "data/persistence/squad_carryover.json"
    if not path.is_file():
        return TeamSquadCarryover(team_id=team)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("invalid squad carryover payload")
    raw = payload.get(team)
    if raw is None:
        return TeamSquadCarryover(team_id=team)
    if not isinstance(raw, Mapping):
        raise ValueError("invalid team squad carryover payload")
    carry = TeamSquadCarryover.from_dict(dict(raw))
    if carry.team_id and carry.team_id != team:
        raise ValueError("squad carryover team identity mismatch")
    carry.team_id = team
    return carry


def _quality(player: Mapping[str, Any]) -> float:
    abilities = player.get("abilities") or {}
    values = [
        float(abilities.get(key, 0.5))
        for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "shot", "power", "aerial", "mental",
        )
    ]
    if str(player.get("role", "")) == "GK":
        values.extend([
            float(abilities.get("gk_reflex", abilities.get("gk", 0.5))),
            float(abilities.get("gk_aerial", abilities.get("gk", 0.5))),
        ] * 2)
    condition = [float(value) for value in (player.get("condition") or {}).values()]
    technical = sum(values) / max(1, len(values))
    readiness = sum(condition) / max(1, len(condition)) if condition else 0.5
    return 0.72 * technical + 0.18 * readiness + 0.10 * float(player.get("form_ema", 0.55))


def _fallback_roles(role: str) -> tuple[str, ...]:
    return {
        "GK": (),
        "CB": ("LB", "RB", "DM"),
        "LB": ("LWB", "CB", "LM"),
        "RB": ("RWB", "CB", "RM"),
        "LWB": ("LB", "LM", "CB"),
        "RWB": ("RB", "RM", "CB"),
        "DM": ("CM", "CB"),
        "CM": ("DM", "AM", "LM", "RM"),
        "AM": ("CM", "LW", "RW", "ST"),
        "LM": ("LW", "CM", "AM"),
        "RM": ("RW", "CM", "AM"),
        "LW": ("LM", "AM", "ST"),
        "RW": ("RM", "AM", "ST"),
        "ST": ("CF", "AM", "LW", "RW"),
        "CF": ("ST", "AM"),
    }.get(role, ())


def _selection_score(player: Mapping[str, Any], rotation: str) -> float:
    quality = _quality(player) * float(player.get("availability", 1.0))
    minutes = max(0.0, float(player.get("minutes_ema", 0.0)))
    original_starter = player.get("original_squad_role") == "starter"
    if rotation == "strongest":
        return quality
    if rotation == "balanced":
        return quality - 0.0015 * minutes
    if rotation == "rotate":
        return quality - 0.0045 * minutes + (0.045 if not original_starter else 0.0)
    raise ValueError("unsupported manager rotation")


def automatic_lineup(catalog: Mapping[str, Any], rotation: str) -> LineupSelection:
    if not catalog.get("available"):
        raise ValueError("manager squad is unavailable")
    available = [
        dict(player) for player in catalog.get("players") or []
        if bool(player.get("selectable"))
    ]
    if len(available) < 11:
        raise ValueError("manager squad has fewer than 11 available players")
    formation = normalize_formation_key(str(catalog.get("formation") or "4-3-3"))
    roles = roles_for_formation(formation)
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def ranked(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            pool,
            key=lambda player: (
                -_selection_score(player, rotation), str(player["player_id"]),
            ),
        )

    for required in roles:
        candidates = [
            player for player in available
            if player["player_id"] not in used and player.get("role") == required
        ]
        if not candidates:
            fallback = set(_fallback_roles(required))
            candidates = [
                player for player in available
                if player["player_id"] not in used and player.get("role") in fallback
            ]
        if not candidates and required != "GK":
            candidates = [
                player for player in available
                if player["player_id"] not in used and player.get("role") != "GK"
            ]
        if not candidates:
            raise ValueError(f"manager squad cannot fill formation role {required}")
        selected = ranked(candidates)[0]
        chosen.append(selected)
        used.add(str(selected["player_id"]))

    remaining = ranked([
        player for player in available if player["player_id"] not in used
    ])
    return LineupSelection(
        starters=tuple(str(player["player_id"]) for player in chosen),
        bench=tuple(str(player["player_id"]) for player in remaining[:MAX_BENCH_PLAYERS]),
        source="automatic",
        roster_fingerprint=str(catalog["roster_fingerprint"]),
    )


def validate_and_freeze_lineup(
    catalog: Mapping[str, Any], selection: LineupSelection,
) -> LineupSelection:
    if not catalog.get("available"):
        raise ValueError("manager squad is unavailable")
    players = {str(row["player_id"]): row for row in catalog.get("players") or []}
    unknown = [pid for pid in selection.starters + selection.bench if pid not in players]
    if unknown:
        raise ValueError("lineup contains a player outside the manager squad")
    unavailable = [
        pid for pid in selection.starters + selection.bench
        if not bool(players[pid].get("selectable"))
    ]
    if unavailable:
        raise ValueError("lineup contains an unavailable player")
    if sum(1 for pid in selection.starters if players[pid].get("role") == "GK") != 1:
        raise ValueError("lineup requires exactly one starting goalkeeper")
    if selection.roster_fingerprint and (
        selection.roster_fingerprint != catalog["roster_fingerprint"]
    ):
        raise ValueError("lineup roster state has changed")
    return LineupSelection(
        starters=selection.starters,
        bench=selection.bench,
        source=selection.source,
        roster_fingerprint=str(catalog["roster_fingerprint"]),
    )


def build_squad_catalog(root: str | Path, team: str) -> dict[str, Any]:
    resolved = Path(root).resolve()
    carry = _load_carryover(resolved, team)
    roster = load_effective_roster(resolved, team)
    if roster:
        roster_ids = {
            str(player.get("player_id", ""))
            for player in roster.get("players") or []
            if str(player.get("player_id", ""))
        }
        carry.players = {
            player_id: player
            for player_id, player in carry.players.items()
            if player_id in roster_ids
        }
    has_persisted_condition = bool(
        carry.players or carry.settled_match_ids or carry.recovery_ids
        or abs(carry.team_fatigue_ema) > 1e-12
        or abs(carry.squad_morale_ema - 0.55) > 1e-12
        or abs(carry.team_media_pressure) > 1e-12
    )
    team_condition = {
        "fatigue": round(float(carry.team_fatigue_ema), 6),
        "morale": round(float(carry.squad_morale_ema), 6),
        "media_pressure": round(float(carry.team_media_pressure), 6),
        "injured_players": sum(
            player.injury_matches_left > 0 for player in carry.players.values()
        ),
        "suspended_players": sum(
            player.suspension_matches_left > 0 for player in carry.players.values()
        ),
        "matches_settled": len(carry.settled_match_ids),
        "source": (
            "persisted_squad_carryover"
            if has_persisted_condition else "default_simulation_baseline"
        ),
    }
    team_condition["unavailable_players"] = (
        team_condition["injured_players"] + team_condition["suspended_players"]
    )
    if not roster:
        return {
            "schema_version": 1, "available": False, "team": team,
            "reason": "roster_unavailable", "players": [], "automatic": {},
            "team_condition": team_condition,
        }
    if str(roster.get("team_id", team)) != team:
        raise ValueError("roster team identity mismatch")
    agent = SimpleNamespace(team_name=team, squad_carryover=carry)
    applied = apply_carryover_to_roster_for_agent(agent, copy.deepcopy(roster))
    fingerprint = roster_fingerprint(applied)
    players = []
    for item in applied.get("players") or []:
        pid = str(item.get("player_id", ""))
        if not pid:
            raise ValueError("roster contains an empty player id")
        pc = carry.players.get(pid)
        availability = float(item.get("availability", 1.0))
        suspended = bool(pc and pc.suspension_matches_left > 0)
        injured = bool(pc and pc.injury_matches_left > 0)
        players.append({
            "player_id": pid,
            "name": str(item.get("name", "")),
            "role": str(item.get("role", "")),
            "age": item.get("age"),
            "shirt_number": item.get("shirt_number"),
            "availability": availability,
            "selectable": availability > 0.15 and not suspended and not injured,
            "injured": injured,
            "suspended": suspended,
            "form_ema": float(item.get("form_ema", 0.55)),
            "minutes_ema": float(pc.minutes_ema if pc else 0.0),
            "quality": round(_quality(item), 6),
            "source": str(item.get("source", "")),
            "recruitment_cost": item.get("recruitment_cost"),
            "original_squad_role": str(item.get("squad_role", "bench")),
        })
    catalog: dict[str, Any] = {
        "schema_version": 1,
        "available": True,
        "team": team,
        "formation": str(roster.get("formation", "4-3-3")),
        "roster_fingerprint": fingerprint,
        "players": players,
        "automatic": {},
        "team_condition": team_condition,
    }
    if not any(str(player.get("role", "")) == "GK" for player in roster.get("players") or []):
        catalog.update({
            "available": False,
            "reason": "roster_missing_goalkeeper",
            "automatic": {},
        })
        return catalog
    for rotation in ("strongest", "balanced", "rotate"):
        try:
            catalog["automatic"][rotation] = automatic_lineup(catalog, rotation).as_dict()
        except ValueError as exc:
            catalog["automatic"][rotation] = {"available": False, "reason": str(exc)}
    if all(
        value.get("available") is False
        for value in catalog["automatic"].values()
    ):
        catalog.update({
            "available": False,
            "reason": "insufficient_available_players_or_roles",
        })
    return catalog


def apply_lineup_selection(
    roster: Mapping[str, Any], selection: LineupSelection,
) -> dict[str, Any]:
    current_fingerprint = roster_fingerprint(roster)
    if selection.roster_fingerprint != current_fingerprint:
        raise ValueError("frozen lineup roster state has changed")
    players = {str(row.get("player_id", "")): row for row in roster.get("players") or []}
    for pid in selection.starters + selection.bench:
        if pid not in players:
            raise ValueError("frozen lineup player is missing from roster")
        if float(players[pid].get("availability", 1.0)) <= 0.15:
            raise ValueError("frozen lineup player is unavailable")
    if sum(1 for pid in selection.starters if players[pid].get("role") == "GK") != 1:
        raise ValueError("frozen lineup requires exactly one starting goalkeeper")
    out = copy.deepcopy(dict(roster))
    starter_ids, bench_ids = set(selection.starters), set(selection.bench)
    for player in out.get("players") or []:
        pid = str(player.get("player_id", ""))
        if float(player.get("availability", 1.0)) <= 0.15:
            player["squad_role"] = "suspended"
        elif pid in starter_ids:
            player["squad_role"] = "starter"
        elif pid in bench_ids:
            player["squad_role"] = "bench"
        else:
            player["squad_role"] = "reserve"
    out["lineup_selection"] = selection.as_dict()
    return out


def lineup_summary(roster: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not roster or not roster.get("lineup_selection"):
        return None
    selected = LineupSelection.from_payload(roster["lineup_selection"])
    names = {
        str(player.get("player_id", "")): str(player.get("name", ""))
        for player in roster.get("players") or []
    }
    return {
        **selected.as_dict(),
        "starter_names": [names.get(pid, pid) for pid in selected.starters],
        "bench_names": [names.get(pid, pid) for pid in selected.bench],
    }
