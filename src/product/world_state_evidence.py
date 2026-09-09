"""Identity-bound three-phase carryover evidence for season fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.infrastructure import FileLease
from src.simulation.cross_match_state import PlayerCarryover, TeamSquadCarryover
from src.simulation.society_continuity import (
    society_public_snapshot,
    society_public_transition,
    validate_society_public_snapshot,
)
from src.simulation.squad_registry import load_effective_roster


LEGACY_WORLD_STATE_SCHEMA_VERSION = 1
SOCIETY_WORLD_STATE_SCHEMA_VERSION = 2
META_COUNTS_WORLD_STATE_SCHEMA_VERSION = 3
META_GOVERNANCE_WORLD_STATE_SCHEMA_VERSION = 4
WORLD_STATE_SCHEMA_VERSION = 5
_METRICS = (
    "team_fatigue_ema", "squad_morale_ema", "team_media_pressure",
    "injured_players", "suspended_players", "unavailable_players",
)
_PLAYER_FIELDS = (
    "matches_played", "minutes_ema", "goals", "assists", "yellow_cards",
    "red_cards", "suspension_matches_left", "injury_matches_left",
    "injury_severity", "medical_recovery_credit", "form_ema",
    "media_sentiment", "last_rating",
)
_BOUNDARY = (
    "simulated carryover state before the match, immediately after settlement and "
    "after matchday recovery; differences are direct persisted-state facts, not "
    "causal estimates of why the score occurred or real-world medical judgments"
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _number(value: Any) -> float:
    return round(float(value), 6)


def _baseline(root: Path, team: str) -> tuple[TeamSquadCarryover, str]:
    roster = load_effective_roster(root, team)
    if not isinstance(roster, Mapping):
        return TeamSquadCarryover(team_id=team), "deterministic_team_baseline"
    players = {}
    for row in roster.get("players") or []:
        player_id = str(row.get("player_id") or "")
        if player_id:
            players[player_id] = PlayerCarryover(
                player_id=player_id, name=str(row.get("name") or ""),
            )
    return (
        TeamSquadCarryover(team_id=team, players=players),
        "deterministic_roster_baseline",
    )


def _snapshot(team: str, carry: TeamSquadCarryover, *, source: str) -> dict[str, Any]:
    players = []
    for player_id in sorted(carry.players):
        row = carry.players[player_id]
        players.append({
            "player_id": player_id, "name": str(row.name),
            "matches_played": int(row.matches_played),
            "minutes_ema": _number(row.minutes_ema),
            "goals": int(row.goals), "assists": int(row.assists),
            "yellow_cards": int(row.yellow_cards), "red_cards": int(row.red_cards),
            "suspension_matches_left": int(row.suspension_matches_left),
            "injury_matches_left": int(row.injury_matches_left),
            "injury_severity": _number(row.injury_severity),
            "medical_recovery_credit": _number(row.medical_recovery_credit),
            "form_ema": _number(row.form_ema),
            "media_sentiment": _number(row.media_sentiment),
            "last_rating": _number(row.last_rating),
            "condition_delta": {
                key: _number(value)
                for key, value in sorted(row.condition_delta.items())
            },
        })
    injured = sum(row["injury_matches_left"] > 0 for row in players)
    suspended = sum(row["suspension_matches_left"] > 0 for row in players)
    unavailable = sum(
        row["injury_matches_left"] > 0
        or row["suspension_matches_left"] > 0
        for row in players
    )
    payload = {
        "schema_version": WORLD_STATE_SCHEMA_VERSION,
        "team_id": team, "source": source,
        "metrics": {
            "team_fatigue_ema": _number(carry.team_fatigue_ema),
            "squad_morale_ema": _number(carry.squad_morale_ema),
            "team_media_pressure": _number(carry.team_media_pressure),
            "injured_players": injured, "suspended_players": suspended,
            "unavailable_players": unavailable,
        },
        "team_dynamics_delta": {
            key: _number(value)
            for key, value in sorted(carry.team_dynamics_delta.items())
        },
        "last_match_stage": str(carry.last_match_stage),
        "players": players,
        "society": society_public_snapshot(
            carry.society_state, team_id=team,
        ),
    }
    payload["source_identity"] = _identity(payload)
    return payload


def capture_world_state(
    root: str | Path, teams: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Capture one consistent carryover-file view for the requested teams."""
    resolved = Path(root).resolve()
    ordered = [str(team) for team in teams]
    if not ordered or len(ordered) != len(set(ordered)) or any(not team for team in ordered):
        raise ValueError("world-state capture requires unique team identities")
    path = resolved / "data/persistence/squad_carryover.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLease(path.with_suffix(".lock"), timeout=5.0):
        raw: Mapping[str, Any] = {}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError("invalid squad carryover payload for world-state capture")
            raw = loaded
        result = {}
        for team in ordered:
            payload = raw.get(team)
            if payload is None:
                carry, source = _baseline(resolved, team)
            elif isinstance(payload, Mapping):
                carry = TeamSquadCarryover.from_dict(dict(payload))
                if carry.team_id != team:
                    raise ValueError("carryover team identity mismatch")
                source = "persisted_carryover"
            else:
                raise ValueError("invalid team carryover payload")
            snapshot = _snapshot(team, carry, source=source)
            validate_team_state_snapshot(snapshot, team=team)
            result[team] = snapshot
        return result


def validate_team_state_snapshot(snapshot: Mapping[str, Any], *, team: str) -> None:
    if not isinstance(snapshot, Mapping):
        raise ValueError("team world-state snapshot is invalid")
    frozen = copy.deepcopy(dict(snapshot))
    identity = frozen.pop("source_identity", None)
    players = snapshot.get("players")
    metrics = snapshot.get("metrics")
    schema_version = snapshot.get("schema_version")
    expected_fields = {
        "schema_version", "team_id", "source", "metrics",
        "team_dynamics_delta", "last_match_stage", "players", "source_identity",
    }
    if schema_version in {
        SOCIETY_WORLD_STATE_SCHEMA_VERSION,
        META_COUNTS_WORLD_STATE_SCHEMA_VERSION,
        META_GOVERNANCE_WORLD_STATE_SCHEMA_VERSION,
        WORLD_STATE_SCHEMA_VERSION,
    }:
        expected_fields.add("society")
    if (
        set(snapshot) != expected_fields
        or schema_version not in {
            LEGACY_WORLD_STATE_SCHEMA_VERSION,
            SOCIETY_WORLD_STATE_SCHEMA_VERSION,
            META_COUNTS_WORLD_STATE_SCHEMA_VERSION,
            META_GOVERNANCE_WORLD_STATE_SCHEMA_VERSION,
            WORLD_STATE_SCHEMA_VERSION,
        }
        or snapshot.get("team_id") != team
        or snapshot.get("source") not in {
            "deterministic_team_baseline", "deterministic_roster_baseline",
            "persisted_carryover",
        }
        or not isinstance(metrics, Mapping) or set(metrics) != set(_METRICS)
        or not isinstance(players, list) or len(players) > 128
        or not isinstance(snapshot.get("last_match_stage"), str)
        or len(snapshot.get("last_match_stage", "")) > 160
        or not isinstance(snapshot.get("team_dynamics_delta"), Mapping)
        or len(snapshot.get("team_dynamics_delta") or {}) > 16
        or identity != _identity(frozen)
    ):
        raise ValueError("team world-state snapshot identity mismatch")
    if schema_version in {
        SOCIETY_WORLD_STATE_SCHEMA_VERSION,
        META_COUNTS_WORLD_STATE_SCHEMA_VERSION,
        META_GOVERNANCE_WORLD_STATE_SCHEMA_VERSION,
        WORLD_STATE_SCHEMA_VERSION,
    }:
        validate_society_public_snapshot(snapshot["society"])
        meta = snapshot["society"].get("meta_learning")
        meta_projection = (
            "governance" if isinstance(meta, Mapping) and "records" in meta
            else "counts" if isinstance(meta, Mapping) else "none"
        )
        expected_projection = {
            SOCIETY_WORLD_STATE_SCHEMA_VERSION: "none",
            META_COUNTS_WORLD_STATE_SCHEMA_VERSION: "counts",
            META_GOVERNANCE_WORLD_STATE_SCHEMA_VERSION: "governance",
            WORLD_STATE_SCHEMA_VERSION: "governance",
        }[schema_version]
        if meta_projection != expected_projection:
            raise ValueError(
                "team world-state society projection version mismatch"
            )
    for key, value in metrics.items():
        if key in {"injured_players", "suspended_players", "unavailable_players"}:
            valid = not isinstance(value, bool) and isinstance(value, int) and value >= 0
        else:
            valid = (
                not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
        if not valid:
            raise ValueError("team world-state metric is invalid")
    if any(
        not isinstance(key, str) or not key or len(key) > 64
        or isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for key, value in snapshot["team_dynamics_delta"].items()
    ):
        raise ValueError("team world-state dynamics are invalid")
    ids = []
    for row in players:
        if not isinstance(row, Mapping) or set(row) != {
            "player_id", "name", *_PLAYER_FIELDS, "condition_delta",
        }:
            raise ValueError("player world-state snapshot is invalid")
        player_id = row.get("player_id")
        if (
            not isinstance(player_id, str) or not player_id or len(player_id) > 128
            or not isinstance(row.get("name"), str) or len(row.get("name", "")) > 160
        ):
            raise ValueError("player world-state identity is invalid")
        for key in (
            "matches_played", "goals", "assists", "yellow_cards", "red_cards",
            "suspension_matches_left", "injury_matches_left",
        ):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("player world-state counter is invalid")
        for key in (
            "minutes_ema", "injury_severity", "medical_recovery_credit",
            "form_ema", "media_sentiment", "last_rating",
        ):
            value = row[key]
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("player world-state metric is invalid")
        condition = row["condition_delta"]
        if (
            not isinstance(condition, Mapping) or len(condition) > 32
            or any(
                not isinstance(key, str) or not key or len(key) > 64
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for key, value in condition.items()
            )
        ):
            raise ValueError("player world-state condition is invalid")
        ids.append(player_id)
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("player world-state identities are not canonical")
    injured = sum(row["injury_matches_left"] > 0 for row in players)
    suspended = sum(row["suspension_matches_left"] > 0 for row in players)
    unavailable = sum(
        row["injury_matches_left"] > 0
        or row["suspension_matches_left"] > 0
        for row in players
    )
    if metrics["injured_players"] != injured or metrics["suspended_players"] != suspended:
        raise ValueError("team world-state availability summary mismatch")
    if metrics["unavailable_players"] != unavailable:
        raise ValueError("team world-state availability summary mismatch")


def _team_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    include_society: bool,
    include_society_details: bool,
) -> dict[str, Any]:
    before_players = {row["player_id"]: row for row in before["players"]}
    after_players = {row["player_id"]: row for row in after["players"]}
    changes = []
    for player_id in sorted(set(before_players) | set(after_players)):
        left = before_players.get(player_id)
        right = after_players.get(player_id)
        if left is None or right is None:
            changes.append({
                "player_id": player_id,
                "name": str((right or left or {}).get("name") or ""),
                "change": "added" if left is None else "removed",
                "fields": {},
            })
            continue
        fields = {}
        for key in _PLAYER_FIELDS:
            if left[key] != right[key]:
                fields[key] = {"before": left[key], "after": right[key]}
        if left["condition_delta"] != right["condition_delta"]:
            fields["condition_delta"] = {
                "before": copy.deepcopy(left["condition_delta"]),
                "after": copy.deepcopy(right["condition_delta"]),
            }
        if fields:
            changes.append({
                "player_id": player_id, "name": right["name"],
                "change": "updated", "fields": fields,
            })
    deltas = {
        key: round(float(after["metrics"][key]) - float(before["metrics"][key]), 6)
        for key in _METRICS
    }
    result = {
        "metrics_delta": deltas,
        "players_changed": changes,
        "summary": {
            "changed_players": len(changes),
            "new_injuries": sum(
                row.get("fields", {}).get("injury_matches_left", {}).get("before", 0) == 0
                and row.get("fields", {}).get("injury_matches_left", {}).get("after", 0) > 0
                for row in changes
            ),
            "injuries_cleared": sum(
                row.get("fields", {}).get("injury_matches_left", {}).get("before", 0) > 0
                and row.get("fields", {}).get("injury_matches_left", {}).get("after", 0) == 0
                for row in changes
            ),
            "new_suspensions": sum(
                row.get("fields", {}).get("suspension_matches_left", {}).get("before", 0) == 0
                and row.get("fields", {}).get("suspension_matches_left", {}).get("after", 0) > 0
                for row in changes
            ),
            "suspensions_cleared": sum(
                row.get("fields", {}).get("suspension_matches_left", {}).get("before", 0) > 0
                and row.get("fields", {}).get("suspension_matches_left", {}).get("after", 0) == 0
                for row in changes
            ),
        },
    }
    if include_society:
        unavailable = society_public_snapshot(None, team_id=str(after["team_id"]))
        result["society_transition"] = society_public_transition(
            before.get("society", unavailable),
            after.get("society", unavailable),
            include_state_changes=include_society_details,
        )
    return result


def _build_fixture_world_state_transition(
    *, season_id: str, fixture_id: str, match_id: str,
    home: str, away: str,
    before_match: Mapping[str, Mapping[str, Any]],
    after_match: Mapping[str, Mapping[str, Any]],
    after_recovery: Mapping[str, Mapping[str, Any]] | None = None,
    schema_version: int,
) -> dict[str, Any]:
    teams = (home, away)
    for snapshots in (before_match, after_match):
        if not isinstance(snapshots, Mapping) or set(snapshots) != set(teams):
            raise ValueError("fixture world-state team coverage mismatch")
    if after_recovery is not None and (
        not isinstance(after_recovery, Mapping) or set(after_recovery) != set(teams)
    ):
        raise ValueError("fixture recovery-state team coverage mismatch")
    phases = {
        "before_match": copy.deepcopy(dict(before_match)),
        "after_match": copy.deepcopy(dict(after_match)),
        "after_recovery": copy.deepcopy(dict(after_recovery))
        if after_recovery is not None else None,
    }
    for phase in phases.values():
        if phase is not None:
            for team in teams:
                validate_team_state_snapshot(phase[team], team=team)
                if phase[team].get("schema_version") != schema_version:
                    raise ValueError(
                        "fixture world-state phase schema version mismatch"
                    )
    match_delta = {
        team: _team_diff(
            phases["before_match"][team],
            phases["after_match"][team],
            include_society=schema_version >= SOCIETY_WORLD_STATE_SCHEMA_VERSION,
            include_society_details=(schema_version >= WORLD_STATE_SCHEMA_VERSION),
        )
        for team in teams
    }
    recovery_delta = (
        {
            team: _team_diff(
                phases["after_match"][team],
                phases["after_recovery"][team],
                include_society=(
                    schema_version >= SOCIETY_WORLD_STATE_SCHEMA_VERSION
                ),
                include_society_details=(
                    schema_version >= WORLD_STATE_SCHEMA_VERSION
                ),
            )
            for team in teams
        }
        if phases["after_recovery"] is not None else None
    )
    payload = {
        "schema_version": schema_version,
        "season_id": season_id, "fixture_id": fixture_id, "match_id": match_id,
        "home": home, "away": away, "phases": phases,
        "match_delta": match_delta, "recovery_delta": recovery_delta,
        "recovery_complete": after_recovery is not None,
        "claim_boundary": _BOUNDARY,
    }
    payload["transition_identity"] = _identity(payload)
    return payload


def build_fixture_world_state_transition(
    *, season_id: str, fixture_id: str, match_id: str,
    home: str, away: str,
    before_match: Mapping[str, Mapping[str, Any]],
    after_match: Mapping[str, Mapping[str, Any]],
    after_recovery: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return _build_fixture_world_state_transition(
        season_id=season_id,
        fixture_id=fixture_id,
        match_id=match_id,
        home=home,
        away=away,
        before_match=before_match,
        after_match=after_match,
        after_recovery=after_recovery,
        schema_version=WORLD_STATE_SCHEMA_VERSION,
    )


def validate_fixture_world_state_transition(
    transition: Mapping[str, Any], *, season_id: str, fixture_id: str,
    match_id: str, home: str, away: str,
) -> None:
    if not isinstance(transition, Mapping):
        raise ValueError("fixture world-state transition is invalid")
    frozen = copy.deepcopy(dict(transition))
    identity = frozen.pop("transition_identity", None)
    if (
        set(transition) != {
            "schema_version", "season_id", "fixture_id", "match_id", "home", "away",
            "phases", "match_delta", "recovery_delta", "recovery_complete",
            "claim_boundary", "transition_identity",
        }
        or transition.get("schema_version") not in {
            LEGACY_WORLD_STATE_SCHEMA_VERSION,
            SOCIETY_WORLD_STATE_SCHEMA_VERSION,
            META_COUNTS_WORLD_STATE_SCHEMA_VERSION,
            META_GOVERNANCE_WORLD_STATE_SCHEMA_VERSION,
            WORLD_STATE_SCHEMA_VERSION,
        }
        or transition.get("season_id") != season_id
        or transition.get("fixture_id") != fixture_id
        or transition.get("match_id") != match_id
        or transition.get("home") != home or transition.get("away") != away
        or transition.get("claim_boundary") != _BOUNDARY
        or identity != _identity(frozen)
    ):
        raise ValueError("fixture world-state transition identity mismatch")
    phases = transition.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != {
        "before_match", "after_match", "after_recovery",
    }:
        raise ValueError("fixture world-state phases are invalid")
    expected = _build_fixture_world_state_transition(
        season_id=season_id, fixture_id=fixture_id, match_id=match_id,
        home=home, away=away, before_match=phases["before_match"],
        after_match=phases["after_match"], after_recovery=phases["after_recovery"],
        schema_version=int(transition["schema_version"]),
    )
    if transition != expected:
        raise ValueError("fixture world-state transition replay mismatch")


__all__ = [
    "WORLD_STATE_SCHEMA_VERSION", "build_fixture_world_state_transition",
    "capture_world_state", "validate_fixture_world_state_transition",
    "validate_team_state_snapshot",
]
