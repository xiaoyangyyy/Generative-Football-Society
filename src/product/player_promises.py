"""Evidence-bound named-player role promises for a managed season."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.simulation.player_development import validate_participation_evidence
from src.simulation.lineup import LineupSelection
from src.simulation.squad_registry import roster_identity


PLAYER_PROMISE_SCHEMA_VERSION = 1
PROMISE_ROLES = ("core", "rotation", "development")
MAX_PLAYER_PROMISES = 3
_HASH = re.compile(r"^[0-9a-f]{64}$")
_POLICY = {
    "core": {"minimum_start_share": 0.70, "minimum_minute_share": 0.60},
    "rotation": {"minimum_start_share": 0.35, "minimum_minute_share": 0.30},
    "development": {"minimum_start_share": 0.25, "minimum_minute_share": 0.20},
}
_CONTRACT_BOUNDARY = (
    "explicit fictional player-role promises over lineup use; no hidden morale, "
    "selection bonus, employment claim, or guaranteed development"
)
_PROGRESS_BOUNDARY = (
    "manager-controlled starting-selection evidence only; fulfilment does not imply "
    "that selection caused results or player development"
)
_OUTCOME_BOUNDARY = (
    "descriptive simulated minutes with explicit report coverage; no causal player "
    "grade, potential estimate, medical claim, or real-world evaluation"
)


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlayerRolePromise:
    player_id: str
    role: str

    def __post_init__(self) -> None:
        player_id = str(self.player_id or "").strip()
        if not player_id or len(player_id) > 128:
            raise ValueError("invalid promised player identity")
        if self.role not in PROMISE_ROLES:
            raise ValueError("unsupported promised player role")
        object.__setattr__(self, "player_id", player_id)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlayerRolePromise":
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"player_id", "role"}
        ):
            raise ValueError("player promise must contain player_id and role")
        return cls(str(payload.get("player_id") or ""), str(payload.get("role") or ""))

    def as_dict(self) -> dict[str, str]:
        return {"player_id": self.player_id, "role": self.role}


@dataclass(frozen=True)
class PlayerPromisePlan:
    promises: tuple[PlayerRolePromise, ...]

    def __post_init__(self) -> None:
        promises = tuple(self.promises)
        if not 1 <= len(promises) <= MAX_PLAYER_PROMISES:
            raise ValueError("player promise plan requires one to three promises")
        if not all(isinstance(item, PlayerRolePromise) for item in promises):
            raise ValueError("invalid player promise plan entry")
        if len({item.player_id for item in promises}) != len(promises):
            raise ValueError("player promises must use unique players")
        if len({item.role for item in promises}) != len(promises):
            raise ValueError("player promise roles must be unique")
        object.__setattr__(self, "promises", promises)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlayerPromisePlan":
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "promises"}
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("promises"), list)
        ):
            raise ValueError("player promise plan must be a versioned object")
        return cls(tuple(
            PlayerRolePromise.from_payload(item) for item in payload["promises"]
        ))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAYER_PROMISE_SCHEMA_VERSION,
            "promises": [item.as_dict() for item in self.promises],
        }


def _player_snapshot(player: Mapping[str, Any], promised_role: str) -> dict[str, Any]:
    age = player.get("age")
    if age is not None and (
        isinstance(age, bool) or not isinstance(age, int) or not 15 <= age <= 70
    ):
        raise ValueError("promised player age evidence is invalid")
    if promised_role == "development" and (age is None or age > 23):
        raise ValueError("development promise requires known age 23 or younger")
    policy = _POLICY[promised_role]
    return {
        "player_id": str(player.get("player_id") or ""),
        "name": str(player.get("name") or player.get("player_id") or ""),
        "position": str(player.get("role") or ""),
        "age": age,
        "promised_role": promised_role,
        "minimum_start_share": policy["minimum_start_share"],
        "minimum_minute_share": policy["minimum_minute_share"],
    }


def build_player_promise_contract(
    *, season_id: str, team: str, total_managed_matches: int,
    roster: Mapping[str, Any] | None, plan: PlayerPromisePlan | None,
    control: str,
) -> dict[str, Any]:
    if control not in {"manager", "deterministic_compatibility"}:
        raise ValueError("invalid player promise control")
    if (
        not season_id or not team
        or isinstance(total_managed_matches, bool)
        or not isinstance(total_managed_matches, int)
        or total_managed_matches < 3
    ):
        raise ValueError("invalid player promise season identity")
    if control == "manager" and plan is None:
        raise ValueError("manager player promise contract requires a plan")
    if control == "deterministic_compatibility" and plan is not None:
        raise ValueError("compatibility player promise contract must be empty")
    players = []
    roster_hash = None
    roster_evidence = "unavailable"
    if roster is not None:
        raw_players = roster.get("players")
        if not isinstance(raw_players, list):
            raise ValueError("player promise roster is invalid")
        by_id = {
            str(player.get("player_id") or ""): player
            for player in raw_players if isinstance(player, Mapping)
        }
        if len(by_id) != len(raw_players) or "" in by_id:
            raise ValueError("player promise roster identities are invalid")
        roster_hash = roster_identity(roster)
        roster_evidence = "effective_roster"
        if plan is not None:
            for promise in plan.promises:
                player = by_id.get(promise.player_id)
                if player is None:
                    raise ValueError("promised player is outside the effective roster")
                players.append(_player_snapshot(player, promise.role))
    elif plan is not None:
        raise ValueError("player promises require an available effective roster")
    contract = {
        "schema_version": PLAYER_PROMISE_SCHEMA_VERSION,
        "season_id": season_id,
        "team": team,
        "control": control,
        "roster_identity": roster_hash,
        "roster_evidence": roster_evidence,
        "total_managed_matches": total_managed_matches,
        "players": players,
        "board_consequence": {
            "fulfilled_confidence": 1, "missed_confidence": -2,
            "reputation_change": 0,
        },
        "claim_boundary": _CONTRACT_BOUNDARY,
    }
    contract["contract_id"] = _identity(contract)
    validate_player_promise_contract(contract)
    return contract


def validate_player_promise_contract(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("invalid player promise contract")
    frozen = copy.deepcopy(dict(contract))
    contract_id = frozen.pop("contract_id", None)
    players = contract.get("players")
    if (
        set(contract) != {
            "schema_version", "contract_id", "season_id", "team", "control",
            "roster_identity", "roster_evidence", "total_managed_matches",
            "players", "board_consequence", "claim_boundary",
        }
        or contract.get("schema_version") != 1
        or _HASH.fullmatch(str(contract_id or "")) is None
        or contract_id != _identity(frozen)
        or not str(contract.get("season_id") or "")
        or not str(contract.get("team") or "")
        or contract.get("control") not in {"manager", "deterministic_compatibility"}
        or contract.get("roster_evidence") not in {"effective_roster", "unavailable"}
        or isinstance(contract.get("total_managed_matches"), bool)
        or not isinstance(contract.get("total_managed_matches"), int)
        or contract["total_managed_matches"] < 3
        or not isinstance(players, list) or len(players) > MAX_PLAYER_PROMISES
        or contract.get("board_consequence") != {
            "fulfilled_confidence": 1, "missed_confidence": -2,
            "reputation_change": 0,
        }
        or contract.get("claim_boundary") != _CONTRACT_BOUNDARY
    ):
        raise ValueError("invalid player promise contract")
    roster_hash = contract.get("roster_identity")
    if (
        (contract["roster_evidence"] == "effective_roster")
        != (_HASH.fullmatch(str(roster_hash or "")) is not None)
        or (contract["control"] == "manager") != bool(players)
        or (players and contract["roster_evidence"] != "effective_roster")
    ):
        raise ValueError("invalid player promise roster boundary")
    ids, roles = [], []
    for row in players:
        if not isinstance(row, Mapping) or set(row) != {
            "player_id", "name", "position", "age", "promised_role",
            "minimum_start_share", "minimum_minute_share",
        }:
            raise ValueError("invalid promised player snapshot")
        player_id = row.get("player_id")
        promised_role = row.get("promised_role")
        age = row.get("age")
        if (
            not isinstance(player_id, str) or not player_id or len(player_id) > 128
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("position"), str) or not row["position"]
            or promised_role not in PROMISE_ROLES
            or row.get("minimum_start_share")
            != _POLICY[promised_role]["minimum_start_share"]
            or row.get("minimum_minute_share")
            != _POLICY[promised_role]["minimum_minute_share"]
            or (
                age is not None and (
                    isinstance(age, bool) or not isinstance(age, int)
                    or not 15 <= age <= 70
                )
            )
            or (promised_role == "development" and (age is None or age > 23))
        ):
            raise ValueError("invalid promised player evidence")
        ids.append(player_id); roles.append(promised_role)
    if len(ids) != len(set(ids)) or len(roles) != len(set(roles)):
        raise ValueError("duplicate player promise identity")


def player_promise_progress(
    contract: Mapping[str, Any], *, fixtures: Sequence[Mapping[str, Any]],
    final: bool,
) -> dict[str, Any]:
    validate_player_promise_contract(contract)
    managed = [
        row for row in fixtures
        if isinstance(row, Mapping)
        and contract["team"] in {row.get("home"), row.get("away")}
        and row.get("state") == "completed"
    ]
    if len(managed) > contract["total_managed_matches"]:
        raise ValueError("player promise evidence exceeds season contract")
    if final and len(managed) != contract["total_managed_matches"]:
        raise ValueError("final player promise selection evidence is incomplete")
    evidence = []
    starts = {row["player_id"]: [] for row in contract["players"]}
    bench = {row["player_id"]: [] for row in contract["players"]}
    excused = {row["player_id"]: [] for row in contract["players"]}
    unavailable = {row["player_id"]: [] for row in contract["players"]}
    seen = set()
    for fixture in managed:
        fixture_id = str(fixture.get("fixture_id") or "")
        matchday = fixture.get("matchday")
        if (
            not fixture_id or fixture_id in seen
            or isinstance(matchday, bool) or not isinstance(matchday, int)
            or matchday < 1
        ):
            raise ValueError("duplicate player promise fixture evidence")
        seen.add(fixture_id)
        decision = fixture.get("manager_decision")
        lineup = decision.get("lineup") if isinstance(decision, Mapping) else None
        availability = fixture.get("player_promise_availability")
        if contract["players"] and (
            not isinstance(availability, Mapping)
            or set(availability) != set(starts)
            or any(not isinstance(value, bool) for value in availability.values())
        ):
            availability = None
        parsed_lineup = (
            LineupSelection.from_payload(lineup)
            if isinstance(lineup, Mapping) else None
        )
        starters = set(parsed_lineup.starters) if parsed_lineup is not None else set()
        substitutes = set(parsed_lineup.bench) if parsed_lineup is not None else set()
        evidence.append({
            "fixture_id": fixture_id, "matchday": matchday,
            "lineup_available": isinstance(lineup, Mapping),
            "availability_available": isinstance(availability, Mapping),
            "promised_starters": sorted(starters & set(starts)),
            "promised_bench": sorted(substitutes & set(bench)),
            "excused_unavailable": sorted(
                player_id for player_id in starts
                if isinstance(availability, Mapping) and not availability[player_id]
            ),
        })
        for player_id in starts:
            if not isinstance(availability, Mapping) or not isinstance(lineup, Mapping):
                unavailable[player_id].append(fixture_id)
            elif not availability[player_id]:
                excused[player_id].append(fixture_id)
            elif player_id in starters:
                starts[player_id].append(fixture_id)
            elif player_id in substitutes:
                bench[player_id].append(fixture_id)
    if final and contract["players"] and any(unavailable.values()):
        raise ValueError("final player promise lineup evidence is incomplete")
    entries = []
    for player in contract["players"]:
        player_id = player["player_id"]
        known = (
            len(managed) - len(unavailable[player_id]) - len(excused[player_id])
        )
        start_share = len(starts[player_id]) / known if known else 0.0
        met = start_share >= player["minimum_start_share"]
        status = (
            "excused" if known == 0 and len(excused[player_id]) == len(managed) and managed
            else "evidence_unavailable" if known == 0 and managed
            else "fulfilled" if final and met
            else "missed" if final
            else "pending" if known == 0
            else "on_track" if met else "at_risk"
        )
        entries.append({
            **copy.deepcopy(player), "status": status,
            "completed_matches": len(managed), "known_lineups": known,
            "starts": len(starts[player_id]), "bench": len(bench[player_id]),
            "excused_unavailable": len(excused[player_id]),
            "start_share": round(start_share, 6),
            "start_fixture_ids": starts[player_id],
            "bench_fixture_ids": bench[player_id],
            "excused_fixture_ids": excused[player_id],
            "unavailable_fixture_ids": unavailable[player_id],
        })
    statuses = [row["status"] for row in entries]
    return {
        "schema_version": PLAYER_PROMISE_SCHEMA_VERSION,
        "contract_id": contract["contract_id"], "season_id": contract["season_id"],
        "team": contract["team"], "control": contract["control"],
        "final": final, "completed_matches": len(managed),
        "remaining_matches": contract["total_managed_matches"] - len(managed),
        "entries": entries, "evidence": evidence,
        "summary": {
            key: statuses.count(key) for key in (
                "fulfilled", "missed", "on_track", "at_risk", "pending",
                "evidence_unavailable",
                "excused",
            )
        },
        "board_consequence": copy.deepcopy(contract["board_consequence"]),
        "claim_boundary": _PROGRESS_BOUNDARY,
    }


def settle_player_promise_outcomes(
    contract: Mapping[str, Any], progress: Mapping[str, Any],
    participation: Mapping[str, Any],
) -> dict[str, Any] | None:
    validate_player_promise_contract(contract)
    if not contract["players"]:
        return None
    validate_participation_evidence(participation)
    if (
        participation.get("season_id") != contract["season_id"]
        or participation.get("team") != contract["team"]
        or progress.get("contract_id") != contract["contract_id"]
        or progress.get("final") is not True
    ):
        raise ValueError("player promise outcome source identity mismatch")
    coverage = participation["coverage"]
    scheduled = len(participation["scheduled_fixture_ids"])
    totals = participation["player_totals"]
    rows = []
    for player in contract["players"]:
        total = totals.get(player["player_id"], {"minutes": 0.0, "appearances": 0})
        minutes = round(float(total["minutes"]), 3)
        share = min(1.0, minutes / max(90.0, scheduled * 90.0))
        observed_status = (
            "evidence_partial" if coverage == "partial"
            else "evidence_unavailable" if coverage == "unavailable"
            else "fulfilled" if share >= player["minimum_minute_share"]
            else "missed"
        )
        selection = next(
            row for row in progress["entries"]
            if row["player_id"] == player["player_id"]
        )
        rows.append({
            "player_id": player["player_id"], "name": player["name"],
            "position": player["position"], "age": player["age"],
            "promised_role": player["promised_role"],
            "selection_status": selection["status"],
            "start_share": selection["start_share"],
            "minutes": minutes, "appearances": int(total["appearances"]),
            "scheduled_matches": scheduled, "minute_share": round(share, 6),
            "minimum_minute_share": player["minimum_minute_share"],
            "observed_status": observed_status,
        })
    outcome = {
        "schema_version": PLAYER_PROMISE_SCHEMA_VERSION,
        "outcome_id": "", "season_id": contract["season_id"],
        "team": contract["team"], "contract_id": contract["contract_id"],
        "participation_identity": _identity(participation),
        "coverage": coverage, "players": rows,
        "claim_boundary": _OUTCOME_BOUNDARY,
    }
    frozen = dict(outcome); frozen.pop("outcome_id")
    outcome["outcome_id"] = _identity(frozen)
    return outcome


__all__ = [
    "PlayerPromisePlan", "PlayerRolePromise", "build_player_promise_contract",
    "player_promise_progress", "settle_player_promise_outcomes",
    "validate_player_promise_contract",
]
