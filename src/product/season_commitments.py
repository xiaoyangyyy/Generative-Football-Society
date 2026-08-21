"""Replayable multi-week commitments for one managed simulated season."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


COMMITMENT_SCHEMA_VERSION = 1
TACTIC_POLICIES = ("club_identity", "adaptive")
ROTATION_POLICIES = ("trust_core", "share_load")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_TACTICS = {
    "team_identity", "balanced", "tiki_taka", "gegenpress",
    "counter_attack", "low_block_counter", "direct_vertical",
}
_ROTATIONS = {"strongest", "balanced", "rotate", "unrecorded"}
_PLAN_BOUNDARY = (
    "explicit simulated-season promises; they affect transparent board review "
    "and never modify match physics or hidden player attributes"
)
_CONTRACT_BOUNDARY = (
    "frozen deterministic gameplay contract derived from the season plan and "
    "club identity; not a performance forecast or real-world management advice"
)
_PROGRESS_BOUNDARY = (
    "descriptive replay of frozen promises against completed simulated decisions; "
    "frequency is not evidence that a tactic or rotation caused results"
)


def _identity(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SeasonCommitmentPlan:
    tactic_policy: str = "club_identity"
    rotation_policy: str = "share_load"

    def __post_init__(self) -> None:
        if self.tactic_policy not in TACTIC_POLICIES:
            raise ValueError("unsupported season tactic commitment")
        if self.rotation_policy not in ROTATION_POLICIES:
            raise ValueError("unsupported season rotation commitment")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SeasonCommitmentPlan":
        allowed = {
            "schema_version", "tactic_policy", "rotation_policy", "claim_boundary",
        }
        if (
            not isinstance(payload, Mapping)
            or not set(payload) <= allowed
            or payload.get("schema_version", 1) != 1
            or (
                "claim_boundary" in payload
                and payload.get("claim_boundary") != _PLAN_BOUNDARY
            )
        ):
            raise ValueError("season commitments must be a versioned object")
        return cls(
            tactic_policy=str(payload.get("tactic_policy") or "club_identity"),
            rotation_policy=str(payload.get("rotation_policy") or "share_load"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMMITMENT_SCHEMA_VERSION,
            "tactic_policy": self.tactic_policy,
            "rotation_policy": self.rotation_policy,
            "claim_boundary": _PLAN_BOUNDARY,
        }


def _identity_tactic(
    club_strategies: Mapping[str, Any] | None, team: str,
) -> str:
    profiles = (
        club_strategies.get("profiles")
        if isinstance(club_strategies, Mapping) else None
    )
    if isinstance(profiles, list):
        matches = [
            row for row in profiles
            if isinstance(row, Mapping) and row.get("team") == team
        ]
        if len(matches) == 1 and isinstance(matches[0].get("primary_tactic"), str):
            return str(matches[0]["primary_tactic"])
    return "team_identity"


def build_commitment_contract_from_sources(
    *, season_id: str, plan: Mapping[str, Any],
    club_strategies: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    team = plan.get("manager_team")
    if team is None:
        return None
    if not isinstance(team, str) or not team.strip():
        raise ValueError("commitment manager team is invalid")
    teams = plan.get("teams")
    legs = plan.get("legs")
    if (
        not isinstance(teams, list) or team not in teams
        or isinstance(legs, bool) or not isinstance(legs, int) or legs not in {1, 2}
    ):
        raise ValueError("commitment season plan is invalid")
    commitments = SeasonCommitmentPlan.from_payload(
        plan.get("manager_commitments") or {},
    )
    total = (len(teams) - 1) * legs
    contract = {
        "schema_version": COMMITMENT_SCHEMA_VERSION,
        "season_id": season_id,
        "team": team,
        "total_managed_matches": total,
        "tactic": {
            "policy": commitments.tactic_policy,
            "identity_tactic": _identity_tactic(club_strategies, team),
            "minimum_share": 0.6 if commitments.tactic_policy == "club_identity" else None,
            "minimum_distinct": 2 if commitments.tactic_policy == "adaptive" else None,
            "maximum_dominant_share": 0.8 if commitments.tactic_policy == "adaptive" else None,
        },
        "rotation": {
            "policy": commitments.rotation_policy,
            "minimum_share": 0.6 if commitments.rotation_policy == "trust_core" else 0.5,
            "counted_rotations": (
                ["strongest"] if commitments.rotation_policy == "trust_core"
                else ["balanced", "rotate"]
            ),
        },
        "board_consequence": {
            "fulfilled_confidence": 3,
            "missed_confidence": -4,
            "fulfilled_reputation": 1,
            "missed_reputation": -1,
        },
        "claim_boundary": _CONTRACT_BOUNDARY,
    }
    contract["contract_id"] = _identity(contract)
    validate_commitment_contract(contract)
    return contract


def build_commitment_contract(state: Mapping[str, Any]) -> dict[str, Any] | None:
    return build_commitment_contract_from_sources(
        season_id=str(state.get("season_id") or ""),
        plan=state.get("plan") or {},
        club_strategies=state.get("club_strategies"),
    )


def validate_commitment_contract(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError("invalid season commitment contract")
    frozen = copy.deepcopy(dict(contract))
    contract_id = frozen.pop("contract_id", None)
    tactic = contract.get("tactic")
    rotation = contract.get("rotation")
    consequence = contract.get("board_consequence")
    if (
        set(contract) != {
            "schema_version", "contract_id", "season_id", "team",
            "total_managed_matches", "tactic", "rotation",
            "board_consequence", "claim_boundary",
        }
        or contract.get("schema_version") != 1
        or _HASH.fullmatch(str(contract_id or "")) is None
        or contract_id != _identity(frozen)
        or not str(contract.get("season_id") or "")
        or not str(contract.get("team") or "")
        or isinstance(contract.get("total_managed_matches"), bool)
        or not isinstance(contract.get("total_managed_matches"), int)
        or contract["total_managed_matches"] < 3
        or not isinstance(tactic, Mapping)
        or set(tactic) != {
            "policy", "identity_tactic", "minimum_share",
            "minimum_distinct", "maximum_dominant_share",
        }
        or tactic.get("policy") not in TACTIC_POLICIES
        or not isinstance(tactic.get("identity_tactic"), str)
        or not tactic["identity_tactic"]
        or not isinstance(rotation, Mapping)
        or set(rotation) != {"policy", "minimum_share", "counted_rotations"}
        or rotation.get("policy") not in ROTATION_POLICIES
        or not isinstance(rotation.get("counted_rotations"), list)
        or not isinstance(consequence, Mapping)
        or consequence != {
            "fulfilled_confidence": 3, "missed_confidence": -4,
            "fulfilled_reputation": 1, "missed_reputation": -1,
        }
        or contract.get("claim_boundary") != _CONTRACT_BOUNDARY
    ):
        raise ValueError("invalid season commitment contract")
    if tactic["policy"] == "club_identity":
        valid_tactic = (
            tactic["minimum_share"] == 0.6
            and tactic["minimum_distinct"] is None
            and tactic["maximum_dominant_share"] is None
        )
    else:
        valid_tactic = (
            tactic["minimum_share"] is None
            and tactic["minimum_distinct"] == 2
            and tactic["maximum_dominant_share"] == 0.8
        )
    expected_rotations = (
        ["strongest"] if rotation["policy"] == "trust_core"
        else ["balanced", "rotate"]
    )
    expected_share = 0.6 if rotation["policy"] == "trust_core" else 0.5
    if (
        not valid_tactic or rotation["minimum_share"] != expected_share
        or rotation["counted_rotations"] != expected_rotations
    ):
        raise ValueError("season commitment policy mismatch")


def _frequency_status(*, met: bool, recorded: int, final: bool) -> str:
    if final:
        return "fulfilled" if met else "missed"
    if recorded == 0:
        return "pending"
    return "on_track" if met else "at_risk"


def commitment_progress_from_evidence(
    contract: Mapping[str, Any], *, journal: Sequence[Mapping[str, Any]],
    objective: Mapping[str, Any], final: bool,
) -> dict[str, Any]:
    validate_commitment_contract(contract)
    if isinstance(journal, (str, bytes)) or not isinstance(journal, Sequence):
        raise ValueError("commitment journal is invalid")
    evidence = []
    seen_fixtures = set()
    tactic_usage: dict[str, int] = {}
    rotation_usage: dict[str, int] = {}
    for entry in journal:
        if not isinstance(entry, Mapping):
            raise ValueError("commitment journal entry is invalid")
        fixture_id = entry.get("fixture_id")
        matchday = entry.get("matchday")
        tactic = entry.get("tactic")
        rotation = entry.get("rotation")
        if (
            not all(isinstance(value, str) and value for value in (fixture_id, tactic, rotation))
            or fixture_id in seen_fixtures
            or isinstance(matchday, bool) or not isinstance(matchday, int)
            or matchday < 1 or tactic not in _TACTICS | {"unrecorded"}
            or rotation not in _ROTATIONS
        ):
            raise ValueError("commitment journal entry is incomplete")
        seen_fixtures.add(fixture_id)
        evidence.append({
            "fixture_id": fixture_id, "matchday": matchday,
            "tactic": tactic, "rotation": rotation,
        })
        tactic_usage[tactic] = tactic_usage.get(tactic, 0) + 1
        rotation_usage[rotation] = rotation_usage.get(rotation, 0) + 1
    recorded = len(evidence)
    if recorded > contract["total_managed_matches"]:
        raise ValueError("commitment evidence exceeds the frozen season")
    if final and recorded != contract["total_managed_matches"]:
        raise ValueError("final commitment evidence is incomplete")
    tactic_contract = contract["tactic"]
    dominant_count = max(tactic_usage.values(), default=0)
    dominant_share = dominant_count / recorded if recorded else 0.0
    identity_count = tactic_usage.get(tactic_contract["identity_tactic"], 0)
    identity_share = identity_count / recorded if recorded else 0.0
    tactic_met = (
        identity_share >= tactic_contract["minimum_share"]
        if tactic_contract["policy"] == "club_identity"
        else len(tactic_usage) >= tactic_contract["minimum_distinct"]
        and dominant_share <= tactic_contract["maximum_dominant_share"]
    )
    rotation_contract = contract["rotation"]
    rotation_count = sum(
        rotation_usage.get(name, 0) for name in rotation_contract["counted_rotations"]
    )
    rotation_share = rotation_count / recorded if recorded else 0.0
    rotation_met = rotation_share >= rotation_contract["minimum_share"]
    objective_status = objective.get("status")
    if objective_status not in {
        "achieved", "missed", "unreachable", "currently_meeting", "in_progress",
    }:
        raise ValueError("commitment objective status is invalid")
    if final and objective_status not in {"achieved", "missed"}:
        raise ValueError("final commitment objective is unsettled")
    board_status = (
        "fulfilled" if objective_status == "achieved"
        else "missed" if objective_status == "missed"
        else "unreachable" if objective_status == "unreachable"
        else "on_track" if objective_status == "currently_meeting"
        else "pending"
    )
    entries = [
        {
            "id": "board_objective", "status": board_status,
            "metric": {
                "objective": objective.get("kind"),
                "position": objective.get("current_position"),
                "points": objective.get("current_points"),
            },
        },
        {
            "id": "tactical_identity", "status": _frequency_status(
                met=tactic_met, recorded=recorded, final=final,
            ),
            "metric": {
                "policy": tactic_contract["policy"],
                "identity_tactic": tactic_contract["identity_tactic"],
                "identity_share": round(identity_share, 6),
                "distinct_tactics": len(tactic_usage),
                "dominant_share": round(dominant_share, 6),
            },
        },
        {
            "id": "squad_stewardship", "status": _frequency_status(
                met=rotation_met, recorded=recorded, final=final,
            ),
            "metric": {
                "policy": rotation_contract["policy"],
                "counted_rotations": list(rotation_contract["counted_rotations"]),
                "counted_share": round(rotation_share, 6),
            },
        },
    ]
    statuses = [entry["status"] for entry in entries]
    result = {
        "schema_version": COMMITMENT_SCHEMA_VERSION,
        "contract_id": contract["contract_id"],
        "season_id": contract["season_id"],
        "team": contract["team"],
        "final": final,
        "recorded_matches": recorded,
        "remaining_matches": contract["total_managed_matches"] - recorded,
        "board_consequence": copy.deepcopy(contract["board_consequence"]),
        "entries": entries,
        "evidence": evidence,
        "summary": {
            name: statuses.count(name)
            for name in (
                "fulfilled", "missed", "unreachable", "on_track",
                "at_risk", "pending",
            )
        },
        "claim_boundary": _PROGRESS_BOUNDARY,
    }
    return result


def commitment_progress(
    state: Mapping[str, Any], objective: Mapping[str, Any],
) -> dict[str, Any] | None:
    contract = state.get("season_commitments")
    if contract is None:
        return None
    team = contract.get("team") if isinstance(contract, Mapping) else None
    fixtures = [
        row for row in state.get("fixtures") or []
        if isinstance(row, Mapping) and team in {row.get("home"), row.get("away")}
        and row.get("state") == "completed"
    ]
    journal = []
    for fixture in fixtures:
        decision = fixture.get("manager_decision")
        if not isinstance(decision, Mapping):
            journal.append({
                "fixture_id": fixture.get("fixture_id"),
                "matchday": fixture.get("matchday"),
                "tactic": "unrecorded", "rotation": "unrecorded",
            })
        else:
            journal.append({
                "fixture_id": fixture.get("fixture_id"),
                "matchday": fixture.get("matchday"),
                "tactic": decision.get("tactic", "team_identity"),
                "rotation": decision.get("rotation", "balanced"),
            })
    return commitment_progress_from_evidence(
        contract, journal=journal, objective=objective,
        final=state.get("state") == "complete",
    )


__all__ = [
    "SeasonCommitmentPlan", "build_commitment_contract",
    "build_commitment_contract_from_sources", "commitment_progress",
    "commitment_progress_from_evidence", "validate_commitment_contract",
]
