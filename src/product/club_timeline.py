"""Deterministic club situations that turn season evidence into manager choices."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TIMELINE_SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "deterministic gameplay situation from prior simulated season evidence; "
    "choices constrain the next decision and do not predict or cause a result"
)
RESOLUTION_BOUNDARY = (
    "frozen manager gameplay commitment applied through existing tactic or "
    "rotation mechanics; no hidden status or win-probability bonus"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClubEventChoice:
    event_id: str
    event_identity: str
    choice_id: str

    def __post_init__(self) -> None:
        if not str(self.event_id).strip() or len(str(self.event_id)) > 192:
            raise ValueError("invalid club event identity")
        if _HASH.fullmatch(str(self.event_identity or "")) is None:
            raise ValueError("invalid club event evidence identity")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", str(self.choice_id or "")):
            raise ValueError("invalid club event choice")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ClubEventChoice":
        if not isinstance(payload, Mapping) or payload.get("schema_version", 1) != 1:
            raise ValueError("club event choice must be a versioned object")
        return cls(
            event_id=str(payload.get("event_id") or ""),
            event_identity=str(payload.get("event_identity") or ""),
            choice_id=str(payload.get("choice_id") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "event_id": self.event_id,
            "event_identity": self.event_identity, "choice_id": self.choice_id,
        }


def _result(fixture: Mapping[str, Any], team: str) -> tuple[str, int]:
    score = fixture["score"]
    home = fixture["home"] == team
    goals_for = score["home"] if home else score["away"]
    goals_against = score["away"] if home else score["home"]
    if goals_for > goals_against:
        return "win", 3
    if goals_for == goals_against:
        return "draw", 1
    return "loss", 0


def _standings(fixtures: Sequence[Mapping[str, Any]], teams: Sequence[str]) -> list[dict[str, Any]]:
    rows = {
        team: {"team": team, "points": 0, "goals_for": 0, "goals_against": 0}
        for team in teams
    }
    for fixture in fixtures:
        if fixture.get("state") != "completed":
            continue
        home, away = fixture["home"], fixture["away"]
        hg, ag = fixture["score"]["home"], fixture["score"]["away"]
        rows[home]["goals_for"] += hg; rows[home]["goals_against"] += ag
        rows[away]["goals_for"] += ag; rows[away]["goals_against"] += hg
        rows[home]["points"] += 3 if hg > ag else 1 if hg == ag else 0
        rows[away]["points"] += 3 if ag > hg else 1 if hg == ag else 0
    ordered = sorted(rows.values(), key=lambda row: (
        -row["points"], -(row["goals_for"] - row["goals_against"]),
        -row["goals_for"], row["team"].casefold(),
    ))
    for index, row in enumerate(ordered, 1):
        row["position"] = index
    return ordered


def _choice(
    choice_id: str, *, label: str, constraint: Mapping[str, Any], tradeoff: str,
) -> dict[str, Any]:
    return {
        "choice_id": choice_id, "label": label,
        "constraint": dict(constraint), "tradeoff": tradeoff,
    }


def derive_club_situation(
    state: Mapping[str, Any], fixture_id: str | None = None,
) -> dict[str, Any] | None:
    """Derive at most one situation from evidence preceding a managed fixture."""
    plan = state.get("plan") if isinstance(state, Mapping) else None
    fixtures = state.get("fixtures") if isinstance(state, Mapping) else None
    team = plan.get("manager_team") if isinstance(plan, Mapping) else None
    teams = plan.get("teams") if isinstance(plan, Mapping) else None
    if not team or not isinstance(teams, list) or not isinstance(fixtures, list):
        return None
    managed = [
        row for row in fixtures if isinstance(row, Mapping)
        and team in {row.get("home"), row.get("away")}
    ]
    target = next((
        row for row in managed
        if row.get("fixture_id") == fixture_id
    ), None) if fixture_id is not None else next((
        row for row in managed if row.get("state") != "completed"
    ), None)
    if target is None or fixture_id is None and target.get("state") == "completed":
        return None
    target_order = (int(target["matchday"]), int(target["order"]))
    prior = [
        row for row in managed
        if row.get("state") == "completed"
        and (int(row["matchday"]), int(row["order"])) < target_order
        and isinstance(row.get("manager_decision"), Mapping)
    ]
    if len(prior) < 2:
        return None
    prior.sort(key=lambda row: (int(row["matchday"]), int(row["order"])))
    last_two = prior[-2:]
    decisions = [row["manager_decision"] for row in last_two]
    results = [_result(row, team) for row in last_two]
    completed_ids = [row["fixture_id"] for row in last_two]
    kind = title = recommended = None
    trigger: dict[str, Any]
    choices: list[dict[str, Any]]
    if all(row.get("rotation") == "strongest" for row in decisions):
        kind, title, recommended = "squad_load", "Repeated strongest-lineup load", "protect_squad"
        trigger = {
            "source": "two_prior_completed_manager_decisions",
            "fixture_ids": completed_ids, "strongest_rotations": 2,
        }
        choices = [
            _choice("protect_squad", label="Protect the squad", constraint={"rotation_in": ["balanced", "rotate"]}, tradeoff="reduces selection strength to use existing lower-load rotation mechanics"),
            _choice("push_starters", label="Push the starters", constraint={"rotation_in": ["strongest"]}, tradeoff="keeps the strongest selection and accepts its existing load tradeoff"),
        ]
    else:
        objective = str(plan.get("manager_objective") or "top_half")
        completed_count = len(prior)
        midpoint = (len(managed) + 1) // 2
        table = _standings(fixtures, teams)
        manager_row = next(row for row in table if row["team"] == team)
        target_position = 1 if objective == "champion" else (len(teams) + 1) // 2
        objective_met = (
            manager_row["points"] >= int(plan.get("manager_points_target") or 0)
            if objective == "points_target"
            else manager_row["position"] <= target_position
        )
        dominant = sorted(
            {str(row["manager_decision"]["tactic"]) for row in prior},
            key=lambda tactic: (
                -sum(row["manager_decision"].get("tactic") == tactic for row in prior),
                tactic,
            ),
        )[0]
        last_tactic = str(decisions[-1]["tactic"])
        if completed_count >= midpoint and not objective_met:
            kind, title, recommended = "objective_pressure", "Objective checkpoint requires a response", "change_approach"
            trigger = {
                "source": "persisted_standings_and_frozen_objective",
                "fixture_ids": completed_ids, "completed_managed_matches": completed_count,
                "midpoint": midpoint, "objective": objective,
                "position": manager_row["position"], "points": manager_row["points"],
                "reference_tactic": dominant,
            }
            choices = [
                _choice("hold_course", label="Hold the course", constraint={"tactic_equals": dominant}, tradeoff="preserves the season's most-used tactical identity"),
                _choice("change_approach", label="Change the approach", constraint={"tactic_not_equals": dominant}, tradeoff="requires a different supported tactic for the next fixture"),
            ]
        elif sum(points for _outcome, points in results) <= 1:
            kind, title, recommended = "form_response", "Poor recent form requires a response", "reset_approach"
            trigger = {
                "source": "two_prior_completed_results", "fixture_ids": completed_ids,
                "recent_points": sum(points for _outcome, points in results),
                "outcomes": [outcome for outcome, _points in results],
                "reference_tactic": last_tactic,
            }
            choices = [
                _choice("stabilize", label="Stabilize the team", constraint={"tactic_equals": last_tactic}, tradeoff="keeps the last supported tactic despite recent results"),
                _choice("reset_approach", label="Reset the approach", constraint={"tactic_not_equals": last_tactic}, tradeoff="requires a different supported tactic for the next fixture"),
            ]
        elif all(outcome == "win" for outcome, _points in results):
            kind, title, recommended = "momentum_management", "Winning momentum creates an identity choice", "keep_identity"
            trigger = {
                "source": "two_prior_completed_results", "fixture_ids": completed_ids,
                "recent_points": 6, "outcomes": ["win", "win"],
                "reference_tactic": last_tactic,
            }
            choices = [
                _choice("keep_identity", label="Keep the identity", constraint={"tactic_equals": last_tactic}, tradeoff="retains the last supported tactic"),
                _choice("surprise_opponent", label="Change before opponents adapt", constraint={"tactic_not_equals": last_tactic}, tradeoff="requires a different supported tactic without claiming a counter advantage"),
            ]
        else:
            return None
    event = {
        "schema_version": 1,
        "event_id": f"{state['season_id']}:{target['fixture_id']}:{kind}",
        "season_id": state["season_id"], "team": team,
        "fixture_id": target["fixture_id"], "matchday": target["matchday"],
        "kind": kind, "title": title, "trigger": trigger,
        "choices": choices, "recommended_choice": recommended,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    event["event_identity"] = _identity(event)
    validate_club_situation(event)
    return event


def validate_club_situation(event: Mapping[str, Any]) -> None:
    if not isinstance(event, Mapping):
        raise ValueError("invalid club situation")
    frozen = copy.deepcopy(dict(event)); identity = frozen.pop("event_identity", None)
    choices = event.get("choices")
    if (
        set(event) != {
            "schema_version", "event_id", "event_identity", "season_id", "team",
            "fixture_id", "matchday", "kind", "title", "trigger", "choices",
            "recommended_choice", "claim_boundary",
        }
        or event.get("schema_version") != 1
        or _HASH.fullmatch(str(identity or "")) is None or identity != _identity(frozen)
        or not str(event.get("event_id") or "").strip()
        or not str(event.get("season_id") or "").strip()
        or not str(event.get("team") or "").strip()
        or not str(event.get("fixture_id") or "").strip()
        or isinstance(event.get("matchday"), bool) or not isinstance(event.get("matchday"), int)
        or event["matchday"] < 1
        or event.get("kind") not in {
            "squad_load", "objective_pressure", "form_response", "momentum_management",
        }
        or not isinstance(event.get("trigger"), Mapping)
        or not isinstance(choices, list) or len(choices) != 2
        or len({row.get("choice_id") for row in choices if isinstance(row, Mapping)}) != 2
        or event.get("recommended_choice") not in {
            row.get("choice_id") for row in choices if isinstance(row, Mapping)
        }
        or event.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("invalid club situation identity")
    for row in choices:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"choice_id", "label", "constraint", "tradeoff"}
            or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", str(row.get("choice_id") or ""))
            or not str(row.get("label") or "").strip()
            or not isinstance(row.get("constraint"), Mapping)
            or not str(row.get("tradeoff") or "").strip()
        ):
            raise ValueError("invalid club situation choice contract")


def choice_matches_decision(
    event: Mapping[str, Any], choice: ClubEventChoice, *, tactic: str, rotation: str,
) -> bool:
    validate_club_situation(event)
    if choice.event_id != event["event_id"] or choice.event_identity != event["event_identity"]:
        return False
    rows = [row for row in event["choices"] if row["choice_id"] == choice.choice_id]
    if len(rows) != 1:
        return False
    constraint = rows[0]["constraint"]
    return (
        ("rotation_in" not in constraint or rotation in constraint["rotation_in"])
        and ("tactic_equals" not in constraint or tactic == constraint["tactic_equals"])
        and ("tactic_not_equals" not in constraint or tactic != constraint["tactic_not_equals"])
    )


def compatible_choice(event: Mapping[str, Any], *, tactic: str, rotation: str) -> ClubEventChoice:
    matches = [
        ClubEventChoice(event["event_id"], event["event_identity"], row["choice_id"])
        for row in event["choices"]
        if choice_matches_decision(
            event, ClubEventChoice(event["event_id"], event["event_identity"], row["choice_id"]),
            tactic=tactic, rotation=rotation,
        )
    ]
    if len(matches) != 1:
        raise ValueError("manager decision does not resolve the club situation unambiguously")
    return matches[0]


def build_timeline_resolution(
    event: Mapping[str, Any], choice: ClubEventChoice, *, tactic: str,
    rotation: str, control: str,
) -> dict[str, Any]:
    if control not in {"manager", "deterministic_compatibility"}:
        raise ValueError("invalid club situation control boundary")
    if not choice_matches_decision(event, choice, tactic=tactic, rotation=rotation):
        raise ValueError("manager decision conflicts with the club situation choice")
    resolution = {
        "schema_version": 1, "event": copy.deepcopy(dict(event)),
        "choice": choice.as_dict(), "control": control,
        "decision": {"tactic": tactic, "rotation": rotation},
        "claim_boundary": RESOLUTION_BOUNDARY,
    }
    resolution["resolution_id"] = _identity(resolution)
    return resolution


def validate_club_timeline(state: Mapping[str, Any]) -> None:
    timeline = state.get("club_timeline")
    if timeline is None:
        if any(
            isinstance(fixture, Mapping)
            and isinstance(fixture.get("manager_decision"), Mapping)
            and fixture["manager_decision"].get("club_event_choice") is not None
            for fixture in state.get("fixtures") or []
        ):
            raise ValueError("manager decision club event is missing from timeline")
        return
    events = timeline.get("events") if isinstance(timeline, Mapping) else None
    if (
        not isinstance(timeline, Mapping) or set(timeline) != {"schema_version", "events"}
        or timeline.get("schema_version") != 1 or not isinstance(events, list)
        or len(events) > len(state.get("fixtures") or [])
    ):
        raise ValueError("invalid club timeline")
    by_fixture = {}
    for resolution in events:
        if not isinstance(resolution, Mapping):
            raise ValueError("invalid club timeline resolution")
        frozen = copy.deepcopy(dict(resolution)); resolution_id = frozen.pop("resolution_id", None)
        event = resolution.get("event"); choice_payload = resolution.get("choice")
        decision = resolution.get("decision")
        if (
            set(resolution) != {
                "schema_version", "resolution_id", "event", "choice", "control",
                "decision", "claim_boundary",
            }
            or resolution.get("schema_version") != 1
            or _HASH.fullmatch(str(resolution_id or "")) is None
            or resolution_id != _identity(frozen)
            or not isinstance(event, Mapping) or not isinstance(choice_payload, Mapping)
            or resolution.get("control") not in {"manager", "deterministic_compatibility"}
            or not isinstance(decision, Mapping)
            or set(decision) != {"tactic", "rotation"}
            or resolution.get("claim_boundary") != RESOLUTION_BOUNDARY
        ):
            raise ValueError("invalid club timeline resolution identity")
        validate_club_situation(event)
        fixture_id = event["fixture_id"]
        if fixture_id in by_fixture:
            raise ValueError("duplicate club situation resolution")
        by_fixture[fixture_id] = resolution
        expected_event = derive_club_situation(state, fixture_id)
        fixture = next((row for row in state["fixtures"] if row["fixture_id"] == fixture_id), None)
        payload = fixture.get("manager_decision") if isinstance(fixture, Mapping) else None
        if expected_event != event or not isinstance(payload, Mapping):
            raise ValueError("club situation source replay mismatch")
        choice = ClubEventChoice.from_payload(choice_payload)
        if payload.get("club_event_choice") != choice.as_dict():
            raise ValueError("club situation decision identity mismatch")
        if decision != {"tactic": payload.get("tactic"), "rotation": payload.get("rotation")}:
            raise ValueError("club situation decision replay mismatch")
        if not choice_matches_decision(
            event, choice, tactic=str(decision["tactic"]), rotation=str(decision["rotation"]),
        ):
            raise ValueError("club situation choice constraint mismatch")
    for fixture in state.get("fixtures") or []:
        payload = fixture.get("manager_decision") if isinstance(fixture, Mapping) else None
        if isinstance(payload, Mapping) and payload.get("club_event_choice") is not None:
            if fixture["fixture_id"] not in by_fixture:
                raise ValueError("manager decision club event is missing from timeline")


def club_timeline_view(state: Mapping[str, Any]) -> dict[str, Any]:
    validate_club_timeline(state)
    events = (state.get("club_timeline") or {}).get("events") or []
    current = derive_club_situation(state)
    if current is not None and any(
        row["event"]["fixture_id"] == current["fixture_id"] for row in events
    ):
        current = None
    return {
        "schema_version": 1, "available": bool(current or events),
        "current_situation": current,
        "resolved_count": len(events),
        "history": list(reversed(copy.deepcopy(events[-12:]))),
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "ClubEventChoice", "build_timeline_resolution", "club_timeline_view",
    "compatible_choice", "derive_club_situation", "validate_club_timeline",
]
