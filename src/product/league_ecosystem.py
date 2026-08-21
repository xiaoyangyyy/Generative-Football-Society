"""Deterministic long-horizon recruitment policy for non-player clubs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping

from src.simulation.squad_registry import (
    RecruitmentMove,
    RecruitmentPlan,
    generate_recruitment_market,
    recruitment_market_id,
    roster_identity,
    squad_player_quality,
)


ECOSYSTEM_SCHEMA_VERSION = 1
MAX_TRANSITIONS = 12
_SEASON_PATTERN = re.compile(r"^season-(\d{4,})$")
_EVENT_FIELDS = {
    "schema_version", "from_season_id", "to_season_id", "archive_identity",
    "manager_controlled_team", "participating_teams", "ai_clubs",
    "claim_boundary",
}
_CLUB_FIELDS = {
    "team", "settlement_identity", "decision", "squad_transaction_identity",
}


def _identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strategy(position: int, league_size: int) -> tuple[str, int, float]:
    if position > math.ceil(league_size / 2):
        return "rebuild", 2, 0.012
    return "selective", 1, 0.025


def ai_club_recruitment_decision(
    archive: Mapping[str, Any], roster: Mapping[str, Any] | None, *,
    team: str, next_season_index: int, allowance: int,
) -> dict[str, Any]:
    """Return a pure, explainable AI recruitment decision for one club."""
    standings = archive.get("final_standings") if isinstance(archive, Mapping) else None
    rows = [
        row for row in standings or []
        if isinstance(row, Mapping) and row.get("team") == team
    ]
    if len(rows) != 1:
        raise ValueError("AI recruitment standings identity mismatch")
    position = rows[0].get("position")
    if (
        isinstance(position, bool) or not isinstance(position, int)
        or not isinstance(standings, list) or not 1 <= position <= len(standings)
    ):
        raise ValueError("AI recruitment standings are invalid")
    if (
        isinstance(allowance, bool) or not isinstance(allowance, int)
        or not 0 <= allowance <= 8
    ):
        raise ValueError("AI recruitment allowance is invalid")
    strategy, maximum_moves, minimum_improvement = _strategy(
        position, len(standings),
    )
    spending_cap = allowance if strategy == "rebuild" else min(allowance, 5)
    base = {
        "schema_version": ECOSYSTEM_SCHEMA_VERSION,
        "team": team,
        "position": position,
        "league_size": len(standings),
        "strategy": strategy,
        "maximum_moves": maximum_moves,
        "minimum_quality_improvement": minimum_improvement,
        "allowance": allowance,
        "spending_cap": spending_cap,
    }
    if roster is None:
        return {
            **base, "available": False, "roster_identity": None,
            "market_id": None, "evaluations": [], "selected_moves": [],
            "plan": None, "reason": "roster_unavailable",
        }

    market_id = recruitment_market_id(
        season_index=next_season_index, team=team,
    )
    market = generate_recruitment_market(roster, team=team, market_id=market_id)
    players = [
        player for player in roster.get("players") or []
        if isinstance(player, Mapping)
    ]
    evaluations = []
    for candidate in market["candidates"]:
        same_role = sorted(
            (player for player in players if player.get("role") == candidate["role"]),
            key=lambda player: (
                squad_player_quality(player), str(player.get("player_id") or ""),
            ),
        )
        outgoing = same_role[0] if same_role else None
        outgoing_quality = squad_player_quality(outgoing) if outgoing is not None else None
        improvement = (
            round(float(candidate["quality"]) - outgoing_quality, 6)
            if outgoing_quality is not None else None
        )
        evaluations.append({
            "candidate_id": candidate["player_id"],
            "candidate_name": str(candidate.get("name") or candidate["player_id"]),
            "candidate_role": candidate["role"],
            "candidate_quality": candidate["quality"],
            "cost": candidate["recruitment_cost"],
            "outgoing_player_id": (
                str(outgoing.get("player_id") or "") if outgoing is not None else None
            ),
            "outgoing_name": (
                str(outgoing.get("name") or outgoing.get("player_id") or "")
                if outgoing is not None else None
            ),
            "outgoing_quality": (
                round(outgoing_quality, 6) if outgoing_quality is not None else None
            ),
            "quality_improvement": improvement,
            "eligible": bool(
                outgoing is not None
                and improvement is not None
                and improvement >= minimum_improvement
                and int(candidate["recruitment_cost"]) <= spending_cap
            ),
        })

    ranked = sorted(
        (item for item in evaluations if item["eligible"]),
        key=lambda item: (
            -float(item["quality_improvement"]), int(item["cost"]),
            str(item["candidate_id"]),
        ),
    )
    selected = []
    spent = 0
    used_outgoing: set[str] = set()
    for item in ranked:
        outgoing_id = str(item["outgoing_player_id"])
        cost = int(item["cost"])
        if (
            len(selected) >= maximum_moves or spent + cost > spending_cap
            or outgoing_id in used_outgoing
        ):
            continue
        selected.append(item)
        used_outgoing.add(outgoing_id)
        spent += cost
    plan = None
    if selected:
        plan = RecruitmentPlan(
            market_id,
            tuple(RecruitmentMove(
                str(item["candidate_id"]), str(item["outgoing_player_id"]),
            ) for item in selected),
        ).as_dict()
    return {
        **base, "available": True, "roster_identity": roster_identity(roster),
        "market_id": market_id, "evaluations": evaluations,
        "selected_moves": [
            {
                "candidate_id": item["candidate_id"],
                "candidate_name": item["candidate_name"],
                "outgoing_player_id": item["outgoing_player_id"],
                "outgoing_name": item["outgoing_name"],
                "cost": item["cost"],
                "quality_improvement": item["quality_improvement"],
            }
            for item in selected
        ],
        "plan": plan,
        "reason": "strengthening_selected" if selected else "no_eligible_upgrade",
    }


def validate_league_ecosystem(
    registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if registry is None:
        return {"schema_version": 1, "transition_count": 0, "ai_club_count": 0}
    if not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1:
        raise ValueError("invalid league ecosystem registry")
    transitions = registry.get("transitions") or []
    if not isinstance(transitions, list) or len(transitions) > MAX_TRANSITIONS:
        raise ValueError("invalid league ecosystem transitions")
    seen: set[tuple[str, str]] = set()
    ai_club_count = 0
    previous_target = 0
    for event in transitions:
        if not isinstance(event, Mapping) or set(event) != _EVENT_FIELDS:
            raise ValueError("invalid league ecosystem event")
        source = str(event.get("from_season_id") or "")
        target = str(event.get("to_season_id") or "")
        source_match = _SEASON_PATTERN.fullmatch(source)
        target_match = _SEASON_PATTERN.fullmatch(target)
        if (
            event.get("schema_version") != 1 or source_match is None
            or target_match is None
            or int(target_match.group(1)) <= int(source_match.group(1))
            or int(target_match.group(1)) <= previous_target
            or (source, target) in seen
            or not isinstance(event.get("archive_identity"), str)
            or len(event["archive_identity"]) != 64
        ):
            raise ValueError("invalid league ecosystem season identity")
        previous_target = int(target_match.group(1))
        seen.add((source, target))
        teams = event.get("participating_teams")
        clubs = event.get("ai_clubs")
        if (
            not isinstance(teams, list)
            or not all(isinstance(team, str) and team.strip() for team in teams)
            or teams != sorted(teams) or len(set(teams)) != len(teams)
            or not isinstance(clubs, list)
            or not all(
                isinstance(club, Mapping)
                and isinstance(club.get("team"), str)
                for club in clubs
            )
            or [club["team"] for club in clubs]
            != sorted(club["team"] for club in clubs)
        ):
            raise ValueError("invalid league ecosystem club ordering")
        manager_team = event.get("manager_controlled_team")
        if manager_team is not None and manager_team not in teams:
            raise ValueError("invalid league ecosystem manager identity")
        expected_ai = sorted(team for team in teams if team != manager_team)
        if len(clubs) != len(expected_ai):
            raise ValueError("invalid league ecosystem club coverage")
        for team, club in zip(expected_ai, clubs):
            if (
                not isinstance(club, Mapping) or set(club) != _CLUB_FIELDS
                or club.get("team") != team
                or not isinstance(club.get("decision"), Mapping)
                or club["decision"].get("team") != team
                or not isinstance(club.get("settlement_identity"), str)
                or len(club["settlement_identity"]) != 64
                or club.get("squad_transaction_identity") is not None
                and (
                    not isinstance(club["squad_transaction_identity"], str)
                    or len(club["squad_transaction_identity"]) != 64
                )
            ):
                raise ValueError("invalid league ecosystem club evidence")
        ai_club_count += len(clubs)
    return {
        "schema_version": 1, "transition_count": len(transitions),
        "ai_club_count": ai_club_count,
    }


def append_league_transition(
    registry: Mapping[str, Any] | None, event: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(registry or {"schema_version": 1, "transitions": []}))
    validate_league_ecosystem(output)
    transitions = output.setdefault("transitions", [])
    if any(
        item.get("from_season_id") == event.get("from_season_id")
        or item.get("to_season_id") == event.get("to_season_id")
        for item in transitions
    ):
        raise ValueError("league ecosystem transition already exists")
    transitions.append(copy.deepcopy(dict(event)))
    output["transitions"] = transitions[-MAX_TRANSITIONS:]
    validate_league_ecosystem(output)
    return output


def league_ecosystem_view(registry: Mapping[str, Any] | None) -> dict[str, Any]:
    summary = validate_league_ecosystem(registry)
    transitions = list((registry or {}).get("transitions") or [])
    return {
        **summary,
        "available": bool(transitions),
        "latest_transition": copy.deepcopy(transitions[-1]) if transitions else None,
        "claim_boundary": (
            "deterministic bounded AI-club game policy; not a claim about real "
            "club recruitment, finance, or sporting causality"
        ),
    }


__all__ = [
    "ai_club_recruitment_decision", "append_league_transition",
    "league_ecosystem_view", "validate_league_ecosystem",
]
