"""Replayable, non-causal reviews of a frozen sporting directive."""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping, Sequence

from src.product.club_finance import evidence_identity, player_wage_tier
from src.product.season import SeasonPlan
from src.product.sporting_director import (
    SUPPORTED_ROLES, validate_sporting_brief, validate_sporting_choices,
)
from src.simulation.player_development import validate_participation_evidence
from src.simulation.squad_registry import roster_identity, squad_player_quality


REVIEW_SCHEMA_VERSION = 1
MAX_REVIEWS = 256
CLAIM_BOUNDARY = (
    "descriptive review of a frozen fictional sporting policy and later "
    "simulated evidence; no signing, renewal, philosophy, or result causality"
)
FEEDBACK_BOUNDARY = (
    "bounded continuity signals from one descriptive fictional strategy review; "
    "not causal credit, blame, or a performance forecast"
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_KEYS = frozenset({
    "schema_version", "type", "review_id", "season_id", "team",
    "planning_id", "archive_identity", "participation_identity",
    "settlement_identity", "result_roster_identity", "directive",
    "execution", "incoming_usage", "priority_role_delivery", "post_squad",
    "season_result", "evidence_coverage", "learning_signals",
    "claim_boundary",
})


def _usage_band(minutes: float, scheduled_matches: int, coverage: str) -> str:
    if coverage == "unavailable":
        return "evidence_unavailable"
    if coverage == "partial":
        return "evidence_partial"
    share = min(1.0, minutes / max(90.0, scheduled_matches * 90.0))
    if share >= 0.65:
        return "core"
    if share >= 0.30:
        return "rotation"
    if minutes > 0:
        return "fringe"
    return "unused"


def _signals(
    *, directive: Mapping[str, Any], incoming: Sequence[Mapping[str, Any]],
    delivery: Sequence[Mapping[str, Any]], result: Mapping[str, Any],
    coverage: str,
) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for row in delivery:
        if row["incoming_count"] == 0:
            signals.append({
                "id": "priority_unaddressed", "role": row["role"],
                "interpretation": "no incoming player filled this frozen priority role",
            })
    if incoming and coverage != "complete":
        signals.append({
            "id": "incoming_evidence_incomplete", "role": "ALL",
            "interpretation": "available reports cannot establish complete season usage",
        })
    if coverage == "complete":
        for row in incoming:
            if row["utilization_share"] < 0.30:
                signals.append({
                    "id": "incoming_low_usage", "role": row["role"],
                    "interpretation": "incoming player recorded below 30% utilization",
                })
    if result["objective_status"] == "missed":
        signals.append({
            "id": "objective_missed", "role": "ALL",
            "interpretation": "the independently frozen season objective was missed",
        })
    if result["finance_delta"] < 0:
        signals.append({
            "id": "negative_season_balance", "role": "ALL",
            "interpretation": "season revenue was below the bounded wage expense",
        })
    if not signals:
        signals.append({
            "id": "no_review_flag", "role": "ALL",
            "interpretation": "no predefined descriptive review flag was triggered",
        })
    return signals


def build_sporting_review(
    archive: Mapping[str, Any], roster: Mapping[str, Any], *,
    participation: Mapping[str, Any], settlement: Mapping[str, Any],
    free_agent_signings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Join a frozen plan to later evidence without assigning causal credit."""
    validate_participation_evidence(participation)
    plan = SeasonPlan.from_payload(archive.get("plan") or {})
    brief = archive.get("sporting_brief")
    evaluation = archive.get("sporting_evaluation")
    directive = plan.manager_sporting_directive
    team = str(plan.manager_team or "")
    season_id = str(archive.get("season_id") or "")
    if (
        directive is None or not team or not isinstance(brief, Mapping)
        or not isinstance(evaluation, Mapping)
        or participation.get("season_id") != season_id
        or participation.get("team") != team
        or settlement.get("season_id") != season_id
        or settlement.get("team") != team
        or settlement.get("type") != "season_settlement"
        or settlement.get("archive_identity") != evidence_identity(archive)
    ):
        raise ValueError("sporting review source evidence is inconsistent")
    validate_sporting_brief(brief)
    expected_evaluation = validate_sporting_choices(
        directive, brief,
        recruitment=(
            plan.manager_recruitment.as_dict()
            if plan.manager_recruitment is not None else None
        ),
        retention=(
            plan.manager_retention.as_dict()
            if plan.manager_retention is not None else None
        ),
        free_agent=(
            plan.manager_free_agent.as_dict()
            if plan.manager_free_agent is not None else None
        ),
    )
    if dict(evaluation) != expected_evaluation:
        raise ValueError("sporting review policy evaluation mismatch")
    recruitment = archive.get("recruitment_transaction")
    recruitment_moves = (
        recruitment.get("moves") or [] if isinstance(recruitment, Mapping) else []
    )
    incoming_sources: list[tuple[str, Mapping[str, Any]]] = []
    for move in recruitment_moves:
        incoming = move.get("incoming") if isinstance(move, Mapping) else None
        if not isinstance(incoming, Mapping):
            raise ValueError("sporting review recruitment evidence is invalid")
        incoming_sources.append(("recruitment", incoming))
    manager_signings = [
        signing for signing in free_agent_signings
        if isinstance(signing, Mapping) and signing.get("control") == "manager"
        and signing.get("team") == team and signing.get("target_season_id") == season_id
    ]
    if len(manager_signings) != (1 if plan.manager_free_agent is not None else 0):
        raise ValueError("sporting review free-agent evidence is inconsistent")
    for signing in manager_signings:
        incoming = signing.get("incoming")
        if not isinstance(incoming, Mapping):
            raise ValueError("sporting review free-agent player is invalid")
        incoming_sources.append(("free_agent", incoming))

    players = roster.get("players") if isinstance(roster, Mapping) else None
    if not isinstance(players, list) or len(players) < 11:
        raise ValueError("sporting review requires the completed-season roster")
    post_squad = [{
        "player_id": str(player.get("player_id") or ""),
        "role": str(player.get("role") or ""),
        "quality": round(squad_player_quality(player), 6),
        "wage_tier": player_wage_tier(player),
    } for player in players if isinstance(player, Mapping)]
    if len(post_squad) != len(players):
        raise ValueError("sporting review roster contains an invalid player")
    totals = participation["player_totals"]
    scheduled_matches = len(participation["scheduled_fixture_ids"])
    coverage = str(participation["coverage"])
    incoming_usage = []
    for source, player in incoming_sources:
        player_id = str(player.get("player_id") or "")
        total = totals.get(player_id) or {"minutes": 0.0, "appearances": 0}
        minutes = round(float(total["minutes"]), 3)
        share = round(min(
            1.0, minutes / max(90.0, scheduled_matches * 90.0),
        ), 6)
        incoming_usage.append({
            "player_id": player_id,
            "name": str(player.get("name") or player_id),
            "source": source, "role": str(player.get("role") or ""),
            "arrival_quality": round(squad_player_quality(player), 6),
            "minutes": minutes, "appearances": int(total["appearances"]),
            "scheduled_matches": scheduled_matches,
            "utilization_share": share,
            "usage_band": _usage_band(minutes, scheduled_matches, coverage),
        })
    diagnostics = {
        row["role"]: row for row in brief["role_diagnostics"]
    }
    priority_delivery = []
    for role in directive.priority_roles:
        source = diagnostics[role]
        post = [row for row in post_squad if row["role"] == role]
        priority_delivery.append({
            "role": role, "source_depth": source["depth"],
            "source_pending_exits": source["expiring"] + source["retiring"],
            "incoming_count": sum(row["role"] == role for row in incoming_usage),
            "post_depth": len(post),
            "post_mean_quality": (
                round(sum(row["quality"] for row in post) / len(post), 6)
                if post else None
            ),
        })
    profile = archive.get("manager_profile") or {}
    objective = profile.get("objective") or {}
    board = archive.get("board_review") or {}
    season_result = {
        "objective_status": objective.get("status"),
        "position": objective.get("current_position"),
        "points": objective.get("current_points"),
        "board_grade": board.get("grade"),
        "board_confidence_delta": board.get("confidence_delta"),
        "reputation_delta": board.get("reputation_delta"),
        "finance_delta": settlement.get("delta"),
        "revenue_total": settlement.get("revenue_total"),
        "expense_total": settlement.get("expense_total"),
    }
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "type": "sporting_strategy_review", "season_id": season_id,
        "team": team, "planning_id": brief["planning_id"],
        "archive_identity": evidence_identity(archive),
        "participation_identity": evidence_identity(participation),
        "settlement_identity": evidence_identity(settlement),
        "result_roster_identity": roster_identity(roster),
        "directive": directive.as_dict(),
        "execution": {
            "selected_recruitment_moves": evaluation["selected_recruitment_moves"],
            "selected_free_agent": evaluation["selected_free_agent"],
            "selected_renewals": evaluation["selected_renewals"],
            "recruitment_spend": evaluation["recruitment_spend"],
        },
        "incoming_usage": incoming_usage,
        "priority_role_delivery": priority_delivery,
        "post_squad": post_squad,
        "season_result": season_result,
        "evidence_coverage": coverage,
        "learning_signals": _signals(
            directive=directive.as_dict(), incoming=incoming_usage,
            delivery=priority_delivery, result=season_result, coverage=coverage,
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    review["review_id"] = evidence_identity(review)
    validate_sporting_review(review)
    return review


def validate_sporting_review(review: Mapping[str, Any]) -> None:
    if not isinstance(review, Mapping) or set(review) != _REVIEW_KEYS:
        raise ValueError("invalid sporting review")
    frozen = copy.deepcopy(dict(review))
    review_id = frozen.pop("review_id", None)
    directive = review.get("directive")
    execution = review.get("execution")
    incoming = review.get("incoming_usage")
    delivery = review.get("priority_role_delivery")
    squad = review.get("post_squad")
    result = review.get("season_result")
    hashes = (
        review.get("planning_id"), review.get("archive_identity"),
        review.get("participation_identity"), review.get("settlement_identity"),
        review.get("result_roster_identity"), review_id,
    )
    if (
        review.get("schema_version") != 1
        or review.get("type") != "sporting_strategy_review"
        or re.fullmatch(r"season-[0-9]{4}", str(review.get("season_id") or "")) is None
        or not str(review.get("team") or "").strip()
        or any(_HASH.fullmatch(str(value or "")) is None for value in hashes)
        or review_id != evidence_identity(frozen)
        or not isinstance(directive, Mapping)
        or directive.get("planning_id") != review.get("planning_id")
        or not isinstance(execution, Mapping)
        or set(execution) != {
            "selected_recruitment_moves", "selected_free_agent",
            "selected_renewals", "recruitment_spend",
        }
        or not isinstance(incoming, list) or not isinstance(delivery, list)
        or not isinstance(squad, list) or len(squad) < 11
        or not isinstance(result, Mapping)
        or review.get("evidence_coverage") not in {"complete", "partial", "unavailable"}
        or review.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("invalid sporting review identity")
    from src.product.sporting_director import SportingDirective
    parsed = SportingDirective.from_payload(directive)
    integers = [
        execution["selected_recruitment_moves"], execution["selected_renewals"],
        execution["recruitment_spend"],
    ]
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers)
        or not isinstance(execution["selected_free_agent"], bool)
        or execution["selected_recruitment_moves"] > 2
        or execution["selected_renewals"] > 4
        or execution["recruitment_spend"] > 8
    ):
        raise ValueError("invalid sporting review execution")
    squad_ids = []
    for row in squad:
        if not isinstance(row, Mapping) or set(row) != {"player_id", "role", "quality", "wage_tier"}:
            raise ValueError("invalid sporting review squad")
        squad_ids.append(row["player_id"])
        if (
            not str(row["player_id"]).strip() or row["role"] not in SUPPORTED_ROLES
            or isinstance(row["quality"], bool) or not isinstance(row["quality"], (int, float))
            or not math.isfinite(float(row["quality"])) or not 0.15 <= float(row["quality"]) <= 0.92
            or isinstance(row["wage_tier"], bool) or not isinstance(row["wage_tier"], int)
            or not 1 <= row["wage_tier"] <= 4
            or row["wage_tier"] != (
                1 if row["quality"] < 0.62 else 2 if row["quality"] < 0.70
                else 3 if row["quality"] < 0.78 else 4
            )
        ):
            raise ValueError("invalid sporting review squad")
    if len(squad_ids) != len(set(squad_ids)):
        raise ValueError("duplicate sporting review squad identity")
    incoming_ids = []
    coverage = review["evidence_coverage"]
    for row in incoming:
        if not isinstance(row, Mapping) or set(row) != {
            "player_id", "name", "source", "role", "arrival_quality", "minutes",
            "appearances", "scheduled_matches", "utilization_share", "usage_band",
        }:
            raise ValueError("invalid sporting review incoming usage")
        incoming_ids.append(row["player_id"])
        numeric = (row["arrival_quality"], row["minutes"], row["utilization_share"])
        if (
            row["source"] not in {"recruitment", "free_agent"}
            or row["role"] not in SUPPORTED_ROLES
            or row["player_id"] not in squad_ids
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric)
            or not 0.15 <= row["arrival_quality"] <= 0.92
            or not 0 <= row["utilization_share"] <= 1
            or any(isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0 for key in ("appearances", "scheduled_matches"))
            or row["minutes"] < 0
            or row["appearances"] > row["scheduled_matches"]
            or row["utilization_share"] != round(min(1.0, row["minutes"] / max(90.0, row["scheduled_matches"] * 90.0)), 6)
            or row["usage_band"] != _usage_band(row["minutes"], row["scheduled_matches"], coverage)
        ):
            raise ValueError("invalid sporting review incoming usage")
    if len(incoming_ids) != len(set(incoming_ids)):
        raise ValueError("duplicate sporting review incoming identity")
    if len(incoming) != execution["selected_recruitment_moves"] + int(execution["selected_free_agent"]):
        raise ValueError("sporting review incoming execution mismatch")
    expected_delivery = []
    for role in parsed.priority_roles:
        stored = [row for row in delivery if isinstance(row, Mapping) and row.get("role") == role]
        if len(stored) != 1:
            raise ValueError("sporting review priority delivery mismatch")
        row = stored[0]
        post = [player for player in squad if player["role"] == role]
        if (
            set(row) != {
                "role", "source_depth", "source_pending_exits",
                "incoming_count", "post_depth", "post_mean_quality",
            }
            or any(
                isinstance(row[key], bool) or not isinstance(row[key], int)
                or row[key] < 0
                for key in (
                    "source_depth", "source_pending_exits", "incoming_count",
                    "post_depth",
                )
            )
        ):
            raise ValueError("sporting review priority delivery mismatch")
        expected = dict(row)
        expected["incoming_count"] = sum(item["role"] == role for item in incoming)
        expected["post_depth"] = len(post)
        expected["post_mean_quality"] = round(sum(item["quality"] for item in post) / len(post), 6) if post else None
        if row != expected:
            raise ValueError("sporting review priority delivery mismatch")
        expected_delivery.append(row)
    if delivery != expected_delivery:
        raise ValueError("sporting review priority order mismatch")
    required_result = {
        "objective_status", "position", "points", "board_grade",
        "board_confidence_delta", "reputation_delta", "finance_delta",
        "revenue_total", "expense_total",
    }
    if (
        set(result) != required_result
        or result["objective_status"] not in {"achieved", "missed"}
        or result["board_grade"] not in {
            "outstanding", "positive", "mixed", "negative", "critical",
        }
        or any(isinstance(result[key], bool) or not isinstance(result[key], int) for key in (
            "position", "points", "board_confidence_delta", "reputation_delta",
            "finance_delta", "revenue_total", "expense_total",
        ))
        or result["position"] < 1 or result["points"] < 0
        or not -28 <= result["board_confidence_delta"] <= 30
        or not -9 <= result["reputation_delta"] <= 10
        or result["revenue_total"] < 0 or result["expense_total"] < 0
        or result["finance_delta"] != result["revenue_total"] - result["expense_total"]
    ):
        raise ValueError("invalid sporting review season result")
    signals = review.get("learning_signals")
    expected_signals = _signals(
        directive=directive, incoming=incoming, delivery=delivery,
        result=result, coverage=coverage,
    )
    if signals != expected_signals:
        raise ValueError("sporting review learning signals mismatch")


def validate_sporting_review_registry(registry: Mapping[str, Any] | None) -> dict[str, int]:
    if registry is None:
        return {"schema_version": 1, "review_count": 0}
    reviews = registry.get("reviews") if isinstance(registry, Mapping) else None
    if registry.get("schema_version") != 1 or not isinstance(reviews, list) or len(reviews) > MAX_REVIEWS:
        raise ValueError("invalid sporting review registry")
    keys = set()
    for review in reviews:
        validate_sporting_review(review)
        key = (review["season_id"], review["team"])
        if key in keys:
            raise ValueError("duplicate sporting strategy review")
        keys.add(key)
    return {"schema_version": 1, "review_count": len(reviews)}


def append_sporting_review(
    registry: Mapping[str, Any] | None, review: Mapping[str, Any],
) -> dict[str, Any]:
    validate_sporting_review(review)
    output = copy.deepcopy(dict(registry)) if registry is not None else {
        "schema_version": 1, "reviews": [],
    }
    validate_sporting_review_registry(output)
    if any(
        row["season_id"] == review["season_id"] and row["team"] == review["team"]
        for row in output["reviews"]
    ):
        raise ValueError("duplicate sporting strategy review")
    output["reviews"].append(copy.deepcopy(dict(review)))
    validate_sporting_review_registry(output)
    return output


def sporting_review_view(registry: Mapping[str, Any] | None) -> dict[str, Any]:
    summary = validate_sporting_review_registry(registry)
    reviews = (registry or {}).get("reviews") or []
    public = [{
        "review_id": row["review_id"], "season_id": row["season_id"],
        "team": row["team"], "directive": copy.deepcopy(row["directive"]),
        "execution": copy.deepcopy(row["execution"]),
        "incoming_usage": copy.deepcopy(row["incoming_usage"]),
        "priority_role_delivery": copy.deepcopy(row["priority_role_delivery"]),
        "season_result": copy.deepcopy(row["season_result"]),
        "evidence_coverage": row["evidence_coverage"],
        "learning_signals": copy.deepcopy(row["learning_signals"]),
        "claim_boundary": row["claim_boundary"],
    } for row in reviews]
    return {
        **summary, "available": bool(public),
        "latest_review": public[-1] if public else None,
        "reviews": list(reversed(public[-12:])),
    }


def sporting_review_feedback(review: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal prior-review evidence allowed into the next plan."""
    validate_sporting_review(review)
    signals = review["learning_signals"]
    return {
        "schema_version": 1, "review_id": review["review_id"],
        "season_id": review["season_id"], "team": review["team"],
        "evidence_coverage": review["evidence_coverage"],
        "priority_unaddressed_roles": sorted({
            row["role"] for row in signals
            if row["id"] == "priority_unaddressed"
        }),
        "incoming_low_usage_roles": sorted({
            row["role"] for row in signals if row["id"] == "incoming_low_usage"
        }),
        "objective_status": review["season_result"]["objective_status"],
        "finance_delta": review["season_result"]["finance_delta"],
        "claim_boundary": FEEDBACK_BOUNDARY,
    }


__all__ = [
    "append_sporting_review", "build_sporting_review", "sporting_review_view",
    "sporting_review_feedback", "validate_sporting_review",
    "validate_sporting_review_registry",
]
