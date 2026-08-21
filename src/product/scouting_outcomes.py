"""Replayable post-season outcomes for fictional free-agent decisions."""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

from src.product.club_finance import evidence_identity, player_wage_tier
from src.simulation.player_market import market_player_quality
from src.simulation.player_development import validate_participation_evidence
from src.simulation.scouting import scouting_observation


OUTCOME_SCHEMA_VERSION = 1
MAX_OUTCOMES = 4096
CLAIM_BOUNDARY = (
    "descriptive post-season evidence for a fictional signing; usage, "
    "points, and position do not identify transfer causality"
)
_OUTCOME_KEYS = {
    "schema_version", "type", "season_id", "team", "player_id",
    "player_name", "control", "market_id", "origin_team",
    "signing_identity", "archive_identity", "participation_identity",
    "settlement_identity", "observation_source", "observation", "realized",
    "claim_boundary",
}
_OBSERVATION_KEYS = {
    "schema_version", "level", "estimated_quality", "quality_low",
    "quality_high", "interval_width", "coverage_contract", "claim_boundary",
}
_REALIZED_KEYS = {
    "actual_quality_at_signing", "absolute_estimation_error",
    "interval_contained_truth", "minutes", "appearances",
    "scheduled_matches", "utilization_share", "usage_band",
    "evidence_coverage", "wage_tier", "team_wage_expense",
    "team_position", "team_points",
}


def _observation_is_valid(observation: Any) -> bool:
    if not isinstance(observation, Mapping):
        return False
    values = [
        observation.get("estimated_quality"), observation.get("quality_low"),
        observation.get("quality_high"), observation.get("interval_width"),
    ]
    return (
        set(observation) == _OBSERVATION_KEYS
        and observation.get("schema_version") == 1
        and observation.get("level") in {"baseline", "scouted"}
        and observation.get("coverage_contract")
        == "contains_simulated_truth_by_construction"
        and all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) for value in values
        )
        and 0.15 <= observation["quality_low"]
        <= observation["estimated_quality"]
        <= observation["quality_high"] <= 0.92
        and round(
            observation["quality_high"] - observation["quality_low"], 6,
        ) == observation["interval_width"]
    )


def resolve_signing_observation(
    signing: Mapping[str, Any], transition: Mapping[str, Any],
    scouting_registry: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Recover exactly what the signing club knew when it made the choice."""
    if signing not in (transition.get("signings") or []):
        raise ValueError("scouting outcome signing is absent from market transition")
    market_id = str(signing.get("market_id") or "")
    team = str(signing.get("team") or "")
    player_id = str(signing.get("free_agent_id") or "")
    entry = signing.get("free_agent_entry")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("player"), Mapping):
        raise ValueError("scouting outcome free-agent entry is unavailable")
    true_quality = market_player_quality(entry["player"])
    report = None
    source = "baseline_observation"
    if signing.get("control") == "manager":
        matches = [
            item for item in (scouting_registry or {}).get("reports") or []
            if item.get("market_id") == market_id and item.get("team") == team
            and item.get("player_id") == player_id
        ]
        if len(matches) > 1:
            raise ValueError("duplicate manager scouting outcome report")
        if matches:
            report, source = matches[0], "manager_scouting_report"
    elif signing.get("control") == "ai":
        decisions = [
            item for item in transition.get("ai_decisions") or []
            if item.get("team") == team and item.get("market_id") == market_id
        ]
        if len(decisions) != 1:
            raise ValueError("AI scouting outcome decision is unavailable")
        matches = [
            item for item in decisions[0].get("scouting_reports") or []
            if item.get("player_id") == player_id
        ]
        if len(matches) > 1:
            raise ValueError("duplicate AI scouting outcome report")
        if matches:
            report, source = matches[0], "ai_scouting_report"
    else:
        raise ValueError("invalid scouting outcome control")
    if report is None:
        return scouting_observation(
            market_id=market_id, team=team, player_id=player_id,
            true_quality=true_quality, level="baseline",
        ), source
    expected = scouting_observation(
        market_id=market_id, team=team, player_id=player_id,
        true_quality=true_quality, level="scouted",
    )
    if report.get("observation") != expected:
        raise ValueError("scouting outcome report replay mismatch")
    return copy.deepcopy(expected), source


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


def build_scouting_outcome(
    signing: Mapping[str, Any], *, observation: Mapping[str, Any],
    observation_source: str, archive: Mapping[str, Any],
    participation: Mapping[str, Any], settlement: Mapping[str, Any],
) -> dict[str, Any]:
    validate_participation_evidence(participation)
    season_id = str(archive.get("season_id") or "")
    team = str(signing.get("team") or "")
    player_id = str(signing.get("free_agent_id") or "")
    if (
        not season_id or signing.get("target_season_id") != season_id
        or participation.get("season_id") != season_id
        or participation.get("team") != team
        or settlement.get("type") != "season_settlement"
        or settlement.get("season_id") != season_id
        or settlement.get("team") != team
        or settlement.get("archive_identity") != evidence_identity(archive)
        or observation_source not in {
            "baseline_observation", "manager_scouting_report",
            "ai_scouting_report",
        }
        or not _observation_is_valid(observation)
    ):
        raise ValueError("invalid scouting outcome source evidence")
    incoming = signing.get("incoming")
    if not isinstance(incoming, Mapping) or incoming.get("player_id") != player_id:
        raise ValueError("invalid scouting outcome incoming player")
    actual_quality = round(market_player_quality(incoming), 6)
    if not observation["quality_low"] <= actual_quality <= observation["quality_high"]:
        raise ValueError("scouting outcome interval does not cover simulated truth")
    totals = participation["player_totals"].get(player_id) or {
        "minutes": 0.0, "appearances": 0,
    }
    minutes = round(float(totals["minutes"]), 3)
    appearances = int(totals["appearances"])
    scheduled_matches = len(participation["scheduled_fixture_ids"])
    utilization = round(min(
        1.0, minutes / max(90.0, scheduled_matches * 90.0),
    ), 6)
    standings = [
        row for row in archive.get("final_standings") or []
        if isinstance(row, Mapping) and row.get("team") == team
    ]
    if len(standings) != 1:
        raise ValueError("scouting outcome standings evidence is unavailable")
    outcome = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "type": "free_agent_season_outcome", "season_id": season_id,
        "team": team, "player_id": player_id,
        "player_name": str(incoming.get("name") or player_id),
        "control": signing["control"], "market_id": signing["market_id"],
        "origin_team": signing["free_agent_entry"]["origin_team"],
        "signing_identity": evidence_identity(signing),
        "archive_identity": evidence_identity(archive),
        "participation_identity": evidence_identity(participation),
        "settlement_identity": evidence_identity(settlement),
        "observation_source": observation_source,
        "observation": copy.deepcopy(dict(observation)),
        "realized": {
            "actual_quality_at_signing": actual_quality,
            "absolute_estimation_error": round(abs(
                float(observation["estimated_quality"]) - actual_quality,
            ), 6),
            "interval_contained_truth": True,
            "minutes": minutes, "appearances": appearances,
            "scheduled_matches": scheduled_matches,
            "utilization_share": utilization,
            "usage_band": _usage_band(
                minutes, scheduled_matches, participation["coverage"],
            ),
            "evidence_coverage": participation["coverage"],
            "wage_tier": player_wage_tier(incoming),
            "team_wage_expense": settlement["expense_total"],
            "team_position": standings[0]["position"],
            "team_points": standings[0]["points"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    validate_scouting_outcome(outcome)
    return outcome


def validate_scouting_outcome(outcome: Mapping[str, Any]) -> None:
    realized = outcome.get("realized") if isinstance(outcome, Mapping) else None
    observation = outcome.get("observation") if isinstance(outcome, Mapping) else None
    hashes = (
        outcome.get("signing_identity"), outcome.get("archive_identity"),
        outcome.get("participation_identity"), outcome.get("settlement_identity"),
    ) if isinstance(outcome, Mapping) else ()
    if (
        not isinstance(outcome, Mapping) or outcome.get("schema_version") != 1
        or set(outcome) != _OUTCOME_KEYS
        or outcome.get("type") != "free_agent_season_outcome"
        or re.fullmatch(r"season-[0-9]{4}", str(outcome.get("season_id") or "")) is None
        or not str(outcome.get("market_id") or "").startswith("free-market-")
        or not all(str(outcome.get(key) or "").strip() for key in (
            "season_id", "team", "player_id", "player_name", "market_id",
            "origin_team",
        ))
        or outcome.get("control") not in {"manager", "ai"}
        or len(str(outcome.get("team") or "")) > 96
        or len(str(outcome.get("origin_team") or "")) > 96
        or len(str(outcome.get("player_id") or "")) > 128
        or len(str(outcome.get("player_name") or "")) > 160
        or outcome.get("observation_source") not in {
            "baseline_observation", "manager_scouting_report",
            "ai_scouting_report",
        }
        or not _observation_is_valid(observation)
        or not isinstance(realized, Mapping)
        or set(realized) != _REALIZED_KEYS
        or outcome.get("claim_boundary") != CLAIM_BOUNDARY
        or any(
            not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        )
    ):
        raise ValueError("invalid scouting outcome")
    if (
        (outcome["observation_source"] == "baseline_observation")
        != (observation["level"] == "baseline")
        or outcome["observation_source"] == "manager_scouting_report"
        and outcome["control"] != "manager"
        or outcome["observation_source"] == "ai_scouting_report"
        and outcome["control"] != "ai"
    ):
        raise ValueError("scouting outcome observation source mismatch")
    numeric = [
        realized.get("actual_quality_at_signing"),
        realized.get("absolute_estimation_error"), realized.get("minutes"),
        realized.get("utilization_share"),
    ]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) for value in numeric
        )
        or not 0.15 <= realized["actual_quality_at_signing"] <= 0.92
        or realized["absolute_estimation_error"] != round(abs(
            observation["estimated_quality"]
            - realized["actual_quality_at_signing"],
        ), 6)
        or realized.get("interval_contained_truth") is not True
        or not observation["quality_low"]
        <= realized["actual_quality_at_signing"] <= observation["quality_high"]
        or realized["minutes"] < 0
        or isinstance(realized.get("appearances"), bool)
        or not isinstance(realized.get("appearances"), int)
        or realized["appearances"] < 0
        or isinstance(realized.get("scheduled_matches"), bool)
        or not isinstance(realized.get("scheduled_matches"), int)
        or realized["scheduled_matches"] < 1
        or not 0 <= realized["utilization_share"] <= 1
        or realized["utilization_share"] != round(min(
            1.0, realized["minutes"] / (realized["scheduled_matches"] * 90.0),
        ), 6)
        or realized.get("evidence_coverage") not in {
            "complete", "partial", "unavailable",
        }
        or realized.get("usage_band") != _usage_band(
            realized["minutes"], realized["scheduled_matches"],
            realized["evidence_coverage"],
        )
        or isinstance(realized.get("wage_tier"), bool)
        or not isinstance(realized.get("wage_tier"), int)
        or not 1 <= realized["wage_tier"] <= 4
        or isinstance(realized.get("team_wage_expense"), bool)
        or not isinstance(realized.get("team_wage_expense"), int)
        or realized["team_wage_expense"] < 1
        or isinstance(realized.get("team_position"), bool)
        or not isinstance(realized.get("team_position"), int)
        or realized["team_position"] < 1
        or isinstance(realized.get("team_points"), bool)
        or not isinstance(realized.get("team_points"), int)
        or realized["team_points"] < 0
    ):
        raise ValueError("scouting outcome replay mismatch")


def validate_scouting_outcome_registry(
    registry: Mapping[str, Any] | None,
) -> dict[str, int]:
    if registry is None:
        return {"schema_version": 1, "outcome_count": 0, "season_count": 0}
    records = registry.get("outcomes") if isinstance(registry, Mapping) else None
    if (
        not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1
        or not isinstance(records, list) or len(records) > MAX_OUTCOMES
    ):
        raise ValueError("invalid scouting outcome registry")
    seen = set()
    seasons = set()
    for outcome in records:
        validate_scouting_outcome(outcome)
        key = (outcome["season_id"], outcome["team"], outcome["player_id"])
        if key in seen:
            raise ValueError("duplicate scouting outcome")
        seen.add(key)
        seasons.add(outcome["season_id"])
    return {
        "schema_version": 1, "outcome_count": len(records),
        "season_count": len(seasons),
    }


def append_scouting_outcome(
    registry: Mapping[str, Any] | None, outcome: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(registry or {"schema_version": 1, "outcomes": []}))
    validate_scouting_outcome_registry(output)
    validate_scouting_outcome(outcome)
    output.setdefault("outcomes", []).append(copy.deepcopy(dict(outcome)))
    validate_scouting_outcome_registry(output)
    return output


def scouting_outcome_view(
    registry: Mapping[str, Any] | None, *, manager_team: str | None,
) -> dict[str, Any]:
    summary = validate_scouting_outcome_registry(registry)
    records = (registry or {}).get("outcomes") or []
    manager_records = [
        {
            "season_id": record["season_id"], "team": record["team"],
            "player_id": record["player_id"],
            "player_name": record["player_name"],
            "origin_team": record["origin_team"],
            "observation_source": record["observation_source"],
            "observation": copy.deepcopy(record["observation"]),
            "realized": copy.deepcopy(record["realized"]),
            "claim_boundary": record["claim_boundary"],
        }
        for record in records if record.get("control") == "manager"
    ]
    coverage = {key: 0 for key in ("complete", "partial", "unavailable")}
    usage = {key: 0 for key in (
        "core", "rotation", "fringe", "unused", "evidence_partial",
        "evidence_unavailable",
    )}
    for record in records:
        coverage[record["realized"]["evidence_coverage"]] += 1
        usage[record["realized"]["usage_band"]] += 1
    return {
        **summary, "manager_outcomes": manager_records[-24:],
        "manager_outcome_count": len(manager_records),
        "ai_outcome_count": sum(record.get("control") == "ai" for record in records),
        "coverage_counts": coverage, "usage_counts": usage,
        "claim_boundary": (
            "manager signing details and league aggregates from simulated seasons; "
            "AI hidden quality remains private and no causal transfer claim is made"
        ),
    }


__all__ = [
    "append_scouting_outcome", "build_scouting_outcome",
    "resolve_signing_observation", "scouting_outcome_view",
    "validate_scouting_outcome", "validate_scouting_outcome_registry",
]
