"""Unified, evidence-bound sporting plan for the next simulated season."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.product.club_finance import evidence_identity
from src.simulation.squad_registry import roster_identity


SPORTING_SCHEMA_VERSION = 1
PHILOSOPHIES = (
    "balanced", "win_now", "youth_pathway", "financial_control",
)
RISK_LEVELS = ("low", "balanced", "high")
SUPPORTED_ROLES = (
    "GK", "RB", "CB", "LB", "DM", "CM", "AM", "RW", "ST", "LW",
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_RECOMMENDATION_REASON = (
    "bounded roster depth, pending exits, available fictional cash, observed "
    "scouting history, and any prior descriptive strategy review"
)
_FEEDBACK_BOUNDARY = (
    "bounded continuity signals from one descriptive fictional strategy review; "
    "not causal credit, blame, or a performance forecast"
)
_CLAIM_BOUNDARY = (
    "deterministic fictional squad-planning diagnosis and constraints; "
    "not real sporting, employment, valuation, or performance advice"
)
_BRIEF_KEYS = frozenset({
    "schema_version", "planning_id", "team", "source_season_id",
    "target_season_id", "source_roster_identity", "recruitment_market_id",
    "lifecycle_cycle_id", "free_agent_market_id", "club_balance",
    "available_budget", "scouting_budget", "squad_size", "wage_tier_total",
    "role_diagnostics", "recruitment_candidates", "free_agent_candidates",
    "outgoing_players", "expiring_player_ids", "retiring_player_ids",
    "manager_outcome_count", "mean_historical_scouting_error",
    "previous_strategy_review", "recommendation", "policy_contract",
    "claim_boundary",
})


@dataclass(frozen=True)
class SportingDirective:
    planning_id: str
    philosophy: str = "balanced"
    risk_level: str = "balanced"
    priority_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _HASH.fullmatch(str(self.planning_id or "")) is None:
            raise ValueError("invalid sporting planning identity")
        if self.philosophy not in PHILOSOPHIES:
            raise ValueError("unsupported sporting philosophy")
        if self.risk_level not in RISK_LEVELS:
            raise ValueError("unsupported sporting risk level")
        roles = tuple(str(role or "").strip().upper() for role in self.priority_roles)
        if (
            len(roles) > 3 or len(roles) != len(set(roles))
            or any(role not in SUPPORTED_ROLES for role in roles)
        ):
            raise ValueError("sporting priority roles must be unique supported roles")
        object.__setattr__(self, "priority_roles", roles)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SportingDirective":
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("sporting directive must be a versioned object")
        roles = payload.get("priority_roles", [])
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
            raise ValueError("sporting priority roles must be an array")
        return cls(
            planning_id=str(payload.get("planning_id") or ""),
            philosophy=str(payload.get("philosophy") or "balanced"),
            risk_level=str(payload.get("risk_level") or "balanced"),
            priority_roles=tuple(roles),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SPORTING_SCHEMA_VERSION,
            "planning_id": self.planning_id,
            "philosophy": self.philosophy,
            "risk_level": self.risk_level,
            "priority_roles": list(self.priority_roles),
        }


def _diagnose_roles(
    outgoing: Sequence[Mapping[str, Any]], *, expiring: Sequence[str],
    retiring: Sequence[str], recruitment_candidates: Sequence[Mapping[str, Any]],
    free_candidates: Sequence[Mapping[str, Any]],
    continuity_roles: Sequence[str] = (),
) -> list[dict[str, Any]]:
    expiring_ids = set(expiring)
    retiring_ids = set(retiring)
    rows = []
    for role in SUPPORTED_ROLES:
        role_players = [row for row in outgoing if row.get("role") == role]
        qualities = [float(row["quality"]) for row in role_players]
        ages = [
            int(row["age"]) for row in role_players
            if isinstance(row.get("age"), int)
            and not isinstance(row.get("age"), bool)
        ]
        depth = len(role_players)
        expiring_count = sum(
            str(row.get("player_id") or "") in expiring_ids
            for row in role_players
        )
        retiring_count = sum(
            str(row.get("player_id") or "") in retiring_ids
            for row in role_players
        )
        exit_count = expiring_count + retiring_count
        mean_quality = round(sum(qualities) / len(qualities), 6) if qualities else None
        continuity_signal = role in continuity_roles
        need_score = round(
            (2 if role == "GK" and depth == 0 else max(0, 2 - depth)) * 2.0
            + exit_count * 1.5
            + (max(0.0, 0.64 - mean_quality) * 8.0 if mean_quality is not None else 2.0),
            6,
        ) + (1.0 if continuity_signal else 0.0)
        need_score = round(
            need_score,
            6,
        )
        rows.append({
            "role": role, "depth": depth, "mean_quality": mean_quality,
            "mean_age": round(sum(ages) / len(ages), 3) if ages else None,
            "expiring": expiring_count, "retiring": retiring_count,
            "recruitment_options": sum(
                candidate.get("role") == role for candidate in recruitment_candidates
            ),
            "free_agent_options": sum(
                candidate.get("role") == role for candidate in free_candidates
            ),
            "continuity_signal": continuity_signal,
            "need_score": need_score,
        })
    return rows


def build_sporting_brief(
    *, team: str, source_season_id: str, target_season_id: str,
    roster: Mapping[str, Any], recruitment_market: Mapping[str, Any],
    lifecycle_preview: Mapping[str, Any], free_agent_market: Mapping[str, Any],
    scouting_outcomes: Mapping[str, Any],
    previous_strategy_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    players = [
        player for player in roster.get("players") or []
        if isinstance(player, Mapping)
    ] if isinstance(roster, Mapping) else []
    recruitment_candidates = [
        candidate for candidate in recruitment_market.get("candidates") or []
        if isinstance(candidate, Mapping)
    ]
    free_candidates = [
        candidate for candidate in free_agent_market.get("candidates") or []
        if isinstance(candidate, Mapping)
    ]
    if (
        not str(team).strip() or not players
        or lifecycle_preview.get("team") != team
        or lifecycle_preview.get("source_season_id") != source_season_id
        or lifecycle_preview.get("target_season_id") != target_season_id
        or recruitment_market.get("team") != team
        or free_agent_market.get("team") != team
    ):
        raise ValueError("sporting brief source evidence is inconsistent")
    outgoing = {
        str(row.get("player_id") or ""): row
        for row in recruitment_market.get("outgoing_players") or []
        if isinstance(row, Mapping)
    }
    if set(outgoing) != {
        str(player.get("player_id") or "") for player in players
    }:
        raise ValueError("sporting brief roster projection mismatch")
    expiring_ids = copy.deepcopy(
        lifecycle_preview.get("expiring_player_ids") or [],
    )
    retiring_ids = copy.deepcopy(
        lifecycle_preview.get("retiring_player_ids") or [],
    )
    continuity_roles = sorted(set(
        list((previous_strategy_review or {}).get(
            "priority_unaddressed_roles", [],
        ))
        + list((previous_strategy_review or {}).get(
            "incoming_low_usage_roles", [],
        ))
    ))
    role_rows = _diagnose_roles(
        list(outgoing.values()), expiring=expiring_ids, retiring=retiring_ids,
        recruitment_candidates=recruitment_candidates,
        free_candidates=free_candidates, continuity_roles=continuity_roles,
    )
    recommended_roles = [
        row["role"] for row in sorted(
            role_rows, key=lambda row: (-row["need_score"], row["role"]),
        )
        if row["recruitment_options"] or row["free_agent_options"]
    ][:3]
    ages = [
        int(row["age"]) for row in outgoing.values()
        if isinstance(row.get("age"), int)
        and not isinstance(row.get("age"), bool)
    ]
    balance = int(recruitment_market.get("club_balance", 0))
    allowance = int(recruitment_market.get("available_budget", 0))
    recommended_philosophy = (
        "financial_control" if allowance <= 2
        else "youth_pathway" if ages and sum(ages) / len(ages) >= 29
        else "win_now" if sum(
            row["expiring"] + row["retiring"] for row in role_rows
        ) == 0 and allowance >= 6
        else "balanced"
    )
    outcome_rows = scouting_outcomes.get("manager_outcomes") or []
    errors = [
        float(row["realized"]["absolute_estimation_error"])
        for row in outcome_rows if isinstance(row, Mapping)
        and isinstance(row.get("realized"), Mapping)
    ]
    evidence = {
        "schema_version": SPORTING_SCHEMA_VERSION, "team": team,
        "source_season_id": source_season_id,
        "target_season_id": target_season_id,
        "source_roster_identity": roster_identity(roster),
        "recruitment_market_id": recruitment_market.get("market_id"),
        "lifecycle_cycle_id": lifecycle_preview.get("cycle_id"),
        "free_agent_market_id": free_agent_market.get("market_id"),
        "club_balance": balance, "available_budget": allowance,
        "scouting_budget": copy.deepcopy(free_agent_market.get("scouting_budget")),
        "squad_size": len(players),
        "wage_tier_total": sum(int(row["wage_tier"]) for row in outgoing.values()),
        "role_diagnostics": role_rows,
        "recruitment_candidates": copy.deepcopy(recruitment_candidates),
        "free_agent_candidates": copy.deepcopy(free_candidates),
        "outgoing_players": copy.deepcopy(list(outgoing.values())),
        "expiring_player_ids": expiring_ids,
        "retiring_player_ids": retiring_ids,
        "manager_outcome_count": int(
            scouting_outcomes.get("manager_outcome_count", 0),
        ),
        "mean_historical_scouting_error": (
            round(sum(errors) / len(errors), 6) if errors else None
        ),
        "previous_strategy_review": (
            copy.deepcopy(dict(previous_strategy_review))
            if previous_strategy_review is not None else None
        ),
    }
    brief = {
        **evidence,
        "recommendation": {
            "priority_roles": recommended_roles,
            "philosophy": recommended_philosophy,
            "risk_level": "balanced",
            "reason": _RECOMMENDATION_REASON,
        },
        "policy_contract": {
            "priority_roles_bind_all_incoming_players": True,
            "low_risk_requires_scouted_lower_bound_upgrade": True,
            "balanced_risk_requires_scouted_free_agent": True,
            "high_risk_allows_baseline_free_agent": True,
            "win_now_requires_estimated_upgrade": 0.015,
            "youth_pathway_max_incoming_age": 24,
            "financial_control_recruitment_cap": min(4, allowance),
            "financial_control_max_renewals": 2,
        },
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    brief["planning_id"] = evidence_identity(brief)
    validate_sporting_brief(brief)
    return brief


def validate_sporting_brief(brief: Mapping[str, Any]) -> None:
    if not isinstance(brief, Mapping):
        raise ValueError("invalid sporting brief")
    frozen = copy.deepcopy(dict(brief))
    planning_id = frozen.pop("planning_id", None)
    rows = brief.get("role_diagnostics")
    outgoing = brief.get("outgoing_players")
    recruitment = brief.get("recruitment_candidates")
    free_agents = brief.get("free_agent_candidates")
    expiring = brief.get("expiring_player_ids")
    retiring = brief.get("retiring_player_ids")
    source_match = re.fullmatch(
        r"season-([0-9]{4})", str(brief.get("source_season_id") or ""),
    )
    target_match = re.fullmatch(
        r"season-([0-9]{4})", str(brief.get("target_season_id") or ""),
    )
    recommendation = brief.get("recommendation")
    policy = brief.get("policy_contract")
    scouting_budget = brief.get("scouting_budget")
    previous_review = brief.get("previous_strategy_review")
    if (
        set(brief) != _BRIEF_KEYS
        or brief.get("schema_version") != 1
        or _HASH.fullmatch(str(planning_id or "")) is None
        or planning_id != evidence_identity(frozen)
        or source_match is None or target_match is None
        or int(target_match.group(1)) != int(source_match.group(1)) + 1
        or not str(brief.get("team") or "").strip()
        or _HASH.fullmatch(str(brief.get("source_roster_identity") or "")) is None
        or any(not str(brief.get(key) or "").strip() for key in (
            "recruitment_market_id", "lifecycle_cycle_id", "free_agent_market_id",
        ))
        or not isinstance(rows, list)
        or [row.get("role") for row in rows if isinstance(row, Mapping)]
        != list(SUPPORTED_ROLES)
        or not isinstance(recommendation, Mapping)
        or set(recommendation) != {"priority_roles", "philosophy", "risk_level", "reason"}
        or not isinstance(policy, Mapping)
        or not isinstance(outgoing, list) or not isinstance(recruitment, list)
        or not isinstance(free_agents, list)
        or not isinstance(expiring, list) or expiring != sorted(set(expiring))
        or not isinstance(retiring, list) or retiring != sorted(set(retiring))
        or set(expiring) & set(retiring)
    ):
        raise ValueError("invalid sporting brief identity")
    integer_evidence = (
        brief.get("club_balance"), brief.get("available_budget"),
        brief.get("squad_size"), brief.get("wage_tier_total"),
        brief.get("manager_outcome_count"),
    )
    historical_error = brief.get("mean_historical_scouting_error")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in integer_evidence)
        or brief["club_balance"] < 0
        or not 0 <= brief["available_budget"] <= 8
        or brief["squad_size"] < 11 or brief["wage_tier_total"] < 0
        or brief["manager_outcome_count"] < 0
        or not isinstance(scouting_budget, Mapping)
        or set(scouting_budget) != {"limit", "used", "remaining"}
        or any(
            isinstance(scouting_budget.get(key), bool)
            or not isinstance(scouting_budget.get(key), int)
            or scouting_budget[key] < 0
            for key in ("limit", "used", "remaining")
        )
        or scouting_budget["used"] > scouting_budget["limit"]
        or scouting_budget["remaining"] != scouting_budget["limit"] - scouting_budget["used"]
        or (
            historical_error is not None
            and (
                isinstance(historical_error, bool)
                or not isinstance(historical_error, (int, float))
                or not math.isfinite(float(historical_error))
                or not 0 <= float(historical_error) <= 0.77
            )
        )
        or brief.get("claim_boundary") != _CLAIM_BOUNDARY
    ):
        raise ValueError("invalid sporting brief evidence summary")
    if previous_review is not None:
        expected_feedback_keys = {
            "schema_version", "review_id", "season_id", "team",
            "evidence_coverage", "priority_unaddressed_roles",
            "incoming_low_usage_roles", "objective_status", "finance_delta",
            "claim_boundary",
        }
        unaddressed = (
            previous_review.get("priority_unaddressed_roles")
            if isinstance(previous_review, Mapping) else None
        )
        low_usage = (
            previous_review.get("incoming_low_usage_roles")
            if isinstance(previous_review, Mapping) else None
        )
        if (
            not isinstance(previous_review, Mapping)
            or set(previous_review) != expected_feedback_keys
            or previous_review.get("schema_version") != 1
            or _HASH.fullmatch(str(previous_review.get("review_id") or "")) is None
            or previous_review.get("season_id") != brief.get("source_season_id")
            or previous_review.get("team") != brief.get("team")
            or previous_review.get("evidence_coverage") not in {
                "complete", "partial", "unavailable",
            }
            or not isinstance(unaddressed, list)
            or unaddressed != sorted(set(unaddressed))
            or any(role not in SUPPORTED_ROLES for role in unaddressed)
            or not isinstance(low_usage, list)
            or low_usage != sorted(set(low_usage))
            or any(role not in SUPPORTED_ROLES for role in low_usage)
            or low_usage and previous_review.get("evidence_coverage") != "complete"
            or previous_review.get("objective_status") not in {"achieved", "missed"}
            or isinstance(previous_review.get("finance_delta"), bool)
            or not isinstance(previous_review.get("finance_delta"), int)
            or previous_review.get("claim_boundary") != _FEEDBACK_BOUNDARY
        ):
            raise ValueError("invalid previous sporting strategy review")
    outgoing_ids = [
        str(row.get("player_id") or "") for row in outgoing
        if isinstance(row, Mapping)
    ]
    if (
        len(outgoing_ids) != len(outgoing)
        or len(outgoing_ids) != len(set(outgoing_ids))
        or any(
            not player_id or row.get("role") not in SUPPORTED_ROLES
            or isinstance(row.get("quality"), bool)
            or not isinstance(row.get("quality"), (int, float))
            or not math.isfinite(float(row["quality"]))
            or not 0.15 <= float(row["quality"]) <= 0.92
            or isinstance(row.get("wage_tier"), bool)
            or not isinstance(row.get("wage_tier"), int)
            or not 1 <= row["wage_tier"] <= 4
            or (
                row.get("age") is not None
                and (
                    isinstance(row.get("age"), bool)
                    or not isinstance(row.get("age"), int)
                    or not 15 <= row["age"] <= 70
                )
            )
            for player_id, row in zip(outgoing_ids, outgoing)
        )
        or not (set(expiring) | set(retiring)) <= set(outgoing_ids)
        or brief.get("squad_size") != len(outgoing)
        or brief.get("wage_tier_total") != sum(row["wage_tier"] for row in outgoing)
    ):
        raise ValueError("invalid sporting roster evidence")
    recruitment_ids = []
    for candidate in recruitment:
        if not isinstance(candidate, Mapping):
            raise ValueError("invalid sporting recruitment evidence")
        player_id = str(candidate.get("player_id") or "")
        recruitment_ids.append(player_id)
        if (
            not player_id or candidate.get("role") not in SUPPORTED_ROLES
            or isinstance(candidate.get("age"), bool)
            or not isinstance(candidate.get("age"), int)
            or not 15 <= candidate["age"] <= 60
            or isinstance(candidate.get("quality"), bool)
            or not isinstance(candidate.get("quality"), (int, float))
            or not math.isfinite(float(candidate["quality"]))
            or not 0.15 <= float(candidate["quality"]) <= 0.92
            or isinstance(candidate.get("recruitment_cost"), bool)
            or not isinstance(candidate.get("recruitment_cost"), int)
            or not 0 <= candidate["recruitment_cost"] <= 8
        ):
            raise ValueError("invalid sporting recruitment evidence")
    if len(recruitment_ids) != len(set(recruitment_ids)):
        raise ValueError("invalid sporting recruitment evidence")
    free_agent_ids = []
    for candidate in free_agents:
        if not isinstance(candidate, Mapping):
            raise ValueError("invalid sporting free-agent evidence")
        player_id = str(candidate.get("player_id") or "")
        free_agent_ids.append(player_id)
        observation = candidate.get("observation")
        values = [
            observation.get(key) if isinstance(observation, Mapping) else None
            for key in ("estimated_quality", "quality_low", "quality_high")
        ]
        if (
            not player_id or candidate.get("role") not in SUPPORTED_ROLES
            or not isinstance(candidate.get("scouted"), bool)
            or (
                candidate.get("age") is not None
                and (
                    isinstance(candidate.get("age"), bool)
                    or not isinstance(candidate.get("age"), int)
                    or not 15 <= candidate["age"] <= 70
                )
            )
            or not isinstance(observation, Mapping)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
            or not 0.15 <= float(values[1]) <= float(values[0]) <= float(values[2]) <= 0.92
        ):
            raise ValueError("invalid sporting free-agent evidence")
    if len(free_agent_ids) != len(set(free_agent_ids)):
        raise ValueError("invalid sporting free-agent evidence")
    expected_rows = _diagnose_roles(
        outgoing, expiring=expiring, retiring=retiring,
        recruitment_candidates=recruitment, free_candidates=free_agents,
        continuity_roles=sorted(set(
            list((previous_review or {}).get("priority_unaddressed_roles", []))
            + list((previous_review or {}).get("incoming_low_usage_roles", []))
        )),
    )
    expected_roles = [
        row["role"] for row in sorted(
            expected_rows, key=lambda row: (-row["need_score"], row["role"]),
        )
        if row["recruitment_options"] or row["free_agent_options"]
    ][:3]
    ages = [
        row["age"] for row in outgoing
        if isinstance(row.get("age"), int) and not isinstance(row.get("age"), bool)
    ]
    allowance = brief.get("available_budget")
    expected_philosophy = (
        "financial_control" if allowance <= 2
        else "youth_pathway" if ages and sum(ages) / len(ages) >= 29
        else "win_now" if sum(
            row["expiring"] + row["retiring"] for row in expected_rows
        ) == 0 and allowance >= 6
        else "balanced"
    )
    expected_policy = {
        "priority_roles_bind_all_incoming_players": True,
        "low_risk_requires_scouted_lower_bound_upgrade": True,
        "balanced_risk_requires_scouted_free_agent": True,
        "high_risk_allows_baseline_free_agent": True,
        "win_now_requires_estimated_upgrade": 0.015,
        "youth_pathway_max_incoming_age": 24,
        "financial_control_recruitment_cap": min(4, allowance),
        "financial_control_max_renewals": 2,
    }
    if (
        rows != expected_rows
        or brief["recommendation"].get("priority_roles") != expected_roles
        or brief["recommendation"].get("philosophy") != expected_philosophy
        or brief["recommendation"].get("risk_level") != "balanced"
        or brief["recommendation"].get("reason") != _RECOMMENDATION_REASON
        or dict(brief["policy_contract"]) != expected_policy
    ):
        raise ValueError("sporting brief diagnostic replay mismatch")


def _selected_recruitment(
    brief: Mapping[str, Any], payload: Mapping[str, Any] | None,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if payload is None:
        return []
    candidates = {
        str(row.get("player_id") or ""): row
        for row in brief["recruitment_candidates"]
    }
    outgoing = {
        str(row.get("player_id") or ""): row for row in brief["outgoing_players"]
    }
    selected = []
    for move in payload.get("moves") or []:
        candidate = candidates.get(str(move.get("candidate_id") or ""))
        replaced = outgoing.get(str(move.get("outgoing_player_id") or ""))
        if candidate is None or replaced is None:
            raise ValueError("sporting plan recruitment source mismatch")
        selected.append((candidate, replaced))
    return selected


def validate_sporting_choices(
    directive: SportingDirective, brief: Mapping[str, Any], *,
    recruitment: Mapping[str, Any] | None,
    retention: Mapping[str, Any] | None,
    free_agent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validate_sporting_brief(brief)
    if directive.planning_id != brief["planning_id"]:
        raise ValueError("sporting directive does not match the frozen brief")
    selected = _selected_recruitment(brief, recruitment)
    free_pair = None
    if free_agent is not None:
        candidates = {
            str(row.get("player_id") or ""): row
            for row in brief["free_agent_candidates"]
        }
        outgoing = {
            str(row.get("player_id") or ""): row
            for row in brief["outgoing_players"]
        }
        candidate = candidates.get(str(free_agent.get("free_agent_id") or ""))
        replaced = outgoing.get(str(free_agent.get("outgoing_player_id") or ""))
        if candidate is None or replaced is None:
            raise ValueError("sporting plan free-agent source mismatch")
        free_pair = (candidate, replaced)
    incoming = [candidate for candidate, _ in selected]
    if free_pair is not None:
        incoming.append(free_pair[0])
    if directive.priority_roles and any(
        candidate.get("role") not in directive.priority_roles
        for candidate in incoming
    ):
        raise ValueError("incoming player is outside sporting priority roles")
    if directive.philosophy == "youth_pathway" and any(
        not isinstance(candidate.get("age"), int) or candidate["age"] > 24
        for candidate in incoming
    ):
        raise ValueError("youth pathway permits only incoming players aged 24 or younger")
    if directive.philosophy == "win_now":
        for candidate, replaced in selected:
            if float(candidate["quality"]) - float(replaced["quality"]) < 0.015:
                raise ValueError("win-now recruitment requires a quality upgrade")
        if free_pair is not None and (
            float(free_pair[0]["observation"]["estimated_quality"])
            - float(free_pair[1]["quality"]) < 0.015
        ):
            raise ValueError("win-now free-agent choice requires an estimated upgrade")
    recruitment_spend = sum(
        int(candidate["recruitment_cost"]) for candidate, _ in selected
    )
    renewals = list((retention or {}).get("renew_player_ids") or [])
    if directive.philosophy == "financial_control":
        cap = int(brief["policy_contract"]["financial_control_recruitment_cap"])
        if recruitment_spend > cap:
            raise ValueError("financial-control recruitment exceeds the planning cap")
        if len(renewals) > int(
            brief["policy_contract"]["financial_control_max_renewals"],
        ):
            raise ValueError("financial-control plan exceeds its renewal limit")
        if free_pair is not None:
            estimated_tier = (
                1 if free_pair[0]["observation"]["estimated_quality"] < 0.62
                else 2 if free_pair[0]["observation"]["estimated_quality"] < 0.70
                else 3 if free_pair[0]["observation"]["estimated_quality"] < 0.78
                else 4
            )
            outgoing_tier = (
                1 if free_pair[1]["quality"] < 0.62
                else 2 if free_pair[1]["quality"] < 0.70
                else 3 if free_pair[1]["quality"] < 0.78 else 4
            )
            if estimated_tier > outgoing_tier:
                raise ValueError("financial-control free agent raises the estimated wage tier")
    if free_pair is not None and directive.risk_level in {"low", "balanced"}:
        if not free_pair[0].get("scouted"):
            raise ValueError("this sporting risk level requires a scouted free agent")
        if (
            directive.risk_level == "low"
            and float(free_pair[0]["observation"]["quality_low"])
            < float(free_pair[1]["quality"])
        ):
            raise ValueError("low-risk plan requires the scouting lower bound to improve the squad")
    return {
        "schema_version": 1, "planning_id": directive.planning_id,
        "team": brief["team"], "target_season_id": brief["target_season_id"],
        "directive": directive.as_dict(),
        "selected_recruitment_moves": len(selected),
        "selected_free_agent": free_pair is not None,
        "selected_renewals": len(renewals),
        "recruitment_spend": recruitment_spend,
        "claim_boundary": (
            "frozen fictional sporting-policy compliance; not a prediction that "
            "the selected decisions will improve results"
        ),
    }


__all__ = [
    "PHILOSOPHIES", "RISK_LEVELS", "SUPPORTED_ROLES",
    "SportingDirective", "build_sporting_brief",
    "validate_sporting_brief", "validate_sporting_choices",
]
