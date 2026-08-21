"""Bounded, replayable club-finance ledger for simulated competitions."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.simulation.squad_registry import roster_identity, squad_player_quality


FINANCE_SCHEMA_VERSION = 1
OPENING_BALANCE = 8
RECRUITMENT_WINDOW_CAP = 8
MAX_FINANCE_CLUBS = 64
MAX_FINANCE_ENTRIES_PER_CLUB = 256
MIN_BALANCE = -20
MAX_BALANCE = 10_000


def evidence_identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def player_wage_tier(player: Mapping[str, Any]) -> int:
    quality = squad_player_quality(player)
    if not math.isfinite(quality):
        raise ValueError("player wage quality must be finite")
    if quality < 0.62:
        return 1
    if quality < 0.70:
        return 2
    if quality < 0.78:
        return 3
    return 4


def seasonal_wage_expense(roster: Mapping[str, Any]) -> dict[str, Any]:
    players = roster.get("players") if isinstance(roster, Mapping) else None
    if not isinstance(players, list) or len(players) < 11 or len(players) > 80:
        raise ValueError("wage calculation requires a bounded playable roster")
    tiers = [player_wage_tier(player) for player in players if isinstance(player, Mapping)]
    if len(tiers) != len(players):
        raise ValueError("wage roster contains an invalid player")
    tier_total = sum(tiers)
    expense = max(1, math.ceil(tier_total / 6))
    return {
        "roster_identity": roster_identity(roster),
        "player_count": len(players),
        "wage_tier_total": tier_total,
        "seasonal_wage_expense": expense,
        "tier_counts": {
            str(tier): tiers.count(tier) for tier in range(1, 5)
        },
    }


def build_season_finance_settlement(
    archive: Mapping[str, Any], roster: Mapping[str, Any] | None, *, team: str,
) -> dict[str, Any]:
    if not isinstance(archive, Mapping) or archive.get("schema_version") != 1:
        raise ValueError("finance settlement requires a season archive")
    plan = archive.get("plan") or {}
    teams = plan.get("teams") if isinstance(plan, Mapping) else None
    if not isinstance(teams, list) or team not in teams:
        raise ValueError("finance settlement competition identity mismatch")
    standings = archive.get("final_standings")
    profile = archive.get("manager_profile")
    if not isinstance(standings, list):
        raise ValueError("finance settlement evidence is incomplete")
    rows = [row for row in standings if isinstance(row, Mapping) and row.get("team") == team]
    is_manager_team = plan.get("manager_team") == team
    objective = profile.get("objective") or {} if isinstance(profile, Mapping) else {}
    if (
        len(rows) != 1
        or is_manager_team
        and objective.get("status") not in {"achieved", "missed"}
    ):
        raise ValueError("finance settlement result identity mismatch")
    row = rows[0]
    position = row.get("position")
    points = row.get("points")
    if (
        isinstance(position, bool) or not isinstance(position, int)
        or not 1 <= position <= len(standings)
        or isinstance(points, bool) or not isinstance(points, int) or points < 0
    ):
        raise ValueError("finance settlement standings are invalid")
    wage = (
        seasonal_wage_expense(roster)
        if roster is not None else {
            "roster_identity": None,
            "player_count": None,
            "wage_tier_total": None,
            "seasonal_wage_expense": 6,
            "tier_counts": None,
            "source": "team_level_baseline_without_roster",
        }
    )
    revenue = {
        "participation": 6,
        "points": points // 3,
        "league_position": max(0, len(standings) - position),
        "objective": 3 if is_manager_team and objective["status"] == "achieved" else 0,
    }
    revenue_total = sum(revenue.values())
    expense_total = int(wage["seasonal_wage_expense"])
    return {
        "schema_version": FINANCE_SCHEMA_VERSION,
        "type": "season_settlement",
        "season_id": str(archive.get("season_id") or ""),
        "team": team,
        "archive_identity": evidence_identity(archive),
        "roster_identity": wage["roster_identity"],
        "revenue": revenue,
        "revenue_total": revenue_total,
        "wage": wage,
        "expense_total": expense_total,
        "delta": revenue_total - expense_total,
        "control": "manager" if is_manager_team else "ai_club",
        "claim_boundary": (
            "bounded game-economy settlement from simulated results and squad "
            "quality tiers; not real club revenue or salary data"
        ),
    }


def recruitment_allowance(balance: int) -> int:
    if isinstance(balance, bool) or not isinstance(balance, int):
        raise ValueError("club balance must be an integer")
    return max(0, min(RECRUITMENT_WINDOW_CAP, balance))


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": FINANCE_SCHEMA_VERSION, "clubs": {}}


def ensure_finance_club(
    registry: Mapping[str, Any] | None, team: str,
) -> dict[str, Any]:
    if not str(team).strip() or len(str(team)) > 96:
        raise ValueError("invalid finance club identity")
    output = json.loads(json.dumps(dict(registry or _empty_registry())))
    if output.get("schema_version", 1) != 1:
        raise ValueError("invalid finance registry")
    clubs = output.setdefault("clubs", {})
    if not isinstance(clubs, dict) or len(clubs) > MAX_FINANCE_CLUBS:
        raise ValueError("invalid finance registry clubs")
    clubs.setdefault(team, {
        "opening_balance": OPENING_BALANCE,
        "entries": [],
    })
    validate_finance_registry(output)
    return output


def _bounded_balance(value: int) -> int:
    return max(MIN_BALANCE, min(MAX_BALANCE, value))


def _canonical_entry(
    *, entry_type: str, season_id: str, team: str, evidence: Mapping[str, Any],
    balance_before: int, delta: int,
) -> dict[str, Any]:
    balance_after = _bounded_balance(balance_before + delta)
    return {
        "schema_version": FINANCE_SCHEMA_VERSION,
        "entry_id": f"{season_id}:{entry_type}",
        "type": entry_type,
        "season_id": season_id,
        "team": team,
        "evidence": dict(evidence),
        "balance_before": balance_before,
        "delta": delta,
        "balance_after": balance_after,
    }


def validate_finance_registry(
    registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if registry is None:
        return {"schema_version": 1, "club_count": 0, "entry_count": 0, "balances": {}}
    if not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1:
        raise ValueError("invalid finance registry")
    clubs = registry.get("clubs") or {}
    if not isinstance(clubs, Mapping) or len(clubs) > MAX_FINANCE_CLUBS:
        raise ValueError("invalid finance registry clubs")
    balances = {}
    total_entries = 0
    for raw_team, club in clubs.items():
        team = str(raw_team)
        if raw_team != team or not team.strip() or len(team) > 96 or not isinstance(club, Mapping):
            raise ValueError("invalid finance club entry")
        opening = club.get("opening_balance")
        entries = club.get("entries")
        if (
            isinstance(opening, bool) or not isinstance(opening, int)
            or not MIN_BALANCE <= opening <= MAX_BALANCE
            or not isinstance(entries, list)
            or len(entries) > MAX_FINANCE_ENTRIES_PER_CLUB
        ):
            raise ValueError("invalid finance club ledger")
        balance = opening
        seen: set[str] = set()
        for raw in entries:
            if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
                raise ValueError("invalid finance ledger entry")
            entry_type = raw.get("type")
            season_id = str(raw.get("season_id") or "")
            evidence = raw.get("evidence")
            if entry_type not in {"season_settlement", "recruitment_charge"} or not season_id:
                raise ValueError("invalid finance ledger entry identity")
            if not isinstance(evidence, Mapping):
                raise ValueError("invalid finance ledger evidence")
            if entry_type == "season_settlement":
                if evidence.get("type") != "season_settlement":
                    raise ValueError("finance settlement evidence type mismatch")
                delta = evidence.get("delta")
            else:
                spent = evidence.get("spent")
                allowance = evidence.get("allowance")
                if (
                    isinstance(spent, bool) or not isinstance(spent, int) or spent < 1
                    or isinstance(allowance, bool) or not isinstance(allowance, int)
                    or spent > allowance or allowance != recruitment_allowance(balance)
                ):
                    raise ValueError("invalid recruitment finance evidence")
                delta = -spent
            if isinstance(delta, bool) or not isinstance(delta, int):
                raise ValueError("finance ledger delta must be an integer")
            expected = _canonical_entry(
                entry_type=entry_type, season_id=season_id, team=team,
                evidence=evidence, balance_before=balance, delta=delta,
            )
            if dict(raw) != expected or expected["entry_id"] in seen:
                raise ValueError("finance ledger replay mismatch")
            seen.add(expected["entry_id"])
            balance = expected["balance_after"]
        balances[team] = balance
        total_entries += len(entries)
    return {
        "schema_version": 1,
        "club_count": len(clubs),
        "entry_count": total_entries,
        "balances": balances,
    }


def append_season_settlement(
    registry: Mapping[str, Any] | None, *, team: str,
    settlement: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = ensure_finance_club(registry, team)
    summary = validate_finance_registry(output)
    club = output["clubs"][team]
    season_id = str(settlement.get("season_id") or "")
    if settlement.get("team") != team or settlement.get("type") != "season_settlement":
        raise ValueError("invalid season finance settlement")
    if any(entry.get("entry_id") == f"{season_id}:season_settlement" for entry in club["entries"]):
        raise ValueError("season finance settlement already exists")
    delta = settlement.get("delta")
    if isinstance(delta, bool) or not isinstance(delta, int):
        raise ValueError("invalid season finance delta")
    entry = _canonical_entry(
        entry_type="season_settlement", season_id=season_id, team=team,
        evidence=settlement, balance_before=summary["balances"][team], delta=delta,
    )
    club["entries"].append(entry)
    validate_finance_registry(output)
    return output, entry


def append_recruitment_charge(
    registry: Mapping[str, Any] | None, *, team: str, season_id: str,
    squad_transaction: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = ensure_finance_club(registry, team)
    summary = validate_finance_registry(output)
    club = output["clubs"][team]
    spent = squad_transaction.get("spent")
    if (
        squad_transaction.get("team") != team
        or squad_transaction.get("season_id") != season_id
        or isinstance(spent, bool) or not isinstance(spent, int) or spent < 1
    ):
        raise ValueError("invalid recruitment charge transaction")
    if any(entry.get("entry_id") == f"{season_id}:recruitment_charge" for entry in club["entries"]):
        raise ValueError("recruitment charge already exists")
    balance = summary["balances"][team]
    allowance = recruitment_allowance(balance)
    if spent > allowance:
        raise ValueError(
            f"recruitment spend exceeds available club funds ({allowance})"
        )
    evidence = {
        "schema_version": 1,
        "type": "recruitment_charge",
        "season_id": season_id,
        "team": team,
        "spent": spent,
        "allowance": allowance,
        "squad_transaction_identity": evidence_identity(squad_transaction),
    }
    entry = _canonical_entry(
        entry_type="recruitment_charge", season_id=season_id, team=team,
        evidence=evidence, balance_before=balance, delta=-spent,
    )
    club["entries"].append(entry)
    validate_finance_registry(output)
    return output, entry


def club_finance_view(
    registry: Mapping[str, Any] | None, *, team: str,
    projected_settlement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = ensure_finance_club(registry, team)
    summary = validate_finance_registry(output)
    club = output["clubs"][team]
    balance = summary["balances"][team]
    projected = None
    if projected_settlement is not None:
        delta = projected_settlement.get("delta")
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise ValueError("invalid projected finance settlement")
        projected = {
            "settlement": dict(projected_settlement),
            "balance_after": _bounded_balance(balance + delta),
        }
    return {
        "schema_version": 1,
        "available": True,
        "team": team,
        "balance": balance,
        "recruitment_window_cap": RECRUITMENT_WINDOW_CAP,
        "recruitment_allowance": recruitment_allowance(balance),
        "opening_balance": club["opening_balance"],
        "ledger_entries": list(club["entries"][-24:]),
        "ledger_entry_count": len(club["entries"]),
        "projected_after_current_season": projected,
        "claim_boundary": (
            "bounded simulated club economy in game credits; not real revenue, "
            "salary, valuation, accounting, or financial advice"
        ),
    }
