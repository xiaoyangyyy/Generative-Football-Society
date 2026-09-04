"""Domain-level player lifecycle transactions from season participation evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


DEVELOPMENT_SCHEMA_VERSION = 1
MAX_DEVELOPMENT_CLUBS = 64
MAX_DEVELOPMENT_TRANSACTIONS = 64
_SEASON_PATTERN = re.compile(r"^season-(\d{4,})$")
_COMMON_ABILITIES = (
    "tech", "pass_skill", "vision", "spatial", "pace", "press",
    "curve", "shot", "power", "aerial", "heading", "mental",
    "gk_reflex", "gk_aerial",
)
_ROLE_EMPHASIS = {
    "GK": {"gk_reflex", "gk_aerial", "mental", "aerial"},
    "CB": {"press", "power", "aerial", "heading", "mental"},
    "LB": {"pace", "press", "pass_skill", "spatial"},
    "RB": {"pace", "press", "pass_skill", "spatial"},
    "DM": {"press", "mental", "pass_skill", "spatial"},
    "CM": {"pass_skill", "vision", "tech", "spatial"},
    "AM": {"tech", "vision", "pass_skill", "shot"},
    "LW": {"pace", "tech", "shot", "curve"},
    "RW": {"pace", "tech", "shot", "curve"},
    "ST": {"shot", "power", "pace", "heading"},
}


def development_identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def roster_identity(value: Mapping[str, Any]) -> str:
    return development_identity(value)


def _safe_report_path(root: Path, reference: Any) -> Path | None:
    text = str(reference or "").strip()
    if not text or Path(text).is_absolute():
        return None
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.suffix.lower() == ".json" else None


def collect_season_participation(
    root: str | Path, season: Mapping[str, Any], *, team: str,
) -> dict[str, Any]:
    """Collect exact player minutes from bounded completed-match reports."""
    resolved = Path(root).resolve()
    season_id = str(season.get("season_id") or "")
    fixtures = [
        fixture for fixture in season.get("fixtures") or []
        if isinstance(fixture, Mapping) and team in {
            fixture.get("home"), fixture.get("away"),
        }
    ]
    if not season_id or not fixtures:
        raise ValueError("development participation season identity mismatch")
    reports = []
    missing = []
    totals: dict[str, dict[str, float | int]] = {}
    for fixture in sorted(fixtures, key=lambda item: str(item.get("fixture_id") or "")):
        fixture_id = str(fixture.get("fixture_id") or "")
        report_path = _safe_report_path(resolved, fixture.get("report"))
        if report_path is None or not report_path.is_file():
            missing.append(fixture_id)
            continue
        if report_path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("development participation report is oversized")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        competition = (payload.get("match_plan") or {}).get("competition") or {}
        if (
            payload.get("match_id") != fixture.get("match_id")
            or competition.get("season_id") != season_id
            or competition.get("fixture_id") != fixture_id
        ):
            raise ValueError("development participation report identity mismatch")
        player_stats = ((payload.get("raw_summary") or {}).get("player_stats") or {})
        team_stats = player_stats.get(team)
        if not isinstance(team_stats, Mapping):
            missing.append(fixture_id)
            continue
        frozen_minutes = {}
        for raw_player_id, stats in sorted(team_stats.items(), key=lambda item: str(item[0])):
            player_id = str(raw_player_id)
            if (
                raw_player_id != player_id or not player_id.strip()
                or len(player_id) > 128 or not isinstance(stats, Mapping)
            ):
                raise ValueError("development player participation is invalid")
            minutes = stats.get("minutes", 0.0)
            if (
                isinstance(minutes, bool) or not isinstance(minutes, (int, float))
                or not math.isfinite(float(minutes)) or not 0 <= float(minutes) <= 120
            ):
                raise ValueError("development player minutes are invalid")
            rounded = round(float(minutes), 3)
            if rounded <= 0:
                continue
            frozen_minutes[player_id] = rounded
            aggregate = totals.setdefault(
                player_id, {"minutes": 0.0, "appearances": 0},
            )
            aggregate["minutes"] = round(float(aggregate["minutes"]) + rounded, 3)
            aggregate["appearances"] = int(aggregate["appearances"]) + 1
        digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        reports.append({
            "fixture_id": fixture_id,
            "match_id": str(fixture.get("match_id") or ""),
            "report": report_path.relative_to(resolved).as_posix(),
            "sha256": digest,
            "minutes": frozen_minutes,
        })
    scheduled = [str(fixture.get("fixture_id") or "") for fixture in fixtures]
    coverage = (
        "complete" if len(reports) == len(fixtures)
        else "unavailable" if not reports else "partial"
    )
    evidence = {
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "season_id": season_id,
        "team": team,
        "scheduled_fixture_ids": sorted(scheduled),
        "reports": reports,
        "missing_fixture_ids": sorted(missing),
        "coverage": coverage,
        "player_totals": {
            player_id: {
                "minutes": round(float(value["minutes"]), 3),
                "appearances": int(value["appearances"]),
            }
            for player_id, value in sorted(totals.items())
        },
        "claim_boundary": (
            "exact minutes from available persisted simulation reports; missing "
            "reports do not receive invented participation"
        ),
    }
    validate_participation_evidence(evidence)
    return evidence


def validate_participation_evidence(evidence: Mapping[str, Any]) -> None:
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != 1:
        raise ValueError("invalid development participation evidence")
    scheduled = evidence.get("scheduled_fixture_ids")
    reports = evidence.get("reports")
    missing = evidence.get("missing_fixture_ids")
    totals = evidence.get("player_totals")
    if (
        not str(evidence.get("season_id") or "")
        or not str(evidence.get("team") or "")
        or not isinstance(scheduled, list) or scheduled != sorted(set(scheduled))
        or not isinstance(reports, list) or not isinstance(missing, list)
        or missing != sorted(set(missing))
        or {item.get("fixture_id") for item in reports if isinstance(item, Mapping)}
        | set(missing) != set(scheduled)
        or len(reports) + len(missing) != len(scheduled)
        or evidence.get("coverage") not in {"complete", "partial", "unavailable"}
        or not isinstance(totals, Mapping)
    ):
        raise ValueError("invalid development participation coverage")
    expected_coverage = (
        "complete" if len(reports) == len(scheduled)
        else "unavailable" if not reports else "partial"
    )
    if evidence.get("coverage") != expected_coverage:
        raise ValueError("development participation coverage mismatch")
    recomputed: dict[str, dict[str, float | int]] = {}
    for report in reports:
        if (
            not isinstance(report, Mapping)
            or set(report) != {"fixture_id", "match_id", "report", "sha256", "minutes"}
            or not isinstance(report.get("minutes"), Mapping)
            or not isinstance(report.get("sha256"), str)
            or len(report["sha256"]) != 64
        ):
            raise ValueError("invalid development participation report")
        for player_id, minutes in report["minutes"].items():
            if (
                not isinstance(player_id, str) or not player_id.strip()
                or isinstance(minutes, bool) or not isinstance(minutes, (int, float))
                or not math.isfinite(float(minutes)) or not 0 < float(minutes) <= 120
            ):
                raise ValueError("invalid development participation minutes")
            aggregate = recomputed.setdefault(
                player_id, {"minutes": 0.0, "appearances": 0},
            )
            aggregate["minutes"] = round(float(aggregate["minutes"]) + float(minutes), 3)
            aggregate["appearances"] = int(aggregate["appearances"]) + 1
    canonical = {
        player_id: {
            "minutes": round(float(value["minutes"]), 3),
            "appearances": int(value["appearances"]),
        }
        for player_id, value in sorted(recomputed.items())
    }
    if dict(totals) != canonical:
        raise ValueError("development participation totals mismatch")


def verify_participation_report_files(
    root: str | Path, evidence: Mapping[str, Any],
) -> dict[str, int]:
    """Recheck report hashes when retained files remain locally available."""
    validate_participation_evidence(evidence)
    resolved = Path(root).resolve()
    checked = 0
    unavailable = 0
    for report in evidence["reports"]:
        path = _safe_report_path(resolved, report["report"])
        if path is None:
            raise ValueError("development participation report path is invalid")
        if not path.is_file():
            unavailable += 1
            continue
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("development participation report is oversized")
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["sha256"]:
            raise ValueError("development participation report hash mismatch")
        checked += 1
    return {"checked": checked, "unavailable": unavailable}


def _age_phase(age: int | None) -> tuple[str, float]:
    if age is None:
        return "unknown", 0.0
    if age <= 21:
        return "prospect", 0.004
    if age <= 24:
        return "developing", 0.003
    if age <= 27:
        return "prime_growth", 0.001
    if age <= 29:
        return "prime", 0.0
    if age <= 32:
        return "veteran", -0.003
    return "late_career", -0.006


def _season_delta(age: int | None, participation: float, science: int) -> tuple[str, float, float, float, float]:
    phase, base = _age_phase(age)
    if age is None:
        return phase, base, 0.0, 0.0, 0.0
    if age <= 24:
        minutes_bonus = 0.004 * participation
    elif age <= 29:
        minutes_bonus = 0.0015 * participation
    else:
        minutes_bonus = 0.001 * participation
    science_bonus = (0.0005 if age <= 27 else 0.0004) * science
    total = min(0.012, max(-0.008, base + minutes_bonus + science_bonus))
    return phase, base, minutes_bonus, science_bonus, total


def build_development_transaction(
    source_archive: Mapping[str, Any], source_roster: Mapping[str, Any],
    target_roster: Mapping[str, Any], participation: Mapping[str, Any], *,
    team: str, target_season_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_participation_evidence(participation)
    source_id = str(source_archive.get("season_id") or "")
    if (
        participation.get("team") != team
        or participation.get("season_id") != source_id
        or str(source_roster.get("team_id") or team) != team
        or str(target_roster.get("team_id") or team) != team
    ):
        raise ValueError("development transaction source identity mismatch")
    plan = source_archive.get("plan") or {}
    resources = plan.get("manager_resources") or {}
    science = (
        int(resources.get("sports_science", 0))
        if plan.get("manager_team") == team else 0
    )
    if not 0 <= science <= 4:
        raise ValueError("development sports-science evidence is invalid")
    source_players = {
        str(player.get("player_id") or ""): player
        for player in source_roster.get("players") or []
        if isinstance(player, Mapping)
    }
    result = copy.deepcopy(dict(target_roster))
    target_players = {
        str(player.get("player_id") or ""): player
        for player in result.get("players") or []
        if isinstance(player, Mapping)
    }
    if not source_players or not target_players or "" in source_players | target_players:
        raise ValueError("development requires valid source and target rosters")
    match_count = max(1, len(participation["scheduled_fixture_ids"]))
    records = []
    changed = 0
    total_delta = 0.0
    totals = participation["player_totals"]
    for player_id in sorted(target_players):
        target = target_players[player_id]
        source = source_players.get(player_id)
        if source is None:
            records.append({
                "player_id": player_id, "name": str(target.get("name") or player_id),
                "status": "new_signing", "age_before": target.get("age"),
                "age_after": target.get("age"), "minutes": 0.0, "appearances": 0,
                "participation_share": 0.0, "age_phase": "not_eligible",
                "base_delta": 0.0, "minutes_bonus": 0.0,
                "sports_science_bonus": 0.0, "season_delta": 0.0,
                "ability_changes": {},
            })
            continue
        raw_age = source.get("age")
        age = (
            int(raw_age)
            if isinstance(raw_age, int) and not isinstance(raw_age, bool)
            and 16 <= raw_age <= 45 else None
        )
        aggregate = totals.get(player_id) or {}
        minutes = round(float(aggregate.get("minutes", 0.0)), 3)
        appearances = int(aggregate.get("appearances", 0))
        share = round(min(1.0, minutes / (match_count * 90.0)), 6)
        phase, base, minute_bonus, science_bonus, delta = _season_delta(
            age, share, science,
        )
        abilities = target.get("abilities")
        changes = {}
        if isinstance(abilities, dict) and age is not None:
            emphasis = _ROLE_EMPHASIS.get(str(target.get("role") or ""), set())
            for key in _COMMON_ABILITIES:
                value = abilities.get(key)
                if (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    continue
                weight = 1.15 if key in emphasis else 0.85
                after = round(min(0.92, max(0.15, float(value) + delta * weight)), 6)
                before = round(float(value), 6)
                if after != before:
                    abilities[key] = after
                    changes[key] = {"before": before, "after": after}
            target["age"] = age + 1
        mean_delta = (
            round(sum(change["after"] - change["before"] for change in changes.values()) / len(changes), 6)
            if changes else 0.0
        )
        if changes:
            changed += 1
            total_delta += mean_delta
        records.append({
            "player_id": player_id, "name": str(target.get("name") or player_id),
            "status": "developed" if changes else "age_unknown",
            "age_before": age, "age_after": age + 1 if age is not None else None,
            "minutes": minutes, "appearances": appearances,
            "participation_share": share, "age_phase": phase,
            "base_delta": round(base, 6),
            "minutes_bonus": round(minute_bonus, 6),
            "sports_science_bonus": round(science_bonus, 6),
            "season_delta": round(delta, 6),
            "ability_changes": changes,
        })
    result["source"] = (
        str(result.get("source") or "unknown") + "+gfs_player_development_v1"
    )
    departed = sorted(set(source_players) - set(target_players))
    transaction = {
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "type": "season_development",
        "source_season_id": source_id,
        "target_season_id": target_season_id,
        "team": team,
        "source_archive_identity": development_identity(source_archive),
        "participation_identity": development_identity(participation),
        "prior_roster_identity": roster_identity(target_roster),
        "result_roster_identity": roster_identity(result),
        "sports_science_points": science,
        "players": records,
        "departed_player_ids": departed,
        "summary": {
            "eligible_players": sum(record["status"] == "developed" for record in records),
            "changed_players": changed,
            "new_signings": sum(record["status"] == "new_signing" for record in records),
            "departed_players": len(departed),
            "total_minutes_observed": round(sum(record["minutes"] for record in records), 3),
            "mean_ability_delta": round(total_delta / changed, 6) if changed else 0.0,
            "evidence_coverage": participation["coverage"],
        },
        "claim_boundary": (
            "bounded fictional lifecycle mechanics from age, exact available "
            "simulation minutes, and frozen sports-science resources"
        ),
    }
    return transaction, result


def apply_development_transaction(
    roster: Mapping[str, Any], transaction: Mapping[str, Any],
) -> dict[str, Any]:
    validate_development_transaction(transaction)
    if (
        not isinstance(transaction, Mapping)
        or transaction.get("schema_version") != 1
        or transaction.get("type") != "season_development"
        or roster_identity(roster) != transaction.get("prior_roster_identity")
    ):
        raise ValueError("development transaction prior roster mismatch")
    result = copy.deepcopy(dict(roster))
    players = {
        str(player.get("player_id") or ""): player
        for player in result.get("players") or []
        if isinstance(player, Mapping)
    }
    records = transaction.get("players")
    if not isinstance(records, list) or [record.get("player_id") for record in records] != sorted(players):
        raise ValueError("development transaction player coverage mismatch")
    for record in records:
        player = players.get(str(record.get("player_id") or ""))
        if player is None or record.get("status") not in {
            "developed", "age_unknown", "new_signing",
        }:
            raise ValueError("invalid development player record")
        changes = record.get("ability_changes")
        if not isinstance(changes, Mapping):
            raise ValueError("invalid development ability changes")
        abilities = player.get("abilities")
        if changes and not isinstance(abilities, dict):
            raise ValueError("development target abilities are unavailable")
        for key, change in changes.items():
            if (
                key not in _COMMON_ABILITIES or not isinstance(change, Mapping)
                or round(float(abilities.get(key)), 6) != change.get("before")
            ):
                raise ValueError("development ability evidence mismatch")
            abilities[key] = change["after"]
        if record.get("age_before") is not None:
            if player.get("age") != record["age_before"]:
                raise ValueError("development age evidence mismatch")
            player["age"] = record["age_after"]
    result["source"] = (
        str(result.get("source") or "unknown") + "+gfs_player_development_v1"
    )
    if roster_identity(result) != transaction.get("result_roster_identity"):
        raise ValueError("development transaction result roster mismatch")
    return result


def validate_development_transaction(transaction: Mapping[str, Any]) -> None:
    """Validate a transaction without requiring its retained source archive."""
    if not isinstance(transaction, Mapping):
        raise ValueError("invalid development transaction")
    source = _SEASON_PATTERN.fullmatch(str(transaction.get("source_season_id") or ""))
    target = _SEASON_PATTERN.fullmatch(str(transaction.get("target_season_id") or ""))
    identities = (
        transaction.get("source_archive_identity"),
        transaction.get("participation_identity"),
        transaction.get("prior_roster_identity"),
        transaction.get("result_roster_identity"),
    )
    science = transaction.get("sports_science_points")
    players = transaction.get("players")
    departed = transaction.get("departed_player_ids")
    if (
        transaction.get("schema_version") != 1
        or transaction.get("type") != "season_development"
        or source is None or target is None
        or int(target.group(1)) <= int(source.group(1))
        or not str(transaction.get("team") or "").strip()
        or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in identities)
        or isinstance(science, bool) or not isinstance(science, int)
        or not 0 <= science <= 4
        or not isinstance(players, list) or not isinstance(departed, list)
        or departed != sorted(set(departed))
    ):
        raise ValueError("invalid development transaction")
    player_ids = [record.get("player_id") for record in players if isinstance(record, Mapping)]
    if len(player_ids) != len(players) or player_ids != sorted(set(player_ids)):
        raise ValueError("invalid development player ordering")
    changed = 0
    total_delta = 0.0
    minutes_total = 0.0
    new_signings = 0
    for record in players:
        status = record.get("status")
        changes = record.get("ability_changes")
        numeric = (
            record.get("minutes"), record.get("participation_share"),
            record.get("base_delta"), record.get("minutes_bonus"),
            record.get("sports_science_bonus"), record.get("season_delta"),
        )
        if (
            status not in {"developed", "age_unknown", "new_signing"}
            or not str(record.get("player_id") or "").strip()
            or not isinstance(changes, Mapping)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric)
            or not 0 <= float(record["minutes"])
            or not 0 <= float(record["participation_share"]) <= 1
            or not -0.008 <= float(record["season_delta"]) <= 0.012
            or isinstance(record.get("appearances"), bool)
            or not isinstance(record.get("appearances"), int)
            or record["appearances"] < 0
        ):
            raise ValueError("invalid development player record")
        age_before = record.get("age_before")
        age_after = record.get("age_after")
        expected_age_after = (
            age_before if status == "new_signing"
            else age_before + 1 if isinstance(age_before, int) and not isinstance(age_before, bool)
            else None
        )
        if (
            (age_before is None) != (age_after is None)
            or (age_before is not None and (
                isinstance(age_before, bool) or not isinstance(age_before, int)
                or age_after != expected_age_after
            ))
        ):
            raise ValueError("invalid development age transition")
        deltas = []
        for ability, change in changes.items():
            before = change.get("before") if isinstance(change, Mapping) else None
            after = change.get("after") if isinstance(change, Mapping) else None
            if (
                ability not in _COMMON_ABILITIES
                or isinstance(before, bool) or not isinstance(before, (int, float))
                or isinstance(after, bool) or not isinstance(after, (int, float))
                or not 0.15 <= float(after) <= 0.92
            ):
                raise ValueError("invalid development ability record")
            deltas.append(float(after) - float(before))
        if bool(changes) != (status == "developed"):
            raise ValueError("development status and changes mismatch")
        if status == "new_signing":
            new_signings += 1
        if changes:
            changed += 1
            total_delta += round(sum(deltas) / len(deltas), 6)
        minutes_total += float(record["minutes"])
    summary = transaction.get("summary")
    expected_summary = {
        "eligible_players": sum(record["status"] == "developed" for record in players),
        "changed_players": changed,
        "new_signings": new_signings,
        "departed_players": len(departed),
        "total_minutes_observed": round(minutes_total, 3),
        "mean_ability_delta": round(total_delta / changed, 6) if changed else 0.0,
        "evidence_coverage": (summary or {}).get("evidence_coverage"),
    }
    if (
        not isinstance(summary, Mapping)
        or summary.get("evidence_coverage") not in {"complete", "partial", "unavailable"}
        or dict(summary) != expected_summary
    ):
        raise ValueError("development transaction summary mismatch")


def validate_development_registry(registry: Mapping[str, Any] | None) -> dict[str, Any]:
    if registry is None:
        return {"schema_version": 1, "club_count": 0, "transaction_count": 0}
    if not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1:
        raise ValueError("invalid development registry")
    clubs = registry.get("clubs") or {}
    if not isinstance(clubs, Mapping) or len(clubs) > MAX_DEVELOPMENT_CLUBS:
        raise ValueError("invalid development registry clubs")
    count = 0
    for raw_team, club in clubs.items():
        team = str(raw_team)
        transactions = club.get("transactions") if isinstance(club, Mapping) else None
        if raw_team != team or not team.strip() or not isinstance(transactions, list):
            raise ValueError("invalid development registry club")
        if len(transactions) > MAX_DEVELOPMENT_TRANSACTIONS:
            raise ValueError("development transaction history is oversized")
        previous_target = 0
        for transaction in transactions:
            validate_development_transaction(transaction)
            source = _SEASON_PATTERN.fullmatch(str(transaction.get("source_season_id") or ""))
            target = _SEASON_PATTERN.fullmatch(str(transaction.get("target_season_id") or ""))
            if (
                not isinstance(transaction, Mapping)
                or transaction.get("schema_version") != 1
                or transaction.get("type") != "season_development"
                or transaction.get("team") != team
                or source is None or target is None
                or int(target.group(1)) <= int(source.group(1))
                or int(target.group(1)) <= previous_target
                or not isinstance(transaction.get("players"), list)
                or not isinstance(transaction.get("summary"), Mapping)
            ):
                raise ValueError("invalid development transaction")
            previous_target = int(target.group(1))
        count += len(transactions)
    return {"schema_version": 1, "club_count": len(clubs), "transaction_count": count}


def append_development_transaction(
    registry: Mapping[str, Any] | None, *, team: str,
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(registry or {"schema_version": 1, "clubs": {}}))
    validate_development_registry(output)
    clubs = output.setdefault("clubs", {})
    club = clubs.setdefault(team, {"transactions": []})
    if transaction.get("team") != team:
        raise ValueError("development transaction team mismatch")
    target = transaction.get("target_season_id")
    if any(item.get("target_season_id") == target for item in club["transactions"]):
        raise ValueError("development transaction already exists")
    club["transactions"].append(copy.deepcopy(dict(transaction)))
    validate_development_registry(output)
    return output


def development_transactions_for_team(
    registry: Mapping[str, Any] | None, team: str,
) -> list[dict[str, Any]]:
    validate_development_registry(registry)
    club = ((registry or {}).get("clubs") or {}).get(team) or {}
    return [copy.deepcopy(item) for item in club.get("transactions") or []]


def development_view(
    registry: Mapping[str, Any] | None, *, team: str | None = None,
) -> dict[str, Any]:
    summary = validate_development_registry(registry)
    clubs = (registry or {}).get("clubs") or {}
    latest = []
    for club_team, club in sorted(clubs.items()):
        if team is not None and club_team != team:
            continue
        transactions = club.get("transactions") or []
        if transactions:
            latest.append(copy.deepcopy(transactions[-1]))
    return {
        **summary, "available": bool(latest), "latest_transactions": latest,
        "claim_boundary": (
            "bounded simulated player lifecycle; not a real development, "
            "medical, scouting, or valuation claim"
        ),
    }


__all__ = [
    "apply_development_transaction", "append_development_transaction",
    "build_development_transaction", "collect_season_participation",
    "development_identity", "development_transactions_for_team",
    "development_view", "validate_development_registry",
    "validate_development_transaction", "validate_participation_evidence",
    "verify_participation_report_files",
]
