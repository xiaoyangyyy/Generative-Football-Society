"""Replayable contracts, retirement, and academy promotion for career squads."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


LIFECYCLE_SCHEMA_VERSION = 1
MAX_RENEWALS = 4
RENEWAL_TERM = 3
NEW_SIGNING_TERM = 3
ACADEMY_TERM = 3
_SEASON = re.compile(r"^season-(\d{4,})$")
_PLAYER_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_ABILITIES = (
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


def lifecycle_identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _clean_player_id(value: Any) -> str:
    player_id = str(value or "").strip()
    if _PLAYER_ID.fullmatch(player_id) is None:
        raise ValueError("invalid lifecycle player identity")
    return player_id


def lifecycle_cycle_id(*, target_season_id: str, team: str, source_roster: Mapping[str, Any]) -> str:
    if _SEASON.fullmatch(str(target_season_id)) is None or not str(team).strip():
        raise ValueError("invalid lifecycle cycle identity")
    roster_hash = lifecycle_identity(source_roster)[:12]
    team_hash = hashlib.sha256(str(team).encode("utf-8")).hexdigest()[:10]
    return f"lifecycle-{target_season_id.split('-')[-1]}-{team_hash}-{roster_hash}"


@dataclass(frozen=True)
class RetentionPlan:
    cycle_id: str
    renew_player_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cycle_id = str(self.cycle_id or "").strip()
        ids = tuple(_clean_player_id(value) for value in self.renew_player_ids)
        if not cycle_id.startswith("lifecycle-") or len(cycle_id) > 96:
            raise ValueError("invalid retention cycle identity")
        if len(ids) > MAX_RENEWALS or len(set(ids)) != len(ids):
            raise ValueError(f"retention allows at most {MAX_RENEWALS} unique renewals")
        object.__setattr__(self, "cycle_id", cycle_id)
        object.__setattr__(self, "renew_player_ids", ids)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RetentionPlan":
        if not isinstance(payload, Mapping) or payload.get("schema_version", 1) != 1:
            raise ValueError("retention plan must be a versioned object")
        raw = payload.get("renew_player_ids", [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("retention renewals must be an array")
        return cls(str(payload.get("cycle_id") or ""), tuple(raw))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "cycle_id": self.cycle_id,
            "renew_player_ids": list(self.renew_player_ids),
        }


def _initial_contract_years(team: str, player_id: str) -> int:
    return 1 + int(_unit("initial-contract", team, player_id) * 3) % 3


def _retirement_age(team: str, player_id: str) -> int:
    return 36 + int(_unit("retirement-age", team, player_id) * 4) % 4


def retirement_age_for_player(team: str, player_id: str) -> int:
    return _retirement_age(str(team), _clean_player_id(player_id))


def _quality(player: Mapping[str, Any]) -> float:
    abilities = player.get("abilities")
    values = [
        float(value) for value in (abilities or {}).values()
        if not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ] if isinstance(abilities, Mapping) else []
    return round(sum(values) / len(values), 6) if values else 0.5


def _career_before(player: Mapping[str, Any], *, team: str) -> dict[str, Any]:
    player_id = _clean_player_id(player.get("player_id"))
    raw = player.get("career")
    if raw is None:
        return {
            "schema_version": 1,
            "contract_years_remaining": _initial_contract_years(team, player_id),
            "contract_source": "deterministic_initialization",
        }
    if (
        not isinstance(raw, Mapping) or raw.get("schema_version") != 1
        or isinstance(raw.get("contract_years_remaining"), bool)
        or not isinstance(raw.get("contract_years_remaining"), int)
        or not 1 <= raw["contract_years_remaining"] <= 5
        or raw.get("contract_source") not in {
            "deterministic_initialization", "renewed", "new_signing", "academy",
        }
    ):
        raise ValueError("invalid player career contract")
    return copy.deepcopy(dict(raw))


def build_lifecycle_preview(
    source_roster: Mapping[str, Any], *, team: str, target_season_id: str,
) -> dict[str, Any]:
    cycle_id = lifecycle_cycle_id(
        target_season_id=target_season_id, team=team, source_roster=source_roster,
    )
    target_index = int(_SEASON.fullmatch(str(target_season_id)).group(1))
    if target_index <= 1:
        raise ValueError("lifecycle requires a prior season")
    source_season_id = f"season-{target_index - 1:04d}"
    rows = []
    for player in sorted(source_roster.get("players") or [], key=lambda item: str(item.get("player_id") or "")):
        if not isinstance(player, Mapping):
            raise ValueError("invalid lifecycle source player")
        player_id = _clean_player_id(player.get("player_id"))
        career = _career_before(player, team=team)
        age = player.get("age")
        valid_age = (
            int(age) if isinstance(age, int) and not isinstance(age, bool)
            and 16 <= age <= 45 else None
        )
        retirement_age = _retirement_age(team, player_id)
        retires = valid_age is not None and valid_age + 1 >= retirement_age
        expires = career["contract_years_remaining"] == 1 and not retires
        rows.append({
            "player_id": player_id,
            "name": str(player.get("name") or player_id),
            "role": str(player.get("role") or ""),
            "age": valid_age,
            "quality": _quality(player),
            "contract_years_before": career["contract_years_remaining"],
            "retirement_age": retirement_age,
            "retires": retires,
            "contract_expires": expires,
        })
    expiring = [row for row in rows if row["contract_expires"]]
    recommended = sorted(
        expiring,
        key=lambda row: (
            0 if row["role"] == "GK" else 1,
            -row["quality"], row["player_id"],
        ),
    )[:MAX_RENEWALS]
    preview = {
        "schema_version": 1,
        "cycle_id": cycle_id,
        "team": team,
        "source_season_id": source_season_id,
        "target_season_id": target_season_id,
        "source_roster_identity": lifecycle_identity(source_roster),
        "renewal_limit": MAX_RENEWALS,
        "players": rows,
        "expiring_player_ids": [row["player_id"] for row in expiring],
        "retiring_player_ids": [row["player_id"] for row in rows if row["retires"]],
        "recommended_renew_player_ids": [row["player_id"] for row in recommended],
        "claim_boundary": (
            "deterministic fictional contract and retirement preview; not a real "
            "employment, medical, scouting, or retirement forecast"
        ),
    }
    return preview


def _academy_player(*, team: str, target_season_id: str, role: str, slot: int) -> dict[str, Any]:
    team_hash = hashlib.sha256(team.encode("utf-8")).hexdigest()[:8]
    seed = f"academy|{team}|{target_season_id}|{role}|{slot}"
    player_id = f"academy-{team_hash}-{target_season_id.split('-')[-1]}-{slot:02d}"
    base = 0.47 + 0.10 * _unit(seed, "base")
    emphasis = _ROLE_EMPHASIS.get(role, set())
    abilities = {}
    for ability in _ABILITIES:
        value = base + 0.035 * (_unit(seed, ability) - 0.5)
        if ability in emphasis:
            value += 0.025
        abilities[ability] = round(min(0.68, max(0.42, value)), 6)
    return {
        "player_id": player_id,
        "name": f"Academy {team_hash.upper()} {slot:02d}",
        "team_id": team,
        "role": role,
        "squad_role": "prospect",
        "age": 17,
        "availability": 1.0,
        "condition": {"composure": round(base, 6)},
        "abilities": abilities,
        "career": {
            "schema_version": 1,
            "contract_years_remaining": ACADEMY_TERM,
            "contract_source": "academy",
        },
        "source": "gfs_fictional_academy_v1",
    }


def build_lifecycle_transaction(
    source_roster: Mapping[str, Any], target_roster: Mapping[str, Any], *,
    team: str, target_season_id: str, plan: RetentionPlan | None = None,
    control: str = "ai",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if control not in {"manager", "ai"}:
        raise ValueError("invalid lifecycle control boundary")
    preview = build_lifecycle_preview(
        source_roster, team=team, target_season_id=target_season_id,
    )
    if plan is not None and plan.cycle_id != preview["cycle_id"]:
        raise ValueError("retention plan cycle identity mismatch")
    expiring = set(preview["expiring_player_ids"])
    renewals = set(
        plan.renew_player_ids if plan is not None
        else preview["recommended_renew_player_ids"]
    )
    if not renewals <= expiring:
        raise ValueError("retention plan contains a non-expiring player")
    source_players = {
        _clean_player_id(player.get("player_id")): player
        for player in source_roster.get("players") or [] if isinstance(player, Mapping)
    }
    result = copy.deepcopy(dict(target_roster))
    target_players = {
        _clean_player_id(player.get("player_id")): player
        for player in result.get("players") or [] if isinstance(player, Mapping)
    }
    if not source_players or not target_players or len(target_players) != len(result.get("players") or []):
        raise ValueError("lifecycle requires valid source and target rosters")
    preview_rows = {row["player_id"]: row for row in preview["players"]}
    exits = []
    updates = []
    survivors = []
    for player_id, player in sorted(target_players.items()):
        source = source_players.get(player_id)
        if source is None:
            before = player.get("career")
            after = {
                "schema_version": 1,
                "contract_years_remaining": NEW_SIGNING_TERM,
                "contract_source": "new_signing",
            }
            player["career"] = copy.deepcopy(after)
            updates.append({
                "player_id": player_id, "status": "new_signing",
                "before": copy.deepcopy(before), "after": after,
            })
            survivors.append(player)
            continue
        row = preview_rows[player_id]
        before = _career_before(source, team=team)
        reason = None
        if row["retires"]:
            reason = "retired"
        elif row["contract_expires"] and player_id not in renewals:
            reason = "contract_released"
        if reason is not None:
            exits.append({
                "player_id": player_id, "name": str(player.get("name") or player_id),
                "role": str(player.get("role") or ""), "reason": reason,
                "age": row["age"], "retirement_age": row["retirement_age"],
                "contract_years_before": before["contract_years_remaining"],
                "player": copy.deepcopy(dict(player)),
            })
            continue
        after = {
            "schema_version": 1,
            "contract_years_remaining": (
                RENEWAL_TERM if row["contract_expires"]
                else before["contract_years_remaining"] - 1
            ),
            "contract_source": (
                "renewed" if row["contract_expires"]
                else before["contract_source"]
            ),
        }
        player["career"] = copy.deepcopy(after)
        updates.append({
            "player_id": player_id,
            "status": "renewed" if row["contract_expires"] else "continued",
            "before": before, "after": after,
        })
        survivors.append(player)
    promotions = []
    existing_ids = {str(player.get("player_id") or "") for player in survivors}
    for slot, exit_row in enumerate(exits, start=1):
        academy = _academy_player(
            team=team, target_season_id=target_season_id,
            role=exit_row["role"], slot=slot,
        )
        if academy["player_id"] in existing_ids:
            raise ValueError("academy identity collision")
        existing_ids.add(academy["player_id"])
        promotions.append(copy.deepcopy(academy))
        survivors.append(academy)
    survivors.sort(key=lambda player: str(player.get("player_id") or ""))
    result["players"] = survivors
    result["squad_size"] = len(survivors)
    result["source"] = str(result.get("source") or "unknown") + "+gfs_player_lifecycle_v1"
    if len(survivors) != len(target_players) or len(existing_ids) != len(survivors):
        raise ValueError("lifecycle must preserve a unique fixed-size squad")
    if len(survivors) < 11 or not any(player.get("role") == "GK" for player in survivors):
        raise ValueError("lifecycle cannot leave an invalid playable squad")
    transaction = {
        "schema_version": 1,
        "type": "season_lifecycle",
        "team": team,
        "source_season_id": preview["source_season_id"],
        "target_season_id": target_season_id,
        "control": control,
        "cycle_id": preview["cycle_id"],
        "preview_identity": lifecycle_identity(preview),
        "plan": plan.as_dict() if plan is not None else None,
        "selection_source": "manager_plan" if plan is not None else "deterministic_recommendation",
        "prior_roster_identity": lifecycle_identity(target_roster),
        "result_roster_identity": lifecycle_identity(result),
        "renewal_limit": MAX_RENEWALS,
        "renewed_player_ids": sorted(renewals & set(target_players)),
        "career_updates": updates,
        "exits": exits,
        "academy_promotions": promotions,
        "summary": {
            "renewed": sum(item["status"] == "renewed" for item in updates),
            "continued": sum(item["status"] == "continued" for item in updates),
            "new_signings": sum(item["status"] == "new_signing" for item in updates),
            "contract_releases": sum(item["reason"] == "contract_released" for item in exits),
            "retirements": sum(item["reason"] == "retired" for item in exits),
            "academy_promotions": len(promotions),
        },
        "claim_boundary": (
            "bounded fictional contracts, retirement ages, and academy generation; "
            "not real employment, medical, valuation, or potential evidence"
        ),
    }
    return transaction, result


def apply_lifecycle_transaction(
    roster: Mapping[str, Any], transaction: Mapping[str, Any],
) -> dict[str, Any]:
    validate_lifecycle_transaction(transaction)
    if (
        not isinstance(transaction, Mapping)
        or transaction.get("schema_version") != 1
        or transaction.get("type") != "season_lifecycle"
        or lifecycle_identity(roster) != transaction.get("prior_roster_identity")
    ):
        raise ValueError("lifecycle transaction prior roster mismatch")
    result = copy.deepcopy(dict(roster))
    players = {
        _clean_player_id(player.get("player_id")): player
        for player in result.get("players") or [] if isinstance(player, Mapping)
    }
    for exit_row in transaction.get("exits") or []:
        player_id = _clean_player_id(exit_row.get("player_id"))
        if player_id not in players:
            raise ValueError("lifecycle exit player is unavailable")
        del players[player_id]
    for update in transaction.get("career_updates") or []:
        player_id = _clean_player_id(update.get("player_id"))
        player = players.get(player_id)
        if player is None or player.get("career") != update.get("before"):
            if not (player is not None and player.get("career") is None and update.get("before") is not None):
                raise ValueError("lifecycle career evidence mismatch")
        player["career"] = copy.deepcopy(update.get("after"))
    for promotion in transaction.get("academy_promotions") or []:
        player_id = _clean_player_id(promotion.get("player_id"))
        if player_id in players:
            raise ValueError("lifecycle academy identity collision")
        players[player_id] = copy.deepcopy(dict(promotion))
    result["players"] = sorted(players.values(), key=lambda player: str(player.get("player_id") or ""))
    result["squad_size"] = len(players)
    result["source"] = str(result.get("source") or "unknown") + "+gfs_player_lifecycle_v1"
    if lifecycle_identity(result) != transaction.get("result_roster_identity"):
        raise ValueError("lifecycle transaction result roster mismatch")
    return result


def validate_lifecycle_transaction(transaction: Mapping[str, Any]) -> None:
    if not isinstance(transaction, Mapping):
        raise ValueError("invalid lifecycle transaction")
    target = _SEASON.fullmatch(str(transaction.get("target_season_id") or ""))
    source = _SEASON.fullmatch(str(transaction.get("source_season_id") or ""))
    identities = (
        transaction.get("preview_identity"), transaction.get("prior_roster_identity"),
        transaction.get("result_roster_identity"),
    )
    updates = transaction.get("career_updates")
    exits = transaction.get("exits")
    promotions = transaction.get("academy_promotions")
    renewed = transaction.get("renewed_player_ids")
    if (
        transaction.get("schema_version") != 1
        or transaction.get("type") != "season_lifecycle"
        or target is None or source is None
        or int(target.group(1)) != int(source.group(1)) + 1
        or not str(transaction.get("team") or "").strip()
        or transaction.get("control") not in {"manager", "ai"}
        or transaction.get("selection_source") not in {
            "manager_plan", "deterministic_recommendation",
        }
        or transaction.get("renewal_limit") != MAX_RENEWALS
        or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in identities)
        or not isinstance(updates, list) or not isinstance(exits, list)
        or not isinstance(promotions, list) or not isinstance(renewed, list)
        or renewed != sorted(set(renewed)) or len(renewed) > MAX_RENEWALS
    ):
        raise ValueError("invalid lifecycle transaction")
    if transaction.get("plan") is not None:
        plan = RetentionPlan.from_payload(transaction["plan"])
        if (
            transaction.get("selection_source") != "manager_plan"
            or plan.cycle_id != transaction.get("cycle_id")
        ):
            raise ValueError("invalid lifecycle retention plan")
    elif transaction.get("selection_source") != "deterministic_recommendation":
        raise ValueError("invalid lifecycle selection source")
    update_ids = []
    statuses = []
    for update in updates:
        if not isinstance(update, Mapping):
            raise ValueError("invalid lifecycle career update")
        player_id = _clean_player_id(update.get("player_id"))
        status = update.get("status")
        if status not in {"renewed", "continued", "new_signing"}:
            raise ValueError("invalid lifecycle career status")
        after = update.get("after")
        before = update.get("before")
        if before is not None:
            if (
                not isinstance(before, Mapping) or before.get("schema_version") != 1
                or isinstance(before.get("contract_years_remaining"), bool)
                or not isinstance(before.get("contract_years_remaining"), int)
                or not 1 <= before["contract_years_remaining"] <= 5
                or before.get("contract_source") not in {
                    "deterministic_initialization", "renewed", "new_signing", "academy",
                }
            ):
                raise ValueError("invalid lifecycle prior career")
        if (
            not isinstance(after, Mapping) or after.get("schema_version") != 1
            or isinstance(after.get("contract_years_remaining"), bool)
            or not isinstance(after.get("contract_years_remaining"), int)
            or not 1 <= after["contract_years_remaining"] <= 5
            or after.get("contract_source") not in {
                "deterministic_initialization", "renewed", "new_signing", "academy",
            }
        ):
            raise ValueError("invalid lifecycle career result")
        if (
            status == "renewed" and (
                before is None or before["contract_years_remaining"] != 1
                or after["contract_years_remaining"] != RENEWAL_TERM
                or after["contract_source"] != "renewed"
            )
        ) or (
            status == "continued" and (
                before is None
                or after["contract_years_remaining"]
                != before["contract_years_remaining"] - 1
                or after["contract_source"] != before["contract_source"]
            )
        ) or (
            status == "new_signing" and (
                after["contract_years_remaining"] != NEW_SIGNING_TERM
                or after["contract_source"] != "new_signing"
            )
        ):
            raise ValueError("invalid lifecycle contract transition")
        update_ids.append(player_id)
        statuses.append(status)
    exit_ids = []
    reasons = []
    for exit_row in exits:
        if not isinstance(exit_row, Mapping):
            raise ValueError("invalid lifecycle exit")
        exit_ids.append(_clean_player_id(exit_row.get("player_id")))
        reason = exit_row.get("reason")
        if reason not in {"retired", "contract_released"}:
            raise ValueError("invalid lifecycle exit reason")
        player = exit_row.get("player")
        if (
            not isinstance(player, Mapping)
            or player.get("player_id") != exit_row.get("player_id")
            or str(player.get("name") or exit_row.get("player_id"))
            != exit_row.get("name")
            or str(player.get("role") or "") != exit_row.get("role")
            or player.get("age") != exit_row.get("age")
        ):
            raise ValueError("invalid lifecycle exit player evidence")
        reasons.append(reason)
    promotion_ids = []
    for slot, player in enumerate(promotions, start=1):
        if (
            not isinstance(player, Mapping)
            or player.get("source") != "gfs_fictional_academy_v1"
            or player.get("age") != 17
            or (player.get("career") or {}).get("contract_source") != "academy"
            or (player.get("career") or {}).get("contract_years_remaining") != ACADEMY_TERM
        ):
            raise ValueError("invalid lifecycle academy promotion")
        if slot > len(exits) or player != _academy_player(
            team=str(transaction.get("team")),
            target_season_id=str(transaction.get("target_season_id")),
            role=str(exits[slot - 1].get("role") or ""), slot=slot,
        ):
            raise ValueError("lifecycle academy source replay mismatch")
        promotion_ids.append(_clean_player_id(player.get("player_id")))
    if (
        update_ids != sorted(set(update_ids))
        or exit_ids != sorted(set(exit_ids))
        or len(set(promotion_ids)) != len(promotion_ids)
        or set(update_ids) & set(exit_ids)
        or set(promotion_ids) & (set(update_ids) | set(exit_ids))
        or len(promotions) != len(exits)
        or renewed != sorted(
            update["player_id"] for update in updates
            if update.get("status") == "renewed"
        )
    ):
        raise ValueError("invalid lifecycle player coverage")
    expected_summary = {
        "renewed": statuses.count("renewed"),
        "continued": statuses.count("continued"),
        "new_signings": statuses.count("new_signing"),
        "contract_releases": reasons.count("contract_released"),
        "retirements": reasons.count("retired"),
        "academy_promotions": len(promotions),
    }
    if transaction.get("summary") != expected_summary:
        raise ValueError("lifecycle transaction summary mismatch")


def validate_lifecycle_registry(registry: Mapping[str, Any] | None) -> dict[str, int]:
    if registry is None:
        return {"schema_version": 1, "club_count": 0, "transaction_count": 0}
    if not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1:
        raise ValueError("invalid lifecycle registry")
    clubs = registry.get("clubs") or {}
    if not isinstance(clubs, Mapping) or len(clubs) > 64:
        raise ValueError("invalid lifecycle registry clubs")
    count = 0
    for raw_team, club in clubs.items():
        team = str(raw_team)
        transactions = club.get("transactions") if isinstance(club, Mapping) else None
        if raw_team != team or not team.strip() or not isinstance(transactions, list) or len(transactions) > 64:
            raise ValueError("invalid lifecycle registry club")
        previous = 0
        for transaction in transactions:
            validate_lifecycle_transaction(transaction)
            match = _SEASON.fullmatch(str(transaction.get("target_season_id") or ""))
            if transaction.get("team") != team or int(match.group(1)) <= previous:
                raise ValueError("lifecycle registry seasons must be strictly increasing")
            previous = int(match.group(1))
        count += len(transactions)
    return {"schema_version": 1, "club_count": len(clubs), "transaction_count": count}


def append_lifecycle_transaction(
    registry: Mapping[str, Any] | None, *, team: str,
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(registry or {"schema_version": 1, "clubs": {}}))
    validate_lifecycle_registry(output)
    clubs = output.setdefault("clubs", {})
    club = clubs.setdefault(team, {"transactions": []})
    if transaction.get("team") != team:
        raise ValueError("lifecycle transaction team mismatch")
    if any(item.get("target_season_id") == transaction.get("target_season_id") for item in club["transactions"]):
        raise ValueError("lifecycle transaction already exists")
    club["transactions"].append(copy.deepcopy(dict(transaction)))
    validate_lifecycle_registry(output)
    return output


def lifecycle_transactions_for_team(
    registry: Mapping[str, Any] | None, team: str,
) -> list[dict[str, Any]]:
    validate_lifecycle_registry(registry)
    club = ((registry or {}).get("clubs") or {}).get(team) or {}
    return [copy.deepcopy(item) for item in club.get("transactions") or []]


def lifecycle_view(registry: Mapping[str, Any] | None) -> dict[str, Any]:
    summary = validate_lifecycle_registry(registry)
    latest = []
    for team, club in sorted(((registry or {}).get("clubs") or {}).items()):
        transactions = club.get("transactions") or []
        if transactions:
            latest.append(copy.deepcopy(transactions[-1]))
    return {
        **summary, "available": bool(latest), "latest_transactions": latest,
        "claim_boundary": (
            "bounded fictional lifecycle mechanics; not employment, medical, "
            "scouting, valuation, or potential evidence"
        ),
    }


__all__ = [
    "ACADEMY_TERM", "MAX_RENEWALS", "RetentionPlan",
    "append_lifecycle_transaction", "apply_lifecycle_transaction",
    "build_lifecycle_preview",
    "build_lifecycle_transaction", "lifecycle_cycle_id", "lifecycle_identity",
    "lifecycle_transactions_for_team", "lifecycle_view",
    "retirement_age_for_player",
    "validate_lifecycle_registry", "validate_lifecycle_transaction",
]
