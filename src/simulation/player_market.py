"""Identity-preserving global free-agent market for persistent careers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from src.simulation.scouting import scouting_observation


MARKET_SCHEMA_VERSION = 1
_SEASON = re.compile(r"^season-(\d{4,})$")
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def market_identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _player_id(value: Any) -> str:
    result = str(value or "").strip()
    if _ID.fullmatch(result) is None:
        raise ValueError("invalid free-agent player identity")
    return result


def _quality(player: Mapping[str, Any]) -> float:
    abilities = player.get("abilities")
    values = [
        float(value) for value in (abilities or {}).values()
        if not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ] if isinstance(abilities, Mapping) else []
    return round(sum(values) / len(values), 6) if values else 0.5


def market_player_quality(player: Mapping[str, Any]) -> float:
    return _quality(player)


@dataclass(frozen=True)
class FreeAgentPlan:
    market_id: str
    free_agent_id: str
    outgoing_player_id: str

    def __post_init__(self) -> None:
        if not str(self.market_id or "").startswith("free-market-"):
            raise ValueError("invalid free-agent market identity")
        object.__setattr__(self, "free_agent_id", _player_id(self.free_agent_id))
        object.__setattr__(
            self, "outgoing_player_id", _player_id(self.outgoing_player_id),
        )
        if self.free_agent_id == self.outgoing_player_id:
            raise ValueError("free-agent signing identities must differ")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FreeAgentPlan":
        if not isinstance(payload, Mapping) or payload.get("schema_version", 1) != 1:
            raise ValueError("free-agent plan must be a versioned object")
        return cls(
            str(payload.get("market_id") or ""),
            str(payload.get("free_agent_id") or ""),
            str(payload.get("outgoing_player_id") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "market_id": self.market_id,
            "free_agent_id": self.free_agent_id,
            "outgoing_player_id": self.outgoing_player_id,
        }


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": 1, "transitions": []}


def _validate_entry(entry: Mapping[str, Any]) -> None:
    player = entry.get("player")
    entered = _SEASON.fullmatch(str(entry.get("entered_season_id") or ""))
    available = _SEASON.fullmatch(str(entry.get("available_from_season_id") or ""))
    if (
        not isinstance(entry, Mapping) or not isinstance(player, Mapping)
        or _player_id(entry.get("player_id")) != player.get("player_id")
        or entered is None or available is None
        or int(available.group(1)) != int(entered.group(1)) + 1
        or not str(entry.get("origin_team") or "").strip()
        or entry.get("entry_reason") not in {
            "contract_released", "squad_replacement",
        }
        or isinstance(entry.get("retirement_age"), bool)
        or not isinstance(entry.get("retirement_age"), int)
        or not 36 <= entry["retirement_age"] <= 39
    ):
        raise ValueError("invalid free-agent pool entry")


def validate_market_registry(registry: Mapping[str, Any] | None) -> dict[str, int]:
    if registry is None:
        return {"schema_version": 1, "transition_count": 0, "pool_size": 0}
    if not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1:
        raise ValueError("invalid player market registry")
    transitions = registry.get("transitions") or []
    if not isinstance(transitions, list) or len(transitions) > 64:
        raise ValueError("invalid player market transitions")
    pool: dict[str, dict[str, Any]] = {}
    previous_target = 0
    for transition in transitions:
        if not isinstance(transition, Mapping) or transition.get("schema_version") != 1:
            raise ValueError("invalid player market transition")
        source = _SEASON.fullmatch(str(transition.get("source_season_id") or ""))
        target = _SEASON.fullmatch(str(transition.get("target_season_id") or ""))
        if (
            source is None or target is None
            or int(target.group(1)) != int(source.group(1)) + 1
            or int(target.group(1)) <= previous_target
            or transition.get("prior_pool_identity")
            != market_identity(sorted(pool.values(), key=lambda item: item["player_id"]))
        ):
            raise ValueError("player market transition source mismatch")
        aged_pool, expected_aging, expected_retirements = _age_pool(
            pool, target_season_id=str(transition["target_season_id"]),
        )
        if (
            transition.get("aging") != expected_aging
            or transition.get("pool_retirements") != expected_retirements
        ):
            raise ValueError("player market aging replay mismatch")
        pool = aged_pool
        signing_ids = []
        for signing in transition.get("signings") or []:
            validate_market_signing_transaction(signing)
            player_id = signing["free_agent_id"]
            if player_id not in pool or signing.get("free_agent_entry") != pool[player_id]:
                raise ValueError("free-agent signing pool evidence mismatch")
            signing_ids.append(player_id)
            del pool[player_id]
        if len(signing_ids) != len(set(signing_ids)):
            raise ValueError("free agent cannot sign twice in one window")
        decisions = transition.get("ai_decisions") or []
        for decision in decisions:
            validate_ai_free_agent_decision(decision)
        if (
            not isinstance(decisions, list)
            or len({str(item.get("team") or "") for item in decisions if isinstance(item, Mapping)})
            != len(decisions)
            or sorted(
                [item.get("plan") for item in decisions
                if isinstance(item, Mapping) and item.get("plan") is not None
                ], key=lambda item: item["free_agent_id"])
            != sorted(
                [signing.get("plan") for signing in transition.get("signings") or []
                if signing.get("control") == "ai"
                ], key=lambda item: item["free_agent_id"])
        ):
            raise ValueError("invalid AI free-agent decision evidence")
        entries = transition.get("entries") or []
        entry_ids = []
        for entry in entries:
            _validate_entry(entry)
            player_id = entry["player_id"]
            if player_id in pool or player_id in entry_ids:
                raise ValueError("duplicate global free-agent identity")
            entry_ids.append(player_id)
            pool[player_id] = copy.deepcopy(dict(entry))
        canonical = sorted(pool.values(), key=lambda item: item["player_id"])
        if (
            transition.get("result_pool_identity") != market_identity(canonical)
            or transition.get("summary") != {
                "aged": len(expected_aging),
                "pool_retirements": len(expected_retirements),
                "signings": len(signing_ids),
                "entries": len(entries),
                "result_pool_size": len(canonical),
            }
        ):
            raise ValueError("player market transition result mismatch")
        previous_target = int(target.group(1))
    return {
        "schema_version": 1, "transition_count": len(transitions),
        "pool_size": len(pool),
    }


def replay_market_pool(registry: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    validate_market_registry(registry)
    pool: dict[str, dict[str, Any]] = {}
    for transition in (registry or {}).get("transitions") or []:
        pool, _aging, _retirements = _age_pool(
            pool, target_season_id=str(transition["target_season_id"]),
        )
        for signing in transition.get("signings") or []:
            del pool[signing["free_agent_id"]]
        for entry in transition.get("entries") or []:
            pool[entry["player_id"]] = copy.deepcopy(dict(entry))
    return sorted(pool.values(), key=lambda item: item["player_id"])


def _age_pool(
    pool: Mapping[str, Mapping[str, Any]], *, target_season_id: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result = {}
    aging = []
    retirements = []
    for player_id, raw in sorted(pool.items()):
        entry = copy.deepcopy(dict(raw))
        player = entry["player"]
        age = player.get("age")
        if isinstance(age, int) and not isinstance(age, bool) and 16 <= age <= 45:
            after = age + 1
            if after >= entry["retirement_age"]:
                retirements.append({
                    "player_id": player_id, "name": str(player.get("name") or player_id),
                    "age_before": age, "age_after": after,
                    "retirement_age": entry["retirement_age"],
                })
                continue
            player["age"] = after
            aging.append({
                "player_id": player_id, "age_before": age, "age_after": after,
            })
        result[player_id] = entry
    return result, aging, retirements


def open_market_window(
    registry: Mapping[str, Any] | None, *, target_season_id: str,
) -> dict[str, Any]:
    validate_market_registry(registry)
    target = _SEASON.fullmatch(str(target_season_id))
    if target is None or int(target.group(1)) <= 1:
        raise ValueError("free-agent market requires a later season")
    source_season_id = f"season-{int(target.group(1)) - 1:04d}"
    prior = replay_market_pool(registry)
    pool_map = {entry["player_id"]: entry for entry in prior}
    aged, aging, retirements = _age_pool(
        pool_map, target_season_id=target_season_id,
    )
    return {
        "schema_version": 1,
        "source_season_id": source_season_id,
        "target_season_id": target_season_id,
        "prior_pool_identity": market_identity(prior),
        "aging": aging,
        "pool_retirements": retirements,
        "available": aged,
        "signings": [],
    }


def free_agent_market_id(window: Mapping[str, Any], *, team: str) -> str:
    team_hash = hashlib.sha256(team.encode("utf-8")).hexdigest()[:10]
    available = sorted(window.get("available", {}).values(), key=lambda item: item["player_id"])
    return (
        f"free-market-{str(window['target_season_id']).split('-')[-1]}-"
        f"{team_hash}-{market_identity(available)[:12]}"
    )


def market_preview_from_window(window: Mapping[str, Any], *, team: str) -> dict[str, Any]:
    candidates = []
    target_index = int(str(window["target_season_id"]).split("-")[-1])
    for entry in sorted(window.get("available", {}).values(), key=lambda item: item["player_id"]):
        available_index = int(str(entry["available_from_season_id"]).split("-")[-1])
        if available_index > target_index:
            continue
        player = entry["player"]
        observation = scouting_observation(
            market_id=free_agent_market_id(window, team=team), team=team,
            player_id=entry["player_id"], true_quality=_quality(player),
            level="baseline",
        )
        candidates.append({
            "player_id": entry["player_id"], "name": str(player.get("name") or entry["player_id"]),
            "role": str(player.get("role") or ""), "age": player.get("age"),
            "origin_team": entry["origin_team"],
            "seasons_available": target_index - available_index + 1,
            "observation": observation,
            "evidence_quality": "bounded_baseline_observation",
        })
    return {
        "schema_version": 1, "market_id": free_agent_market_id(window, team=team),
        "team": team, "target_season_id": window["target_season_id"],
        "candidate_count": len(candidates), "candidates": candidates,
        "signing_limit": 1, "transfer_fee": 0,
        "claim_boundary": (
            "identity-preserving fictional free-agent pool; not real availability, "
            "employment, scouting, valuation, or transfer evidence"
        ),
    }


def execute_free_agent_signing(
    window: Mapping[str, Any], roster: Mapping[str, Any], *, team: str,
    plan: FreeAgentPlan, control: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if control not in {"manager", "ai"}:
        raise ValueError("invalid free-agent signing control")
    output_window = copy.deepcopy(dict(window))
    preview = market_preview_from_window(output_window, team=team)
    if plan.market_id != preview["market_id"]:
        raise ValueError("free-agent market identity mismatch")
    entry = (output_window.get("available") or {}).get(plan.free_agent_id)
    if entry is None:
        raise ValueError("free agent is no longer available")
    current = [copy.deepcopy(player) for player in roster.get("players") or []]
    outgoing = next((player for player in current if player.get("player_id") == plan.outgoing_player_id), None)
    incoming_source = entry["player"]
    if outgoing is None or outgoing.get("role") != incoming_source.get("role"):
        raise ValueError("free-agent signing requires an available same-role replacement")
    incoming = copy.deepcopy(dict(incoming_source))
    incoming["team_id"] = team
    incoming["career"] = {
        "schema_version": 1, "contract_years_remaining": 2,
        "contract_source": "new_signing",
    }
    incoming["source"] = "gfs_global_free_agent_v1"
    incoming["free_agent_origin_team"] = entry["origin_team"]
    result_players = [player for player in current if player.get("player_id") != plan.outgoing_player_id]
    result_players.append(incoming)
    result_players.sort(key=lambda player: str(player.get("player_id") or ""))
    result = copy.deepcopy(dict(roster))
    result["players"] = result_players
    result["squad_size"] = len(result_players)
    result["source"] = str(result.get("source") or "unknown") + "+gfs_global_free_agent_v1"
    if len(result_players) != len(current) or len({_player_id(p.get("player_id")) for p in result_players}) != len(current):
        raise ValueError("free-agent signing must preserve a unique fixed-size squad")
    if not any(player.get("role") == "GK" for player in result_players):
        raise ValueError("free-agent signing cannot remove goalkeeper coverage")
    transaction = {
        "schema_version": 1, "type": "free_agent_signing",
        "target_season_id": window["target_season_id"], "team": team,
        "control": control, "market_id": plan.market_id, "plan": plan.as_dict(),
        "prior_roster_identity": market_identity(roster),
        "result_roster_identity": market_identity(result),
        "free_agent_id": plan.free_agent_id,
        "free_agent_entry": copy.deepcopy(entry),
        "incoming": incoming, "outgoing": outgoing,
        "transfer_fee": 0,
    }
    del output_window["available"][plan.free_agent_id]
    output_window.setdefault("signings", []).append(copy.deepcopy(transaction))
    return output_window, result, transaction


def ai_free_agent_decision(
    window: Mapping[str, Any], roster: Mapping[str, Any], *, team: str,
    minimum_improvement: float = 0.015,
) -> dict[str, Any]:
    preview = market_preview_from_window(window, team=team)
    players = [player for player in roster.get("players") or [] if isinstance(player, Mapping)]
    baseline_options = []
    for candidate in preview["candidates"]:
        same_role = [player for player in players if player.get("role") == candidate["role"]]
        if not same_role:
            continue
        outgoing = min(same_role, key=lambda player: (_quality(player), str(player.get("player_id") or "")))
        improvement = round(
            candidate["observation"]["estimated_quality"] - _quality(outgoing), 6,
        )
        baseline_options.append({
            "free_agent_id": candidate["player_id"],
            "outgoing_player_id": str(outgoing.get("player_id") or ""),
            "role": candidate["role"], "quality_improvement": improvement,
            "observation": copy.deepcopy(candidate["observation"]),
        })
    scouting_targets = sorted(
        baseline_options,
        key=lambda item: (-item["quality_improvement"], item["free_agent_id"]),
    )[:2]
    scouting_ids = {item["free_agent_id"] for item in scouting_targets}
    scouting_reports = []
    options = []
    for option in baseline_options:
        updated = copy.deepcopy(option)
        if option["free_agent_id"] in scouting_ids:
            entry = window["available"][option["free_agent_id"]]
            observation = scouting_observation(
                market_id=preview["market_id"], team=team,
                player_id=option["free_agent_id"],
                true_quality=_quality(entry["player"]), level="scouted",
            )
            outgoing = next(
                player for player in players
                if player.get("player_id") == option["outgoing_player_id"]
            )
            updated["observation"] = observation
            updated["quality_improvement"] = round(
                observation["estimated_quality"] - _quality(outgoing), 6,
            )
            scouting_reports.append({
                "player_id": option["free_agent_id"],
                "observation": copy.deepcopy(observation),
            })
        options.append(updated)
    eligible = [item for item in options if item["quality_improvement"] >= minimum_improvement]
    eligible.sort(key=lambda item: (-item["quality_improvement"], item["free_agent_id"], item["outgoing_player_id"]))
    selected = eligible[0] if eligible else None
    return {
        "schema_version": 1, "team": team, "market_id": preview["market_id"],
        "minimum_quality_improvement": minimum_improvement,
        "scouting_budget": 2, "scouting_reports": scouting_reports,
        "evaluated_options": options, "selected": copy.deepcopy(selected),
        "plan": (
            FreeAgentPlan(
                preview["market_id"], selected["free_agent_id"],
                selected["outgoing_player_id"],
            ).as_dict() if selected is not None else None
        ),
        "reason": (
            "best_affordable_same_role_upgrade" if selected is not None
            else "no_remaining_same_role_upgrade_meets_threshold"
        ),
    }


def validate_market_signing_transaction(transaction: Mapping[str, Any]) -> None:
    if (
        not isinstance(transaction, Mapping)
        or transaction.get("schema_version") != 1
        or transaction.get("type") != "free_agent_signing"
        or _SEASON.fullmatch(str(transaction.get("target_season_id") or "")) is None
        or transaction.get("control") not in {"manager", "ai"}
        or transaction.get("transfer_fee") != 0
    ):
        raise ValueError("invalid free-agent signing transaction")
    plan = FreeAgentPlan.from_payload(transaction.get("plan") or {})
    entry = transaction.get("free_agent_entry")
    incoming = transaction.get("incoming")
    outgoing = transaction.get("outgoing")
    _validate_entry(entry)
    expected_incoming = copy.deepcopy(dict(entry["player"]))
    expected_incoming["team_id"] = transaction.get("team")
    expected_incoming["career"] = {
        "schema_version": 1, "contract_years_remaining": 2,
        "contract_source": "new_signing",
    }
    expected_incoming["source"] = "gfs_global_free_agent_v1"
    expected_incoming["free_agent_origin_team"] = entry["origin_team"]
    if (
        plan.market_id != transaction.get("market_id")
        or plan.free_agent_id != transaction.get("free_agent_id")
        or entry.get("player_id") != plan.free_agent_id
        or not isinstance(incoming, Mapping) or not isinstance(outgoing, Mapping)
        or incoming.get("player_id") != plan.free_agent_id
        or outgoing.get("player_id") != plan.outgoing_player_id
        or incoming.get("role") != outgoing.get("role")
        or incoming.get("source") != "gfs_global_free_agent_v1"
        or incoming.get("free_agent_origin_team") != entry.get("origin_team")
        or dict(incoming) != expected_incoming
        or any(
            not isinstance(transaction.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", transaction[field]) is None
            for field in ("prior_roster_identity", "result_roster_identity")
        )
    ):
        raise ValueError("free-agent signing evidence mismatch")


def validate_ai_free_agent_decision(decision: Mapping[str, Any]) -> None:
    if (
        not isinstance(decision, Mapping) or decision.get("schema_version") != 1
        or not str(decision.get("team") or "").strip()
        or not str(decision.get("market_id") or "").startswith("free-market-")
        or decision.get("minimum_quality_improvement") != 0.015
        or decision.get("scouting_budget") != 2
        or not isinstance(decision.get("scouting_reports"), list)
        or len(decision["scouting_reports"]) > 2
        or not isinstance(decision.get("evaluated_options"), list)
    ):
        raise ValueError("invalid AI free-agent decision")
    options = decision["evaluated_options"]
    identities = []
    option_by_player = {}
    for option in options:
        if (
            not isinstance(option, Mapping)
            or not math.isfinite(float(option.get("quality_improvement", math.nan)))
            or not str(option.get("role") or "").strip()
            or not isinstance(option.get("observation"), Mapping)
            or option["observation"].get("level") not in {"baseline", "scouted"}
        ):
            raise ValueError("invalid AI free-agent option")
        identities.append((
            _player_id(option.get("free_agent_id")),
            _player_id(option.get("outgoing_player_id")),
        ))
        option_by_player[option["free_agent_id"]] = option
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate AI free-agent option")
    report_ids = []
    for report in decision["scouting_reports"]:
        observation = report.get("observation") if isinstance(report, Mapping) else None
        player_id = str(report.get("player_id") or "") if isinstance(report, Mapping) else ""
        option = option_by_player.get(player_id)
        if (
            not isinstance(report, Mapping) or not player_id
            or not isinstance(observation, Mapping)
            or observation.get("level") != "scouted"
            or option is None or option.get("observation") != observation
        ):
            raise ValueError("AI scouting report evidence mismatch")
        report_ids.append(player_id)
    if (
        len(report_ids) != len(set(report_ids))
        or set(report_ids) != {
            str(option["free_agent_id"]) for option in options
            if option["observation"].get("level") == "scouted"
        }
        or len(report_ids) != min(2, len(options))
    ):
        raise ValueError("AI scouting budget replay mismatch")
    eligible = [
        item for item in options
        if float(item["quality_improvement"]) >= 0.015
    ]
    eligible.sort(key=lambda item: (
        -float(item["quality_improvement"]), item["free_agent_id"],
        item["outgoing_player_id"],
    ))
    expected = eligible[0] if eligible else None
    expected_plan = (
        FreeAgentPlan(
            str(decision["market_id"]), str(expected["free_agent_id"]),
            str(expected["outgoing_player_id"]),
        ).as_dict() if expected is not None else None
    )
    if (
        decision.get("selected") != expected
        or decision.get("plan") != expected_plan
        or decision.get("reason") != (
            "best_affordable_same_role_upgrade" if expected is not None
            else "no_remaining_same_role_upgrade_meets_threshold"
        )
    ):
        raise ValueError("AI free-agent decision replay mismatch")


def apply_market_signing_transaction(
    roster: Mapping[str, Any], transaction: Mapping[str, Any],
) -> dict[str, Any]:
    validate_market_signing_transaction(transaction)
    if market_identity(roster) != transaction.get("prior_roster_identity"):
        raise ValueError("free-agent signing prior roster mismatch")
    result = copy.deepcopy(dict(roster))
    players = [copy.deepcopy(player) for player in result.get("players") or []]
    outgoing_id = transaction["outgoing"]["player_id"]
    matches = [player for player in players if player.get("player_id") == outgoing_id]
    if len(matches) != 1 or matches[0] != transaction["outgoing"]:
        raise ValueError("free-agent outgoing player evidence mismatch")
    players = [player for player in players if player.get("player_id") != outgoing_id]
    players.append(copy.deepcopy(transaction["incoming"]))
    players.sort(key=lambda player: str(player.get("player_id") or ""))
    result["players"] = players
    result["squad_size"] = len(players)
    result["source"] = str(result.get("source") or "unknown") + "+gfs_global_free_agent_v1"
    if market_identity(result) != transaction.get("result_roster_identity"):
        raise ValueError("free-agent signing result roster mismatch")
    return result


def pool_entry_from_player(
    player: Mapping[str, Any], *, origin_team: str, entered_season_id: str,
    retirement_age: int, reason: str,
) -> dict[str, Any]:
    match = _SEASON.fullmatch(str(entered_season_id))
    if match is None:
        raise ValueError("invalid free-agent entry season")
    frozen = copy.deepcopy(dict(player))
    frozen.pop("team_id", None)
    entry = {
        "schema_version": 1,
        "player_id": _player_id(frozen.get("player_id")),
        "player": frozen, "origin_team": origin_team,
        "entered_season_id": entered_season_id,
        "available_from_season_id": f"season-{int(match.group(1)) + 1:04d}",
        "retirement_age": retirement_age, "entry_reason": reason,
    }
    _validate_entry(entry)
    return entry


def finalize_market_transition(
    registry: Mapping[str, Any] | None, window: Mapping[str, Any], *,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(dict(registry or _empty_registry()))
    available = copy.deepcopy(dict(window.get("available") or {}))
    frozen_entries = []
    for raw in entries:
        entry = copy.deepcopy(dict(raw))
        _validate_entry(entry)
        if entry["player_id"] in available:
            raise ValueError("duplicate free-agent transition entry")
        available[entry["player_id"]] = entry
        frozen_entries.append(entry)
    canonical = sorted(available.values(), key=lambda item: item["player_id"])
    transition = {
        "schema_version": 1,
        "source_season_id": window["source_season_id"],
        "target_season_id": window["target_season_id"],
        "prior_pool_identity": window["prior_pool_identity"],
        "aging": copy.deepcopy(window.get("aging") or []),
        "pool_retirements": copy.deepcopy(window.get("pool_retirements") or []),
        "signings": copy.deepcopy(window.get("signings") or []),
        "ai_decisions": copy.deepcopy(window.get("ai_decisions") or []),
        "entries": frozen_entries,
        "result_pool_identity": market_identity(canonical),
        "summary": {
            "aged": len(window.get("aging") or []),
            "pool_retirements": len(window.get("pool_retirements") or []),
            "signings": len(window.get("signings") or []),
            "entries": len(frozen_entries), "result_pool_size": len(canonical),
        },
        "claim_boundary": (
            "replayable fictional cross-club labor market; not real contract, "
            "availability, scouting, valuation, or transfer evidence"
        ),
    }
    output.setdefault("transitions", []).append(transition)
    validate_market_registry(output)
    return output, transition


def market_signings_for_team(
    registry: Mapping[str, Any] | None, team: str,
) -> list[dict[str, Any]]:
    validate_market_registry(registry)
    return [
        copy.deepcopy(signing)
        for transition in (registry or {}).get("transitions") or []
        for signing in transition.get("signings") or []
        if signing.get("team") == team
    ]


def market_view(registry: Mapping[str, Any] | None) -> dict[str, Any]:
    summary = validate_market_registry(registry)
    transitions = (registry or {}).get("transitions") or []
    pool = replay_market_pool(registry)
    public_pool = [
        {
            "player_id": entry["player_id"],
            "name": str(entry["player"].get("name") or entry["player_id"]),
            "role": str(entry["player"].get("role") or ""),
            "age": entry["player"].get("age"),
            "origin_team": entry["origin_team"],
            "entered_season_id": entry["entered_season_id"],
            "available_from_season_id": entry["available_from_season_id"],
        }
        for entry in pool
    ]
    latest = transitions[-1] if transitions else None
    public_transition = None
    if latest is not None:
        public_transition = {
            "schema_version": latest["schema_version"],
            "source_season_id": latest["source_season_id"],
            "target_season_id": latest["target_season_id"],
            "summary": copy.deepcopy(latest["summary"]),
            "signings": [
                {
                    "team": signing["team"], "control": signing["control"],
                    "free_agent_id": signing["free_agent_id"],
                    "origin_team": signing["free_agent_entry"]["origin_team"],
                    "free_agent_entry": {
                        "origin_team": signing["free_agent_entry"]["origin_team"],
                    },
                    "incoming": {
                        "player_id": signing["incoming"]["player_id"],
                        "name": signing["incoming"].get("name"),
                        "role": signing["incoming"].get("role"),
                        "age": signing["incoming"].get("age"),
                    },
                    "outgoing": {
                        "player_id": signing["outgoing"]["player_id"],
                        "name": signing["outgoing"].get("name"),
                        "role": signing["outgoing"].get("role"),
                        "age": signing["outgoing"].get("age"),
                    },
                }
                for signing in latest.get("signings") or []
            ],
            "entries": [
                {
                    "player_id": entry["player_id"],
                    "player": {
                        "name": entry["player"].get("name"),
                        "role": entry["player"].get("role"),
                        "age": entry["player"].get("age"),
                    },
                    "origin_team": entry["origin_team"],
                    "available_from_season_id": entry["available_from_season_id"],
                    "entry_reason": entry["entry_reason"],
                }
                for entry in latest.get("entries") or []
            ],
            "claim_boundary": latest["claim_boundary"],
        }
    return {
        **summary, "available": bool(transitions), "pool": public_pool,
        "latest_transition": public_transition,
        "claim_boundary": (
            "fictional identity-preserving free-agent market; not real employment, "
            "availability, scouting, valuation, or transfer evidence"
        ),
    }


__all__ = [
    "FreeAgentPlan", "ai_free_agent_decision",
    "apply_market_signing_transaction",
    "execute_free_agent_signing", "finalize_market_transition",
    "free_agent_market_id", "market_identity", "market_preview_from_window",
    "market_player_quality",
    "market_signings_for_team", "market_view", "open_market_window",
    "pool_entry_from_player", "replay_market_pool", "validate_market_registry",
    "validate_market_signing_transaction",
    "validate_ai_free_agent_decision",
]
