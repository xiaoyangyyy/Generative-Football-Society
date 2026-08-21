"""Deterministic, replayable squad-building transactions for product careers."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from src.match_engine.formation import roles_for_formation
from src.match_engine.tactical_catalog import normalize_formation_key


RECRUITMENT_SCHEMA_VERSION = 1
RECRUITMENT_BUDGET = 8
RECRUITMENT_MAX_MOVES = 2
RECRUITMENT_CANDIDATE_COUNT = 5
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_ROLE_FALLBACK = ("GK", "CB", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST")
_SEASON_ID_PATTERN = re.compile(r"^season-(\d{4,})$")


def _clean_id(value: Any, field: str) -> str:
    cleaned = str(value or "").strip()
    if not _ID_PATTERN.fullmatch(cleaned):
        raise ValueError(f"invalid recruitment {field}")
    return cleaned


@dataclass(frozen=True)
class RecruitmentMove:
    candidate_id: str
    outgoing_player_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _clean_id(self.candidate_id, "candidate id"))
        object.__setattr__(
            self, "outgoing_player_id",
            _clean_id(self.outgoing_player_id, "outgoing player id"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RecruitmentMove":
        if not isinstance(payload, Mapping):
            raise ValueError("recruitment move must be an object")
        return cls(
            candidate_id=payload.get("candidate_id", ""),
            outgoing_player_id=payload.get("outgoing_player_id", ""),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "outgoing_player_id": self.outgoing_player_id,
        }


@dataclass(frozen=True)
class RecruitmentPlan:
    market_id: str
    moves: tuple[RecruitmentMove, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "market_id", _clean_id(self.market_id, "market id"))
        moves = tuple(self.moves)
        if not 1 <= len(moves) <= RECRUITMENT_MAX_MOVES:
            raise ValueError(
                f"recruitment requires 1-{RECRUITMENT_MAX_MOVES} moves"
            )
        if any(not isinstance(move, RecruitmentMove) for move in moves):
            raise ValueError("recruitment moves must be RecruitmentMove values")
        if len({move.candidate_id for move in moves}) != len(moves):
            raise ValueError("recruitment candidates must be unique")
        if len({move.outgoing_player_id for move in moves}) != len(moves):
            raise ValueError("outgoing players must be unique")
        object.__setattr__(self, "moves", moves)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RecruitmentPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("recruitment plan must be an object")
        if payload.get("schema_version", 1) != RECRUITMENT_SCHEMA_VERSION:
            raise ValueError("unsupported recruitment plan schema")
        raw_moves = payload.get("moves")
        if not isinstance(raw_moves, Sequence) or isinstance(raw_moves, (str, bytes)):
            raise ValueError("recruitment moves must be an array")
        return cls(
            market_id=payload.get("market_id", ""),
            moves=tuple(RecruitmentMove.from_payload(move) for move in raw_moves),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECRUITMENT_SCHEMA_VERSION,
            "market_id": self.market_id,
            "moves": [move.as_dict() for move in self.moves],
            "budget": RECRUITMENT_BUDGET,
            "claim_boundary": (
                "fictional deterministic squad-building market for the simulated "
                "career; not a real transfer valuation"
            ),
        }


def roster_identity(roster: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(roster), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recruitment_market_id(*, season_index: int, team: str) -> str:
    if isinstance(season_index, bool) or not isinstance(season_index, int) or season_index < 2:
        raise ValueError("recruitment market requires a later season index")
    team_digest = hashlib.sha256(str(team).encode("utf-8")).hexdigest()[:12]
    return f"market-{season_index:04d}-{team_digest}"


def squad_player_quality(player: Mapping[str, Any]) -> float:
    abilities = player.get("abilities") or {}
    values = [
        float(abilities.get(key, 0.5))
        for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "shot", "power", "aerial", "mental",
        )
    ]
    if str(player.get("role", "")) == "GK":
        values.extend([
            float(abilities.get("gk_reflex", abilities.get("gk", 0.5))),
            float(abilities.get("gk_aerial", abilities.get("gk", 0.5))),
        ] * 2)
    return sum(values) / max(1, len(values))


def _target_roles(roster: Mapping[str, Any]) -> list[str]:
    formation = normalize_formation_key(str(roster.get("formation") or "4-3-3"))
    required = list(dict.fromkeys(roles_for_formation(formation)))
    players = [player for player in roster.get("players") or [] if isinstance(player, Mapping)]
    ranked = sorted(
        required,
        key=lambda role: (
            max(
                (squad_player_quality(player) for player in players if player.get("role") == role),
                default=-1.0,
            ),
            role,
        ),
    )
    for role in _ROLE_FALLBACK:
        if role not in ranked:
            ranked.append(role)
    return ranked[:RECRUITMENT_CANDIDATE_COUNT]


def _unit(digest: bytes, index: int) -> float:
    return digest[index % len(digest)] / 255.0


def _candidate(
    *, team: str, market_id: str, roster_hash: str, role: str,
    index: int, cost: int,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{team}|{market_id}|{roster_hash}|{role}|{index}".encode("utf-8")
    ).digest()
    base = 0.545 + 0.034 * cost + 0.012 * (_unit(digest, 0) - 0.5)
    ability_keys = (
        "tech", "pass_skill", "vision", "spatial", "pace", "press",
        "curve", "shot", "power", "aerial", "heading", "mental",
        "gk_reflex", "gk_aerial",
    )
    abilities = {}
    for offset, key in enumerate(ability_keys, 1):
        value = base + 0.045 * (_unit(digest, offset) - 0.5)
        if role == "GK":
            value += 0.095 if key in {"gk_reflex", "gk_aerial", "mental"} else -0.025
        elif key in {"gk_reflex", "gk_aerial"}:
            value = 0.18 + 0.04 * _unit(digest, offset)
        elif role in {"CB", "LB", "RB", "DM"} and key in {"press", "power", "aerial", "mental"}:
            value += 0.045
        elif role in {"AM", "CM", "LW", "RW"} and key in {"tech", "pass_skill", "vision", "spatial"}:
            value += 0.045
        elif role == "ST" and key in {"shot", "power", "pace", "heading"}:
            value += 0.055
        abilities[key] = round(min(0.88, max(0.16, value)), 6)
    suffix = digest.hex()[:8]
    player_id = f"gfs_{market_id.replace('-', '_')}_{index + 1}"
    condition_base = min(0.82, base + 0.02)
    return {
        "player_id": player_id,
        "name": f"GFS {role} Prospect {suffix}",
        "team_id": team,
        "role": role,
        "shirt_number": 70 + index,
        "age": 19 + digest[2] % 10,
        "club": team,
        "market_value_eur": float(cost * 5_000_000),
        "international_caps": 0,
        "squad_role": "bench",
        "source": "gfs_fictional_recruitment_v1",
        "recruitment_cost": cost,
        "condition": {
            "technical_quality": round(condition_base, 6),
            "physical_power": round(min(0.86, condition_base + 0.03 * (_unit(digest, 20) - 0.5)), 6),
            "composure": round(min(0.86, condition_base + 0.03 * (_unit(digest, 21) - 0.5)), 6),
            "pace_threat": round(min(0.86, condition_base + 0.03 * (_unit(digest, 22) - 0.5)), 6),
            "vision_playmaking": round(min(0.86, condition_base + 0.03 * (_unit(digest, 23) - 0.5)), 6),
            "defensive_intensity": round(min(0.86, condition_base + 0.03 * (_unit(digest, 24) - 0.5)), 6),
        },
        "abilities": abilities,
        "availability": 1.0,
    }


def generate_recruitment_market(
    roster: Mapping[str, Any], *, team: str, market_id: str,
) -> dict[str, Any]:
    if not isinstance(roster, Mapping) or str(roster.get("team_id") or team) != team:
        raise ValueError("recruitment roster identity mismatch")
    _clean_id(market_id, "market id")
    players = roster.get("players")
    if not isinstance(players, list) or len(players) < 11:
        raise ValueError("recruitment requires a valid base squad")
    roster_hash = roster_identity(roster)
    costs = (3, 3, 4, 5, 6)
    candidates = [
        _candidate(
            team=team, market_id=market_id, roster_hash=roster_hash,
            role=role, index=index, cost=costs[index],
        )
        for index, role in enumerate(_target_roles(roster))
    ]
    for candidate in candidates:
        candidate["quality"] = round(squad_player_quality(candidate), 6)
    return {
        "schema_version": RECRUITMENT_SCHEMA_VERSION,
        "market_id": market_id,
        "team": team,
        "budget": RECRUITMENT_BUDGET,
        "maximum_moves": RECRUITMENT_MAX_MOVES,
        "roster_identity": roster_hash,
        "candidates": candidates,
        "claim_boundary": (
            "deterministic fictional candidates and game credits; no real player, "
            "price, scouting, or performance claim"
        ),
    }


def apply_recruitment_plan(
    roster: Mapping[str, Any], *, team: str, plan: RecruitmentPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    market = generate_recruitment_market(roster, team=team, market_id=plan.market_id)
    candidates = {candidate["player_id"]: candidate for candidate in market["candidates"]}
    current_players = [copy.deepcopy(player) for player in roster.get("players") or []]
    current_ids = {str(player.get("player_id") or "") for player in current_players}
    selected = []
    for move in plan.moves:
        candidate = candidates.get(move.candidate_id)
        if candidate is None:
            raise ValueError("recruitment candidate is outside the frozen market")
        if move.outgoing_player_id not in current_ids:
            raise ValueError("outgoing player is outside the current squad")
        selected.append((move, copy.deepcopy(candidate)))
    total_cost = sum(int(candidate["recruitment_cost"]) for _, candidate in selected)
    if total_cost > RECRUITMENT_BUDGET:
        raise ValueError("recruitment plan exceeds the fixed budget")
    outgoing = {move.outgoing_player_id for move, _ in selected}
    result_players = [
        player for player in current_players
        if str(player.get("player_id") or "") not in outgoing
    ]
    result_players.extend(candidate for _, candidate in selected)
    result_ids = [str(player.get("player_id") or "") for player in result_players]
    if (
        len(result_players) != len(current_players)
        or len(set(result_ids)) != len(result_ids)
        or any(not player_id for player_id in result_ids)
    ):
        raise ValueError("recruitment must preserve a unique fixed-size squad")
    if len(result_players) < 11 or not any(player.get("role") == "GK" for player in result_players):
        raise ValueError("recruitment cannot leave an invalid playable squad")
    result = copy.deepcopy(dict(roster))
    result["players"] = result_players
    result["squad_size"] = len(result_players)
    result["source"] = f"{roster.get('source', 'unknown')}+gfs_squad_registry_v1"
    transaction = {
        "schema_version": RECRUITMENT_SCHEMA_VERSION,
        "team": team,
        "market_id": plan.market_id,
        "plan": plan.as_dict(),
        "budget": RECRUITMENT_BUDGET,
        "spent": total_cost,
        "remaining": RECRUITMENT_BUDGET - total_cost,
        "prior_roster_identity": market["roster_identity"],
        "result_roster_identity": roster_identity(result),
        "moves": [
            {
                **move.as_dict(),
                "incoming": candidate,
                "outgoing": next(
                    copy.deepcopy(player) for player in current_players
                    if str(player.get("player_id") or "") == move.outgoing_player_id
                ),
            }
            for move, candidate in selected
        ],
    }
    return result, transaction


def replay_squad_registry(
    base_roster: Mapping[str, Any], *, team: str,
    registry: Mapping[str, Any] | None,
    development_registry: Mapping[str, Any] | None = None,
    lifecycle_registry: Mapping[str, Any] | None = None,
    market_registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        registry is not None
        and (
            not isinstance(registry, Mapping)
            or registry.get("schema_version", 1) != 1
        )
    ):
        raise ValueError("invalid squad registry")
    clubs = (registry or {}).get("clubs") or {}
    if not isinstance(clubs, Mapping):
        raise ValueError("invalid squad registry clubs")
    entry = clubs.get(team) or {"transactions": []}
    if not isinstance(entry, Mapping):
        raise ValueError("invalid squad registry club entry")
    raw_transactions = entry.get("transactions") or []
    if not isinstance(raw_transactions, list) or len(raw_transactions) > 64:
        raise ValueError("invalid squad registry transaction history")
    roster = copy.deepcopy(dict(base_roster))
    transactions = []
    from src.simulation.player_development import (
        apply_development_transaction, development_transactions_for_team,
    )
    from src.simulation.player_lifecycle import (
        apply_lifecycle_transaction, lifecycle_transactions_for_team,
    )
    from src.simulation.player_market import (
        apply_market_signing_transaction, market_signings_for_team,
    )
    development_transactions = development_transactions_for_team(
        development_registry, team,
    )
    lifecycle_transactions = lifecycle_transactions_for_team(
        lifecycle_registry, team,
    )
    market_transactions = market_signings_for_team(market_registry, team)
    evolution_events = []
    for phase, items, apply in (
        (1, market_transactions, apply_market_signing_transaction),
        (2, lifecycle_transactions, apply_lifecycle_transaction),
        (3, development_transactions, apply_development_transaction),
    ):
        for item in items:
            target = _SEASON_ID_PATTERN.fullmatch(
                str(item.get("target_season_id") or ""),
            )
            if target is None:
                raise ValueError("invalid squad evolution season")
            evolution_events.append((int(target.group(1)), phase, item, apply))
    evolution_events.sort(key=lambda event: (event[0], event[1]))
    evolution_index = 0
    seen_markets: set[str] = set()
    previous_season_index = 0
    for raw in raw_transactions:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
            raise ValueError("invalid squad registry transaction")
        plan = RecruitmentPlan.from_payload(raw.get("plan") or {})
        if plan.market_id in seen_markets:
            raise ValueError("duplicate squad registry market")
        seen_markets.add(plan.market_id)
        season_id = _clean_id(raw.get("season_id"), "season id")
        season_match = _SEASON_ID_PATTERN.fullmatch(season_id)
        if season_match is None or int(season_match.group(1)) <= previous_season_index:
            raise ValueError("squad registry seasons must be strictly increasing")
        season_index = int(season_match.group(1))
        while evolution_index < len(evolution_events):
            event_season, _phase, event, apply = evolution_events[evolution_index]
            if event_season >= season_index:
                break
            roster = apply(roster, event)
            evolution_index += 1
        result, expected = apply_recruitment_plan(roster, team=team, plan=plan)
        if set(raw) != set(expected) | {"season_id"}:
            raise ValueError("squad registry transaction fields mismatch")
        for field in (
            "team", "market_id", "budget", "spent", "remaining",
            "prior_roster_identity", "result_roster_identity", "moves",
        ):
            if raw.get(field) != expected.get(field):
                raise ValueError("squad registry transaction identity mismatch")
        previous_season_index = season_index
        transaction = {**expected, "season_id": season_id}
        roster = result
        transactions.append(transaction)
        while evolution_index < len(evolution_events):
            event_season, _phase, event, apply = evolution_events[evolution_index]
            if event_season != season_index:
                break
            roster = apply(roster, event)
            evolution_index += 1
    while evolution_index < len(evolution_events):
        _event_season, _phase, event, apply = evolution_events[evolution_index]
        roster = apply(roster, event)
        evolution_index += 1
    return roster, transactions


def replay_squad_registry_through(
    base_roster: Mapping[str, Any], *, team: str,
    registry: Mapping[str, Any] | None, season_id: str,
    development_registry: Mapping[str, Any] | None = None,
    lifecycle_registry: Mapping[str, Any] | None = None,
    market_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    match = _SEASON_ID_PATTERN.fullmatch(str(season_id))
    if match is None:
        raise ValueError("invalid squad registry season boundary")
    boundary = int(match.group(1))
    clubs = registry.get("clubs") if isinstance(registry, Mapping) else None
    entry = clubs.get(team) if isinstance(clubs, Mapping) else None
    selected = []
    for transaction in (
        entry.get("transactions") if isinstance(entry, Mapping) else []
    ) or []:
        tx_match = _SEASON_ID_PATTERN.fullmatch(str(transaction.get("season_id") or ""))
        if tx_match is None:
            raise ValueError("invalid squad registry transaction season")
        if int(tx_match.group(1)) <= boundary:
            selected.append(copy.deepcopy(transaction))
    filtered = {
        "schema_version": 1,
        "clubs": {team: {"transactions": selected}},
    }
    development_club = (
        ((development_registry or {}).get("clubs") or {}).get(team) or {}
        if isinstance(development_registry, Mapping) else {}
    )
    selected_development = []
    for transaction in development_club.get("transactions") or []:
        target_match = _SEASON_ID_PATTERN.fullmatch(
            str(transaction.get("target_season_id") or "")
        )
        if target_match is None:
            raise ValueError("invalid development transaction season")
        if int(target_match.group(1)) <= boundary:
            selected_development.append(copy.deepcopy(transaction))
    filtered_development = {
        "schema_version": 1,
        "clubs": {team: {"transactions": selected_development}},
    }
    lifecycle_club = (
        ((lifecycle_registry or {}).get("clubs") or {}).get(team) or {}
        if isinstance(lifecycle_registry, Mapping) else {}
    )
    selected_lifecycle = []
    for transaction in lifecycle_club.get("transactions") or []:
        target_match = _SEASON_ID_PATTERN.fullmatch(
            str(transaction.get("target_season_id") or "")
        )
        if target_match is None:
            raise ValueError("invalid lifecycle transaction season")
        if int(target_match.group(1)) <= boundary:
            selected_lifecycle.append(copy.deepcopy(transaction))
    filtered_lifecycle = {
        "schema_version": 1,
        "clubs": {team: {"transactions": selected_lifecycle}},
    }
    selected_market = []
    for transition in (market_registry or {}).get("transitions") or []:
        target_match = _SEASON_ID_PATTERN.fullmatch(
            str(transition.get("target_season_id") or "")
        )
        if target_match is None:
            raise ValueError("invalid player market transition season")
        if int(target_match.group(1)) <= boundary:
            selected_market.append(copy.deepcopy(transition))
    filtered_market = {"schema_version": 1, "transitions": selected_market}
    roster, _ = replay_squad_registry(
        base_roster, team=team, registry=filtered,
        development_registry=filtered_development,
        lifecycle_registry=filtered_lifecycle,
        market_registry=filtered_market,
    )
    return roster


def validate_squad_registry(
    base_dir: str | Path, registry: Mapping[str, Any] | None,
    development_registry: Mapping[str, Any] | None = None,
    lifecycle_registry: Mapping[str, Any] | None = None,
    market_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every registered club, not only the next fixture's teams."""
    from src.data_engine.roster_loader import load_roster_json, roster_path_for_team

    if (
        registry is None and development_registry is None
        and lifecycle_registry is None and market_registry is None
    ):
        return {"schema_version": 1, "club_count": 0, "transaction_count": 0}
    if (
        registry is not None
        and (
            not isinstance(registry, Mapping)
            or registry.get("schema_version", 1) != 1
        )
    ):
        raise ValueError("invalid squad registry")
    clubs = (registry or {}).get("clubs") or {}
    if not isinstance(clubs, Mapping) or len(clubs) > 64:
        raise ValueError("invalid squad registry clubs")
    root = Path(base_dir).resolve()
    transaction_count = 0
    development_clubs = (
        (development_registry or {}).get("clubs") or {}
        if isinstance(development_registry, Mapping) else {}
    )
    lifecycle_clubs = (
        (lifecycle_registry or {}).get("clubs") or {}
        if isinstance(lifecycle_registry, Mapping) else {}
    )
    market_clubs = {
        str(signing.get("team") or "")
        for transition in (market_registry or {}).get("transitions") or []
        for signing in transition.get("signings") or []
    } if isinstance(market_registry, Mapping) else set()
    all_clubs = set(clubs) | set(development_clubs) | set(lifecycle_clubs) | market_clubs
    for raw_team in sorted(all_clubs, key=str):
        team = str(raw_team)
        if not team.strip() or len(team) > 96 or raw_team != team:
            raise ValueError("invalid squad registry team identity")
        base = load_roster_json(roster_path_for_team(str(root), team))
        if base is None or str(base.get("team_id") or team) != team:
            raise ValueError("squad registry base roster is unavailable")
        _, transactions = replay_squad_registry(
            base, team=team, registry=registry,
            development_registry=development_registry,
            lifecycle_registry=lifecycle_registry,
            market_registry=market_registry,
        )
        transaction_count += len(transactions)
    return {
        "schema_version": 1,
        "club_count": len(all_clubs),
        "transaction_count": transaction_count,
    }


def append_recruitment_transaction(
    base_roster: Mapping[str, Any], *, team: str, plan: RecruitmentPlan,
    season_id: str, registry: Mapping[str, Any] | None,
    development_registry: Mapping[str, Any] | None = None,
    lifecycle_registry: Mapping[str, Any] | None = None,
    market_registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    current, transactions = replay_squad_registry(
        base_roster, team=team, registry=registry,
        development_registry=development_registry,
        lifecycle_registry=lifecycle_registry,
        market_registry=market_registry,
    )
    result, transaction = apply_recruitment_plan(current, team=team, plan=plan)
    transaction = {**transaction, "season_id": _clean_id(season_id, "season id")}
    new_registry = copy.deepcopy(dict(registry or {"schema_version": 1, "clubs": {}}))
    new_registry.setdefault("schema_version", 1)
    clubs = new_registry.setdefault("clubs", {})
    if not isinstance(clubs, dict):
        raise ValueError("invalid squad registry clubs")
    clubs[team] = {"transactions": [*transactions, transaction]}
    replayed, replayed_transactions = replay_squad_registry(
        base_roster, team=team, registry=new_registry,
        development_registry=development_registry,
        lifecycle_registry=lifecycle_registry,
        market_registry=market_registry,
    )
    if roster_identity(replayed) != roster_identity(result):
        raise ValueError("squad registry replay mismatch")
    return new_registry, result, replayed_transactions[-1]


def load_effective_roster(base_dir: str | Path, team: str) -> dict[str, Any] | None:
    from src.data_engine.roster_loader import load_roster_json, roster_path_for_team

    root = Path(base_dir).resolve()
    base = load_roster_json(roster_path_for_team(str(root), team))
    if base is None:
        return None
    session_path = root / "data/persistence/product_session.json"
    registry = None
    development_registry = None
    lifecycle_registry = None
    market_registry = None
    if session_path.is_file():
        if session_path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("product session is oversized")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        if not isinstance(session, Mapping):
            raise ValueError("invalid product session")
        registry = session.get("squad_registry")
        development_registry = session.get("player_development")
        lifecycle_registry = session.get("player_lifecycle")
        market_registry = session.get("player_market")
    roster, _ = replay_squad_registry(
        base, team=team, registry=registry,
        development_registry=development_registry,
        lifecycle_registry=lifecycle_registry,
        market_registry=market_registry,
    )
    return roster


def recruitment_market_for_workspace(
    base_dir: str | Path, *, team: str, season_index: int,
    registry: Mapping[str, Any] | None,
    development_registry: Mapping[str, Any] | None = None,
    lifecycle_registry: Mapping[str, Any] | None = None,
    market_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from src.data_engine.roster_loader import load_roster_json, roster_path_for_team

    root = Path(base_dir).resolve()
    base = load_roster_json(roster_path_for_team(str(root), team))
    if base is None:
        raise ValueError("recruitment roster is unavailable")
    roster, _ = replay_squad_registry(
        base, team=team, registry=registry,
        development_registry=development_registry,
        lifecycle_registry=lifecycle_registry,
        market_registry=market_registry,
    )
    return generate_recruitment_market(
        roster, team=team,
        market_id=recruitment_market_id(season_index=season_index, team=team),
    )
