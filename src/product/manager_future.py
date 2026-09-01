"""Identity-bound bridge from one manager decision to simulator future sets."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from src.product.club_strategy import resolve_fixture_club_strategy
from src.product.match_plan import NATIVE_TACTIC, PLAYABLE_TACTICS
from src.product.season import SeasonPlan, opponent_preparation, validate_season_state


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manager_future_context(
    season: Mapping[str, Any], *, fixture_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the exact unstarted managed fixture controls used by matchday."""
    validate_season_state(season)
    plan = SeasonPlan.from_payload(season["plan"])
    if plan.manager_team is None:
        raise ValueError("season has no focus team")
    candidates = [
        row for row in season["fixtures"]
        if plan.manager_team in {row["home"], row["away"]}
        and row["state"] != "completed"
    ]
    if not candidates:
        raise ValueError("focus team has no pending fixture")
    fixture = candidates[0]
    if fixture_id is not None and fixture_id != fixture["fixture_id"]:
        raise ValueError("only the next focus-team fixture can receive a future set")
    if fixture["state"] != "scheduled" or int(fixture.get("attempts", 0)) != 0:
        raise ValueError("manager future set is frozen after fixture execution starts")
    decision = fixture.get("manager_decision")
    if not isinstance(decision, Mapping):
        raise ValueError("freeze the manager decision before creating a future set")
    if decision.get("team") != plan.manager_team:
        raise ValueError("manager future-set decision identity mismatch")
    preparation = opponent_preparation(season, str(fixture["fixture_id"]))
    home = str(fixture["home"])
    away = str(fixture["away"])
    strategy = season.get("club_strategies")
    resolved_strategy = None
    if isinstance(strategy, Mapping):
        resolved_strategy = resolve_fixture_club_strategy(
            strategy, home=home, away=away,
            manager_decision=decision,
            opponent_preparation=preparation,
        )
        home_tactic = str(resolved_strategy["home_tactic"])
        away_tactic = str(resolved_strategy["away_tactic"])
        tactic_source = "frozen_club_strategy"
    else:
        home_tactic = away_tactic = NATIVE_TACTIC
        opponent_tactic = str(preparation["selected_tactic"])
        if decision["team"] == home:
            home_tactic = str(decision["tactic"])
            away_tactic = opponent_tactic
        else:
            away_tactic = str(decision["tactic"])
            home_tactic = opponent_tactic
        tactic_source = "manager_decision_and_opponent_preparation"
    if home_tactic not in PLAYABLE_TACTICS or away_tactic not in PLAYABLE_TACTICS:
        raise ValueError("manager future-set tactics are unsupported")
    fixture_identity = {
        "fixture_id": str(fixture["fixture_id"]),
        "matchday": int(fixture["matchday"]),
        "order": int(fixture["order"]),
        "home": home,
        "away": away,
    }
    decision_payload = dict(decision)
    preparation_payload = dict(preparation)
    base = {
        "schema_version": 1,
        "kind": "manager_prematch_world_model_future_set",
        "season_id": str(season["season_id"]),
        "season_revision": int(season.get("revision", 0)),
        "fixture": fixture_identity,
        "manager_team": str(plan.manager_team),
        "decision_identity": _identity(decision_payload),
        "opponent_preparation_identity": _identity(preparation_payload),
        "club_strategy_identity": (
            _identity(dict(resolved_strategy))
            if isinstance(resolved_strategy, Mapping) else None
        ),
        "match_seed": (
            int(season["seed"])
            + int(fixture["matchday"]) * 100
            + int(fixture["order"])
        ),
        "fast": bool(plan.fast),
        "home_tactic": home_tactic,
        "away_tactic": away_tactic,
        "tactic_source": tactic_source,
        "claim_boundary": (
            "identity-bound simulator exploration of one frozen manager decision; "
            "not a score forecast, best-time recommendation, causal effect, or "
            "real-football coaching recommendation"
        ),
    }
    return {**base, "context_identity": _identity(base)}


def validate_manager_future_context(
    context: Mapping[str, Any], season: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the persisted season still produces this exact context."""
    normalized = validate_manager_future_context_shape(context)
    fixture = normalized.get("fixture")
    fixture_id = fixture.get("fixture_id") if isinstance(fixture, Mapping) else None
    expected = build_manager_future_context(season, fixture_id=fixture_id)
    if normalized != expected:
        raise ValueError("manager future-set context is stale or modified")
    return expected


def validate_manager_future_context_shape(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a queued context's self-contained identity before persistence."""
    if not isinstance(context, Mapping):
        raise ValueError("manager future-set context must be an object")
    required = {
        "schema_version", "kind", "season_id", "season_revision", "fixture",
        "manager_team", "decision_identity", "opponent_preparation_identity",
        "club_strategy_identity", "match_seed", "fast", "home_tactic",
        "away_tactic", "tactic_source", "claim_boundary", "context_identity",
    }
    if set(context) != required:
        raise ValueError("manager future-set context fields are invalid")
    fixture = context.get("fixture")
    if not isinstance(fixture, Mapping) or set(fixture) != {
        "fixture_id", "matchday", "order", "home", "away",
    }:
        raise ValueError("manager future-set fixture identity is invalid")
    if (
        not all(
            isinstance(fixture.get(key), str) and bool(fixture.get(key))
            for key in ("fixture_id", "home", "away")
        )
        or fixture["home"].casefold() == fixture["away"].casefold()
        or isinstance(fixture.get("matchday"), bool)
        or not isinstance(fixture.get("matchday"), int)
        or int(fixture["matchday"]) < 1
        or isinstance(fixture.get("order"), bool)
        or not isinstance(fixture.get("order"), int)
        or int(fixture["order"]) < 1
        or context.get("manager_team") not in {
            fixture.get("home"), fixture.get("away"),
        }
        or context.get("tactic_source") not in {
            "frozen_club_strategy",
            "manager_decision_and_opponent_preparation",
        }
        or context.get("claim_boundary") != (
            "identity-bound simulator exploration of one frozen manager decision; "
            "not a score forecast, best-time recommendation, causal effect, or "
            "real-football coaching recommendation"
        )
    ):
        raise ValueError("manager future-set fixture values are invalid")
    if (
        context.get("schema_version") != 1
        or context.get("kind") != "manager_prematch_world_model_future_set"
        or not isinstance(context.get("season_id"), str)
        or not context.get("season_id")
        or isinstance(context.get("season_revision"), bool)
        or not isinstance(context.get("season_revision"), int)
        or int(context["season_revision"]) < 0
        or isinstance(context.get("match_seed"), bool)
        or not isinstance(context.get("match_seed"), int)
        or not 0 <= int(context["match_seed"]) <= 2**31 - 1
        or not isinstance(context.get("fast"), bool)
        or context.get("home_tactic") not in PLAYABLE_TACTICS
        or context.get("away_tactic") not in PLAYABLE_TACTICS
        or not all(
            isinstance(context.get(key), str)
            and len(str(context[key])) == 64
            and all(char in "0123456789abcdef" for char in str(context[key]))
            for key in (
                "decision_identity", "opponent_preparation_identity",
                "context_identity",
            )
        )
        or (
            context.get("club_strategy_identity") is not None
            and (
                not isinstance(context.get("club_strategy_identity"), str)
                or len(str(context["club_strategy_identity"])) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in str(context["club_strategy_identity"])
                )
            )
        )
    ):
        raise ValueError("manager future-set context values are invalid")
    normalized = dict(context)
    base = {key: value for key, value in normalized.items() if key != "context_identity"}
    if normalized["context_identity"] != _identity(base):
        raise ValueError("manager future-set context identity is invalid")
    return normalized


__all__ = [
    "build_manager_future_context", "validate_manager_future_context",
    "validate_manager_future_context_shape",
]
