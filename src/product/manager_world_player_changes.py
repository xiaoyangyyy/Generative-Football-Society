"""Bounded player-level projection of one persisted simulator transition."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_SOURCE_PLAYERS = 100
MAX_VISIBLE_PLAYERS = 12
MAX_VISIBLE_FIELDS = 12
_PLAYER_FIELDS = (
    "matches_played", "minutes_ema", "goals", "assists", "yellow_cards",
    "red_cards", "suspension_matches_left", "injury_matches_left",
    "injury_severity", "medical_recovery_credit", "form_ema",
    "media_sentiment", "last_rating",
)
_ALLOWED_FIELDS = {*_PLAYER_FIELDS, "condition_delta"}
_CHANGE_TYPES = {"added", "removed", "updated"}


def _unavailable(reason: str, *, expected_total: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "reason": reason,
        "total_changed": expected_total,
        "visible_count": 0,
        "players_truncated": False,
        "players": [],
        "descriptive_persisted_state_only": True,
        "manager_or_action_effect_authorized": False,
        "outcome_attribution_authorized": False,
    }


def _safe_text(value: Any, *, limit: int = 160) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= limit else None


def _safe_name(value: Any) -> str | None:
    return value if isinstance(value, str) and len(value) <= 160 else None


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return value


def _validate_scalar_change(raw: Any) -> tuple[int | float, int | float] | None:
    if not isinstance(raw, Mapping) or set(raw) != {"before", "after"}:
        return None
    before = _safe_number(raw.get("before"))
    after = _safe_number(raw.get("after"))
    if before is None or after is None or before == after:
        return None
    return before, after


def _validate_condition_change(
    raw: Any,
) -> list[dict[str, Any]] | None:
    if not isinstance(raw, Mapping) or set(raw) != {"before", "after"}:
        return None
    before = raw.get("before")
    after = raw.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    keys = sorted(set(before) | set(after))
    if len(keys) > 32 or any(_safe_text(key, limit=64) is None for key in keys):
        return None
    changes = []
    for key in keys:
        left = _safe_number(before.get(key, 0.0))
        right = _safe_number(after.get(key, 0.0))
        if left is None or right is None:
            return None
        if left != right:
            changes.append({
                "field": f"condition_delta.{key}",
                "before": left,
                "after": right,
            })
    return changes if changes else None


def _project_fields(raw: Any) -> tuple[list[dict[str, Any]], bool] | None:
    if (
        not isinstance(raw, Mapping)
        or len(raw) > 32
        or not raw
        or any(field not in _ALLOWED_FIELDS for field in raw)
    ):
        return None
    changes: list[dict[str, Any]] = []
    for field in _PLAYER_FIELDS:
        if field not in raw:
            continue
        values = _validate_scalar_change(raw[field])
        if values is None:
            return None
        changes.append({
            "field": field,
            "before": values[0],
            "after": values[1],
        })
    if "condition_delta" in raw:
        condition = _validate_condition_change(raw["condition_delta"])
        if condition is None:
            return None
        changes.extend(condition)
    return changes[:MAX_VISIBLE_FIELDS], len(changes) > MAX_VISIBLE_FIELDS


def project_manager_world_player_changes(
    raw: Any, *, expected_total: int,
) -> dict[str, Any]:
    """Validate the full source list and expose a bounded descriptive view."""
    if (
        isinstance(expected_total, bool)
        or not isinstance(expected_total, int)
        or not 0 <= expected_total <= 100_000
    ):
        raise ValueError("player change expected total is invalid")
    if not isinstance(raw, list):
        return _unavailable(
            "legacy_or_missing_player_change_evidence",
            expected_total=expected_total,
        )
    if len(raw) > MAX_SOURCE_PLAYERS or len(raw) != expected_total:
        return _unavailable(
            "player_change_total_mismatch",
            expected_total=expected_total,
        )
    validated = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != {
            "player_id", "name", "change", "fields",
        }:
            return _unavailable(
                "invalid_player_change_evidence", expected_total=expected_total,
            )
        player_id = _safe_text(row.get("player_id"))
        name = _safe_name(row.get("name"))
        change = row.get("change")
        fields = row.get("fields")
        if (
            player_id is None
            or name is None
            or change not in _CHANGE_TYPES
            or not isinstance(fields, Mapping)
            or (change in {"added", "removed"} and fields)
        ):
            return _unavailable(
                "invalid_player_change_evidence", expected_total=expected_total,
            )
        if change == "updated":
            projected_fields = _project_fields(fields)
            if projected_fields is None:
                return _unavailable(
                    "invalid_player_change_evidence",
                    expected_total=expected_total,
                )
            changes, fields_truncated = projected_fields
        else:
            changes, fields_truncated = [], False
        validated.append({
            "player_id": player_id,
            "name": name,
            "change": change,
            "fields": changes,
            "fields_truncated": fields_truncated,
        })
    identities = [row["player_id"] for row in validated]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        return _unavailable(
            "noncanonical_player_change_identities",
            expected_total=expected_total,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "reason": None,
        "total_changed": expected_total,
        "visible_count": min(len(validated), MAX_VISIBLE_PLAYERS),
        "players_truncated": len(validated) > MAX_VISIBLE_PLAYERS,
        "players": validated[:MAX_VISIBLE_PLAYERS],
        "descriptive_persisted_state_only": True,
        "manager_or_action_effect_authorized": False,
        "outcome_attribution_authorized": False,
    }


__all__ = [
    "MAX_SOURCE_PLAYERS",
    "MAX_VISIBLE_FIELDS",
    "MAX_VISIBLE_PLAYERS",
    "SCHEMA_VERSION",
    "project_manager_world_player_changes",
]
