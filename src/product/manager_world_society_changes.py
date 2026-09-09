"""Bounded projection of one persisted simulator society transition."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

from src.simulation.society_continuity import (
    EMOTION_BOUNDS,
    PSYCHOLOGICAL_BOUNDS,
    SOCIAL_BOUNDS,
    TACTICAL_BOUNDS,
)


SCHEMA_VERSION = 1
MAX_SOURCE_STATE_CHANGES = 19
MAX_VISIBLE_STATE_CHANGES = 19
_SCOPE_ORDER = (
    "tactical_controls",
    "emotion_profile",
    "social_narrative_state",
    "psychological_state",
    "referee_grievance",
)
_SCOPE_FIELDS = {
    "tactical_controls": TACTICAL_BOUNDS,
    "emotion_profile": EMOTION_BOUNDS,
    "social_narrative_state": SOCIAL_BOUNDS,
    "psychological_state": PSYCHOLOGICAL_BOUNDS,
    "referee_grievance": {"value": (0.0, 0.92)},
}
_CANONICAL_KEYS = tuple(
    (scope, field)
    for scope in _SCOPE_ORDER
    for field in sorted(_SCOPE_FIELDS[scope])
)
_CANONICAL_RANK = {
    key: index for index, key in enumerate(_CANONICAL_KEYS)
}


def _unavailable(
    reason: str, *, changed_state_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "reason": reason,
        "changed_state_fields": changed_state_fields or [],
        "total_changes": 0,
        "visible_count": 0,
        "changes_truncated": False,
        "changes": [],
        "descriptive_persisted_state_only": True,
        "manager_or_action_effect_authorized": False,
        "outcome_attribution_authorized": False,
    }


def _changed_scopes(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    if any(not isinstance(scope, str) for scope in raw):
        return None
    canonical = [scope for scope in _SCOPE_ORDER if scope in raw]
    if raw != canonical or len(raw) != len(set(raw)):
        return None
    return canonical


def _state_value(
    value: Any, *, bounds: tuple[float, float], available: bool,
) -> int | float | None:
    if value is None:
        return None if not available else False
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    number = float(value)
    if not math.isfinite(number) or not bounds[0] <= number <= bounds[1]:
        return False
    return value


def project_manager_world_society_changes(raw: Any) -> dict[str, Any]:
    """Validate full content-free state changes before exposing them."""
    if not isinstance(raw, Mapping):
        return _unavailable("legacy_or_missing_society_transition")
    if raw.get("available") is not True:
        return _unavailable("society_continuity_not_recorded")
    scopes = _changed_scopes(raw.get("changed_state_fields"))
    before_available = raw.get("before_available")
    after_available = raw.get("after_available")
    if (
        scopes is None
        or not isinstance(before_available, bool)
        or not isinstance(after_available, bool)
        or not (before_available or after_available)
    ):
        return _unavailable("invalid_society_state_change_evidence")
    source = raw.get("state_changes")
    if source is None:
        return _unavailable(
            "legacy_or_missing_society_state_change_evidence",
            changed_state_fields=scopes,
        )
    if not isinstance(source, list) or len(source) > MAX_SOURCE_STATE_CHANGES:
        return _unavailable("invalid_society_state_change_evidence")
    changes = []
    keys = []
    for row in source:
        if not isinstance(row, Mapping) or set(row) != {
            "scope", "field", "before", "after",
        }:
            return _unavailable("invalid_society_state_change_evidence")
        scope = row.get("scope")
        field = row.get("field")
        if scope not in _SCOPE_FIELDS or field not in _SCOPE_FIELDS[scope]:
            return _unavailable("invalid_society_state_change_evidence")
        bounds = _SCOPE_FIELDS[scope][field]
        before = _state_value(
            row.get("before"), bounds=bounds, available=before_available,
        )
        after = _state_value(
            row.get("after"), bounds=bounds, available=after_available,
        )
        if before is False or after is False or before == after:
            return _unavailable("invalid_society_state_change_evidence")
        keys.append((scope, field))
        changes.append({
            "scope": scope,
            "field": field,
            "before": before,
            "after": after,
        })
    if (
        keys != sorted(keys, key=_CANONICAL_RANK.__getitem__)
        or len(keys) != len(set(keys))
    ):
        return _unavailable("noncanonical_society_state_changes")
    observed_scopes = [
        scope for scope in _SCOPE_ORDER
        if any(row[0] == scope for row in keys)
    ]
    if observed_scopes != scopes:
        return _unavailable("society_state_scope_mismatch")
    visible = changes[:MAX_VISIBLE_STATE_CHANGES]
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "reason": None,
        "changed_state_fields": scopes,
        "total_changes": len(changes),
        "visible_count": len(visible),
        "changes_truncated": len(changes) > MAX_VISIBLE_STATE_CHANGES,
        "changes": visible,
        "descriptive_persisted_state_only": True,
        "manager_or_action_effect_authorized": False,
        "outcome_attribution_authorized": False,
    }


__all__ = [
    "MAX_SOURCE_STATE_CHANGES",
    "MAX_VISIBLE_STATE_CHANGES",
    "SCHEMA_VERSION",
    "project_manager_world_society_changes",
]
