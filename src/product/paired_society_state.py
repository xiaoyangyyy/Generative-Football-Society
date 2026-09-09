"""Identity-bound terminal society-state comparison for paired worlds."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from src.simulation.society_continuity import (
    EMOTION_BOUNDS,
    MEMORY_LIMITS,
    PSYCHOLOGICAL_BOUNDS,
    SOCIAL_BOUNDS,
    TACTICAL_BOUNDS,
    validate_society_public_snapshot,
)


SCHEMA_VERSION = 1
MAX_TERMINAL_STATE_CHANGES = 38
MAX_COUNT_DELTAS = 8
_COUNT_FIELDS = (
    "memory_records", "cognitive_memory_records", "beliefs", "reflections",
)
_COUNT_ABSOLUTE_LIMITS = {
    "memory_records": (
        MEMORY_LIMITS["episodic_memory"] + MEMORY_LIMITS["procedural_memory"]
    ),
    "cognitive_memory_records": (
        MEMORY_LIMITS["episodic_memory"] + MEMORY_LIMITS["procedural_memory"]
    ),
    "beliefs": MEMORY_LIMITS["beliefs"],
    "reflections": MEMORY_LIMITS["reflection_audit"],
}
_SCOPE_ORDER = (
    "tactical_controls", "emotion_profile", "social_narrative_state",
    "psychological_state", "referee_grievance",
)
_SCOPE_FIELDS = {
    "tactical_controls": TACTICAL_BOUNDS,
    "emotion_profile": EMOTION_BOUNDS,
    "social_narrative_state": SOCIAL_BOUNDS,
    "psychological_state": PSYCHOLOGICAL_BOUNDS,
    "referee_grievance": {"value": (0.0, 0.92)},
}
_BOUNDARY = (
    "identity-bound terminal society-state differences between one paired "
    "simulator baseline and treatment; policy-to-state, outcome and "
    "real-football causality remain unauthorized"
)
_FIELDS = {
    "schema_version", "available", "reason", "status", "teams",
    "pair_eligible", "baseline_state_identities",
    "treatment_state_identities", "count_deltas", "state_changes",
    "state_change_count", "count_delta_count",
    "teams_with_terminal_divergence",
    "paired_terminal_state_comparison_authorized",
    "policy_to_state_causality_authorized",
    "outcome_causality_authorized", "real_football_causality_authorized",
    "claim_boundary", "divergence_identity",
}


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _is_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _unavailable(reason: str, teams: Sequence[str], *, eligible: bool) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "reason": reason,
        "status": reason,
        "teams": list(teams),
        "pair_eligible": eligible,
        "baseline_state_identities": {},
        "treatment_state_identities": {},
        "count_deltas": [],
        "state_changes": [],
        "state_change_count": 0,
        "count_delta_count": 0,
        "teams_with_terminal_divergence": 0,
        "paired_terminal_state_comparison_authorized": False,
        "policy_to_state_causality_authorized": False,
        "outcome_causality_authorized": False,
        "real_football_causality_authorized": False,
        "claim_boundary": _BOUNDARY,
    }
    return {**payload, "divergence_identity": _identity(payload)}


def _society_layer(report: Mapping[str, Any]) -> Any:
    layers = report.get("layers")
    if not isinstance(layers, Mapping):
        return None
    return layers.get("society")


def _state_value(snapshot: Mapping[str, Any], scope: str, field: str) -> Any:
    if scope == "referee_grievance":
        return snapshot.get(scope)
    values = snapshot.get(scope)
    return values.get(field) if isinstance(values, Mapping) else None


def build_paired_society_divergence(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    teams: Sequence[str],
    pair_eligible: bool,
) -> dict[str, Any]:
    """Compare two content-free terminal snapshots without causal promotion."""
    normalized_teams = list(teams)
    if (
        len(normalized_teams) != 2
        or len(set(normalized_teams)) != 2
        or any(not isinstance(team, str) or not team for team in normalized_teams)
        or not isinstance(pair_eligible, bool)
    ):
        raise ValueError("paired society teams are invalid")
    baseline_states = _society_layer(baseline)
    treatment_states = _society_layer(treatment)
    if baseline_states in (None, {}) or treatment_states in (None, {}):
        return _unavailable(
            "legacy_or_missing_terminal_society_state",
            normalized_teams,
            eligible=pair_eligible,
        )
    if (
        not isinstance(baseline_states, Mapping)
        or not isinstance(treatment_states, Mapping)
        or set(baseline_states) != set(normalized_teams)
        or set(treatment_states) != set(normalized_teams)
    ):
        raise ValueError("paired society terminal state fields are invalid")
    for team in normalized_teams:
        validate_society_public_snapshot(baseline_states[team])
        validate_society_public_snapshot(treatment_states[team])
    if any(
        states[team].get("available") is not True
        for states in (baseline_states, treatment_states)
        for team in normalized_teams
    ):
        return _unavailable(
            "terminal_society_state_unavailable",
            normalized_teams,
            eligible=pair_eligible,
        )

    count_deltas = []
    state_changes = []
    changed_teams = set()
    for team in normalized_teams:
        left = baseline_states[team]
        right = treatment_states[team]
        for field in _COUNT_FIELDS:
            delta = int(right[field]) - int(left[field])
            if delta:
                count_deltas.append({
                    "team": team, "field": field, "delta": delta,
                })
                changed_teams.add(team)
        for scope in _SCOPE_ORDER:
            for field in sorted(_SCOPE_FIELDS[scope]):
                before = _state_value(left, scope, field)
                after = _state_value(right, scope, field)
                if before != after:
                    state_changes.append({
                        "team": team,
                        "scope": scope,
                        "field": field,
                        "baseline": before,
                        "treatment": after,
                    })
                    changed_teams.add(team)
    changed = bool(changed_teams)
    status = (
        "terminal_society_divergence_observed"
        if changed and pair_eligible else
        "descriptive_terminal_society_divergence_pair_ineligible"
        if changed else
        "no_terminal_society_divergence"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "reason": None,
        "status": status,
        "teams": normalized_teams,
        "pair_eligible": pair_eligible,
        "baseline_state_identities": {
            team: baseline_states[team]["state_identity"]
            for team in normalized_teams
        },
        "treatment_state_identities": {
            team: treatment_states[team]["state_identity"]
            for team in normalized_teams
        },
        "count_deltas": count_deltas,
        "state_changes": state_changes,
        "state_change_count": len(state_changes),
        "count_delta_count": len(count_deltas),
        "teams_with_terminal_divergence": len(changed_teams),
        "paired_terminal_state_comparison_authorized": True,
        "policy_to_state_causality_authorized": False,
        "outcome_causality_authorized": False,
        "real_football_causality_authorized": False,
        "claim_boundary": _BOUNDARY,
    }
    result = {**payload, "divergence_identity": _identity(payload)}
    return validate_paired_society_divergence(
        result, expected_teams=normalized_teams,
    )


def validate_paired_society_divergence(
    raw: Any, *, expected_teams: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fail closed on edited paired terminal-state evidence."""
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
        raise ValueError("paired society divergence fields are invalid")
    frozen = dict(raw)
    observed_identity = frozen.pop("divergence_identity")
    teams = raw.get("teams")
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or not isinstance(teams, list)
        or len(teams) != 2
        or len(set(teams)) != 2
        or any(not isinstance(team, str) or not team for team in teams)
        or expected_teams is not None and teams != list(expected_teams)
        or not isinstance(raw.get("available"), bool)
        or not isinstance(raw.get("pair_eligible"), bool)
        or not _is_identity(observed_identity)
        or observed_identity != _identity(frozen)
        or raw.get("claim_boundary") != _BOUNDARY
        or raw.get("policy_to_state_causality_authorized") is not False
        or raw.get("outcome_causality_authorized") is not False
        or raw.get("real_football_causality_authorized") is not False
    ):
        raise ValueError("paired society divergence values are invalid")
    if raw["available"] is False:
        if (
            raw.get("reason") not in {
                "legacy_or_missing_terminal_society_state",
                "terminal_society_state_unavailable",
            }
            or raw.get("status") != raw.get("reason")
            or raw.get("baseline_state_identities") != {}
            or raw.get("treatment_state_identities") != {}
            or raw.get("count_deltas") != []
            or raw.get("state_changes") != []
            or any(raw.get(field) != 0 for field in (
                "state_change_count", "count_delta_count",
                "teams_with_terminal_divergence",
            ))
            or raw.get(
                "paired_terminal_state_comparison_authorized"
            ) is not False
        ):
            raise ValueError("unavailable paired society divergence is invalid")
        return copy.deepcopy(dict(raw))
    if (
        raw.get("reason") is not None
        or raw.get("paired_terminal_state_comparison_authorized") is not True
    ):
        raise ValueError("available paired society divergence is invalid")
    identities = (
        raw.get("baseline_state_identities"),
        raw.get("treatment_state_identities"),
    )
    if any(
        not isinstance(values, Mapping)
        or list(values) != teams
        or any(not _is_identity(values.get(team)) for team in teams)
        for values in identities
    ):
        raise ValueError("paired society state identities are invalid")
    counts = raw.get("count_deltas")
    changes = raw.get("state_changes")
    if (
        not isinstance(counts, list)
        or len(counts) > MAX_COUNT_DELTAS
        or not isinstance(changes, list)
        or len(changes) > MAX_TERMINAL_STATE_CHANGES
    ):
        raise ValueError("paired society divergence budget is invalid")
    team_rank = {team: index for index, team in enumerate(teams)}
    count_rank = {field: index for index, field in enumerate(_COUNT_FIELDS)}
    count_keys = []
    for row in counts:
        if not isinstance(row, Mapping) or set(row) != {"team", "field", "delta"}:
            raise ValueError("paired society count delta fields are invalid")
        team, field, delta = row["team"], row["field"], row["delta"]
        if (
            team not in team_rank
            or field not in count_rank
            or isinstance(delta, bool)
            or not isinstance(delta, int)
            or delta == 0
            or abs(delta) > _COUNT_ABSOLUTE_LIMITS[field]
        ):
            raise ValueError("paired society count delta is invalid")
        count_keys.append((team, field))
    expected_count_order = sorted(
        count_keys, key=lambda key: (team_rank[key[0]], count_rank[key[1]]),
    )
    if count_keys != expected_count_order or len(count_keys) != len(set(count_keys)):
        raise ValueError("paired society count delta order is invalid")
    scope_rank = {scope: index for index, scope in enumerate(_SCOPE_ORDER)}
    change_keys = []
    for row in changes:
        if not isinstance(row, Mapping) or set(row) != {
            "team", "scope", "field", "baseline", "treatment",
        }:
            raise ValueError("paired society state change fields are invalid")
        team, scope, field = row["team"], row["scope"], row["field"]
        if team not in team_rank or scope not in _SCOPE_FIELDS:
            raise ValueError("paired society state change key is invalid")
        bounds = _SCOPE_FIELDS[scope].get(field)
        values = (row["baseline"], row["treatment"])
        if (
            bounds is None
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not bounds[0] <= float(value) <= bounds[1]
                for value in values
            )
            or values[0] == values[1]
        ):
            raise ValueError("paired society state change value is invalid")
        change_keys.append((team, scope, field))
    expected_change_order = sorted(change_keys, key=lambda key: (
        team_rank[key[0]], scope_rank[key[1]], key[2],
    ))
    if (
        change_keys != expected_change_order
        or len(change_keys) != len(set(change_keys))
    ):
        raise ValueError("paired society state change order is invalid")
    changed_teams = {
        row["team"] for row in [*counts, *changes]
    }
    expected_status = (
        "terminal_society_divergence_observed"
        if changed_teams and raw["pair_eligible"] else
        "descriptive_terminal_society_divergence_pair_ineligible"
        if changed_teams else
        "no_terminal_society_divergence"
    )
    if (
        raw.get("state_change_count") != len(changes)
        or raw.get("count_delta_count") != len(counts)
        or raw.get("teams_with_terminal_divergence") != len(changed_teams)
        or raw.get("status") != expected_status
    ):
        raise ValueError("paired society divergence summary is invalid")
    return copy.deepcopy(dict(raw))


__all__ = [
    "MAX_COUNT_DELTAS",
    "MAX_TERMINAL_STATE_CHANGES",
    "SCHEMA_VERSION",
    "build_paired_society_divergence",
    "validate_paired_society_divergence",
]
