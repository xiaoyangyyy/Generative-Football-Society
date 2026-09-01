"""Action-scoped held-out validation contract for cross planning authority."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


CROSS_VALIDATION_VERSION = 1
CROSS_MIN_SAMPLES = 96
CROSS_MIN_GROUPS = 6
CROSS_REFERENCE_SAMPLES = 192
CROSS_REFERENCE_GROUPS = 8
CROSS_MIN_SKILL = 0.02
_SCOPE = "grouped_heldout_simulator_cross_transitions"
_BASELINE = "same_state_persistence"
_METRIC = "weighted_next_observation_mse"


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 and result == value else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _derived(
    *, samples: int, groups: int, model_mse: float, persistence_mse: float,
) -> dict[str, Any]:
    skill = (
        1.0 - model_mse / persistence_mse
        if persistence_mse > 1e-12 else 0.0
    )
    support = math.sqrt(
        min(1.0, samples / CROSS_REFERENCE_SAMPLES)
        * min(1.0, groups / CROSS_REFERENCE_GROUPS)
    )
    eligible = bool(
        samples >= CROSS_MIN_SAMPLES
        and groups >= CROSS_MIN_GROUPS
        and persistence_mse > 1e-12
        and skill >= CROSS_MIN_SKILL
    )
    quality = float(np.clip(skill, 0.0, 1.0) * support) if eligible else 0.0
    reason = (
        "validated_skill_vs_persistence" if eligible
        else "insufficient_cross_samples" if samples < CROSS_MIN_SAMPLES
        else "insufficient_cross_groups" if groups < CROSS_MIN_GROUPS
        else "degenerate_persistence_baseline" if persistence_mse <= 1e-12
        else "cross_skill_below_minimum"
    )
    return {
        "skill_vs_persistence": float(skill),
        "support_factor": float(support),
        "planner_quality": quality,
        "eligible": eligible,
        "reason": reason,
    }


def build_cross_action_validation(
    current_observations: Any,
    predicted_observations: Any,
    actual_observations: Any,
    observation_weights: Any,
    groups: Sequence[str],
) -> dict[str, Any]:
    """Build grouped holdout evidence without claiming real-football validity."""
    current = np.asarray(current_observations, dtype=np.float64)
    predicted = np.asarray(predicted_observations, dtype=np.float64)
    actual = np.asarray(actual_observations, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64).reshape(-1)
    group_values = np.asarray(groups, dtype=str).reshape(-1)
    if (
        current.ndim != 2 or predicted.shape != current.shape
        or actual.shape != current.shape or weights.shape != (current.shape[1],)
        or group_values.shape != (current.shape[0],)
    ):
        raise ValueError("cross validation arrays have incompatible shapes")
    if not all(np.isfinite(value).all() for value in (
        current, predicted, actual, weights,
    )):
        raise ValueError("cross validation arrays must be finite")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("cross validation weights must have positive mass")
    samples = int(len(current))
    group_count = int(len(np.unique(group_values))) if samples else 0
    if samples:
        model_mse = float(np.mean((predicted - actual) ** 2 * weights))
        persistence_mse = float(np.mean((current - actual) ** 2 * weights))
    else:
        model_mse = persistence_mse = 0.0
    derived = _derived(
        samples=samples, groups=group_count,
        model_mse=model_mse, persistence_mse=persistence_mse,
    )
    return {
        "schema_version": CROSS_VALIDATION_VERSION,
        "validation_scope": _SCOPE,
        "data_domain": "simulator_transition_rows_only",
        "external_football_validity": False,
        "grouped_holdout": True,
        "action": "cross",
        "baseline": _BASELINE,
        "metric": _METRIC,
        "samples": samples,
        "groups": group_count,
        "minimum_samples": CROSS_MIN_SAMPLES,
        "minimum_groups": CROSS_MIN_GROUPS,
        "minimum_skill_vs_persistence": CROSS_MIN_SKILL,
        "model_weighted_mse": model_mse,
        "persistence_weighted_mse": persistence_mse,
        **derived,
    }


def replay_cross_action_validation(raw: Any) -> dict[str, Any]:
    """Validate and recompute a checkpoint's cross authority evidence."""
    unavailable = {
        "valid": False, "quality": 0.0,
        "reason": "cross_validation_unavailable",
    }
    if not isinstance(raw, Mapping):
        return unavailable
    if (
        raw.get("schema_version") != CROSS_VALIDATION_VERSION
        or raw.get("validation_scope") != _SCOPE
        or raw.get("data_domain") != "simulator_transition_rows_only"
        or raw.get("external_football_validity") is not False
        or raw.get("grouped_holdout") is not True
        or raw.get("action") != "cross"
        or raw.get("baseline") != _BASELINE
        or raw.get("metric") != _METRIC
        or raw.get("minimum_samples") != CROSS_MIN_SAMPLES
        or raw.get("minimum_groups") != CROSS_MIN_GROUPS
        or raw.get("minimum_skill_vs_persistence") != CROSS_MIN_SKILL
    ):
        return {**unavailable, "reason": "cross_validation_contract_mismatch"}
    samples = _nonnegative_int(raw.get("samples"))
    groups = _nonnegative_int(raw.get("groups"))
    model_mse = _finite(raw.get("model_weighted_mse"))
    persistence_mse = _finite(raw.get("persistence_weighted_mse"))
    if (
        samples is None or groups is None or groups > samples
        or model_mse is None or persistence_mse is None
        or model_mse < 0.0 or persistence_mse < 0.0
    ):
        return {**unavailable, "reason": "cross_validation_values_invalid"}
    derived = _derived(
        samples=samples, groups=groups,
        model_mse=model_mse, persistence_mse=persistence_mse,
    )
    for field in (
        "skill_vs_persistence", "support_factor", "planner_quality",
    ):
        stored = _finite(raw.get(field))
        if stored is None or not math.isclose(
            stored, float(derived[field]), rel_tol=1e-9, abs_tol=1e-12,
        ):
            return {**unavailable, "reason": "cross_validation_replay_mismatch"}
    if (
        raw.get("eligible") is not derived["eligible"]
        or raw.get("reason") != derived["reason"]
    ):
        return {**unavailable, "reason": "cross_validation_replay_mismatch"}
    return {
        "valid": True,
        "quality": float(derived["planner_quality"]),
        "reason": str(derived["reason"]),
        "eligible": bool(derived["eligible"]),
        "samples": samples,
        "groups": groups,
        "skill_vs_persistence": float(derived["skill_vs_persistence"]),
        "model_weighted_mse": model_mse,
        "persistence_weighted_mse": persistence_mse,
        "external_football_validity": False,
    }
