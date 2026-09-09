"""Bounded same-chapter dossier for one manager decision and its world."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.product.manager_world_player_changes import (
    project_manager_world_player_changes,
)

SCHEMA_VERSION = 3
CLAIM_BOUNDARY = (
    "same_simulator_chapter_descriptive_chain_not_outcome_causality"
)
_WORLD_METRICS = (
    "team_fatigue_ema",
    "squad_morale_ema",
    "team_media_pressure",
    "injured_players",
    "suspended_players",
    "unavailable_players",
)
_WORLD_TRANSITIONS = (
    "changed_players",
    "new_injuries",
    "injuries_cleared",
    "new_suspensions",
    "suspensions_cleared",
)
_ACTIONS = ("hold", "pass", "cross", "shot", "none")


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, min(100_000, int(number))) if math.isfinite(number) else 0


def _safe_text(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 160 else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _public_count(value: Any) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 100_000
    ):
        return None
    return value


def _unavailable_transition_evidence(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "schema_version": None,
        "transitions": [],
        "observed_transition_total": 0,
        "counts_match_official": False,
        "expectation_available": False,
        "expected_counterfactual_action_changes": None,
        "full_source_transition_distribution_authorized": False,
        "full_source_expectation_authorized": False,
        "same_chapter_cooccurrence_only": True,
        "outcome_attribution_authorized": False,
    }


def _action_transition_evidence(
    raw: Any, *, expected_changes: int,
) -> dict[str, Any]:
    semantic = raw if isinstance(raw, Mapping) else {}
    schema_version = semantic.get("schema_version")
    matrix = semantic.get("locally_attributable_action_transition_counts")
    if schema_version not in {2, 3} or not isinstance(matrix, Mapping):
        return _unavailable_transition_evidence(
            "legacy_or_missing_transition_semantics"
        )
    if (
        set(matrix) != set(_ACTIONS)
        or any(
            not isinstance(matrix.get(before), Mapping)
            or set(matrix[before]) != set(_ACTIONS)
            for before in _ACTIONS
        )
    ):
        return _unavailable_transition_evidence(
            "invalid_or_incomplete_transition_matrix"
        )
    counts = {
        before: {
            after: _public_count(matrix[before][after])
            for after in _ACTIONS
        }
        for before in _ACTIONS
    }
    values = [
        count for row in counts.values() for count in row.values()
    ]
    if (
        any(count is None for count in values)
        or any(counts[action][action] != 0 for action in _ACTIONS)
        or any(
            counts["none"][action] != 0
            or counts[action]["none"] != 0
            for action in _ACTIONS
        )
    ):
        return _unavailable_transition_evidence(
            "invalid_or_incomplete_transition_matrix"
        )
    total = sum(int(count) for count in values)
    if total != expected_changes:
        return _unavailable_transition_evidence(
            "transition_total_mismatch"
        )
    transitions = [
        {
            "transition_id": f"{before}_to_{after}",
            "baseline_action": before,
            "actual_action": after,
            "count": counts[before][after],
        }
        for before in _ACTIONS
        for after in _ACTIONS
        if counts[before][after]
    ]
    expected = _finite_number(
        semantic.get("expected_counterfactual_action_changes")
    )
    expectation_available = bool(
        schema_version == 3
        and semantic.get("expected_change_estimator")
        == "shared_uniform_inverse_cdf_overlap_v1"
        and expected is not None
        and 0.0 <= expected <= 100_000.0
    )
    return {
        "available": True,
        "reason": None,
        "schema_version": schema_version,
        "transitions": transitions,
        "observed_transition_total": total,
        "counts_match_official": True,
        "expectation_available": expectation_available,
        "expected_counterfactual_action_changes": (
            expected if expectation_available else None
        ),
        "full_source_transition_distribution_authorized": (
            semantic.get("full_source_distribution_authorized") is True
        ),
        "full_source_expectation_authorized": bool(
            expectation_available
            and semantic.get("full_source_expectation_authorized") is True
        ),
        "same_chapter_cooccurrence_only": True,
        "outcome_attribution_authorized": False,
    }


def _manager_choice(raw: Any) -> dict[str, Any]:
    choice = raw if isinstance(raw, Mapping) else {}
    decision_identity = _safe_text(choice.get("decision_identity"))
    selected_tactic = _safe_text(choice.get("selected_tactic"))
    rotation = _safe_text(choice.get("rotation"))
    applied_tactic = _safe_text(choice.get("applied_tactic"))
    return {
        "available": bool(decision_identity and selected_tactic and rotation),
        "decision_identity": decision_identity,
        "selected_tactic": selected_tactic,
        "rotation": rotation,
        "applied_tactic": applied_tactic,
        "runtime_verified": choice.get("runtime_verified") is True,
        "runtime_matches_selection": (
            choice.get("runtime_matches_selection") is True
        ),
    }


def _future_review(
    raw: Any, *, status: str, active: bool,
) -> dict[str, Any]:
    review = raw if isinstance(raw, Mapping) else {}
    available = (
        status in {"available", "selected"} if active
        else review.get("available") is True
    )
    return {
        "available": available,
        "status": status,
        "intent": _safe_text(review.get("intent")),
        "evidence_level": _safe_text(review.get("evidence_level")),
        "registered_scenarios": _bounded_count(
            review.get(
                "registered_scenarios",
                review.get("fixed_scenario_budget"),
            )
        ),
        "eligible_scenarios": _bounded_count(
            review.get("eligible_scenarios")
        ),
        "action_divergence_scenarios": _bounded_count(
            review.get("action_divergence_scenarios")
        ),
        "local_attribution_scenarios": _bounded_count(
            review.get("local_attribution_scenarios")
        ),
        "descriptive_future_difference_scenarios": _bounded_count(
            review.get("descriptive_future_difference_scenarios")
        ),
        "timing_sensitivity_observed": (
            review.get("timing_sensitivity_observed")
            if isinstance(review.get("timing_sensitivity_observed"), bool)
            else None
        ),
        "ranking_performed": False,
        "outcome_comparison_performed": False,
    }


def _official_action(raw: Any, *, status: str) -> dict[str, Any]:
    action = raw if isinstance(raw, Mapping) else {}
    locally_attributable_changes = _bounded_count(
        action.get("locally_attributable_action_changes")
    )
    return {
        "status": status,
        "retained_records": _bounded_count(action.get("retained_records")),
        "influenced_decisions": _bounded_count(
            action.get("influenced_decisions")
        ),
        "attribution_eligible_decisions": _bounded_count(
            action.get("attribution_eligible_decisions")
        ),
        "locally_attributable_action_changes": locally_attributable_changes,
        "direct_ball_event_links": _bounded_count(
            action.get("direct_ball_event_links")
        ),
        "manager_record_coverage_complete": (
            action.get("manager_record_coverage_complete") is True
        ),
        "transition_evidence": _action_transition_evidence(
            action.get("retained_record_semantics"),
            expected_changes=locally_attributable_changes,
        ),
    }


def _society_summary(raw: Any) -> dict[str, Any]:
    society = raw if isinstance(raw, Mapping) else {}
    fields = society.get("changed_state_fields")
    fields = fields if isinstance(fields, list) else []
    return {
        "available": society.get("available") is True,
        "memory_record_delta": _bounded_count(
            society.get("memory_record_delta")
        ),
        "cognitive_memory_delta": _bounded_count(
            society.get("cognitive_memory_delta")
        ),
        "belief_delta": _bounded_count(society.get("belief_delta")),
        "reflection_delta": _bounded_count(society.get("reflection_delta")),
        "changed_state_fields": [
            text for value in fields[:12]
            if (text := _safe_text(value)) is not None
        ],
    }


def _world_after(raw: Any, *, status: str) -> dict[str, Any]:
    world = raw if isinstance(raw, Mapping) else {}
    persistent = world.get("persistent_state_available") is True
    metrics = world.get("metrics_delta")
    metrics = metrics if persistent and isinstance(metrics, Mapping) else {}
    transitions = world.get("transition_summary")
    transitions = (
        transitions if persistent and isinstance(transitions, Mapping) else {}
    )
    transition_summary = {
        field: _bounded_count(transitions.get(field))
        for field in _WORLD_TRANSITIONS
    } if persistent else None
    return {
        "status": status,
        "result_available": world.get("result_available") is True,
        "outcome": _safe_text(world.get("outcome")),
        "points_earned": (
            _bounded_count(world.get("points_earned"))
            if world.get("result_available") is True else None
        ),
        "persistent_state_available": persistent,
        "recovery_complete": world.get("recovery_complete") is True,
        "metrics_delta": {
            field: _finite_number(metrics.get(field))
            for field in _WORLD_METRICS
        } if persistent else None,
        "transition_summary": transition_summary,
        "player_changes": (
            project_manager_world_player_changes(
                world.get("players_changed"),
                expected_total=transition_summary["changed_players"],
            )
            if persistent and transition_summary is not None else None
        ),
        "society_transition": _society_summary(
            world.get("society_transition")
        ),
    }


def build_manager_world_dossier(
    *,
    phase: str,
    choice: Any,
    review: Any,
    review_status: str,
    action: Any,
    action_status: str,
    world: Any,
    world_status: str,
    continuity_gaps: Any,
) -> dict[str, Any]:
    """Build one replayable read model without inferring an effect edge."""
    if phase not in {"active", "completed"}:
        raise ValueError("manager world dossier phase is invalid")
    gaps = continuity_gaps if isinstance(continuity_gaps, list) else []
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "phase": phase,
        "manager_choice": _manager_choice(choice),
        "future_review": _future_review(
            review, status=review_status, active=phase == "active",
        ),
        "official_action": _official_action(action, status=action_status),
        "world_after": _world_after(world, status=world_status),
        "continuity_gaps": [
            text for value in gaps[:20]
            if (text := _safe_text(value)) is not None
        ],
        "same_chapter_evidence": True,
        "outcome_improvement_authorized": False,
        "causal_effect_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA_VERSION",
    "build_manager_world_dossier",
]
