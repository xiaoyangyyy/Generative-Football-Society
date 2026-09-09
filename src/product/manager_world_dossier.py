"""Bounded same-chapter dossier for one manager decision and its world."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
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
    return {
        "status": status,
        "retained_records": _bounded_count(action.get("retained_records")),
        "influenced_decisions": _bounded_count(
            action.get("influenced_decisions")
        ),
        "attribution_eligible_decisions": _bounded_count(
            action.get("attribution_eligible_decisions")
        ),
        "locally_attributable_action_changes": _bounded_count(
            action.get("locally_attributable_action_changes")
        ),
        "direct_ball_event_links": _bounded_count(
            action.get("direct_ball_event_links")
        ),
        "manager_record_coverage_complete": (
            action.get("manager_record_coverage_complete") is True
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
        "transition_summary": {
            field: _bounded_count(transitions.get(field))
            for field in _WORLD_TRANSITIONS
        } if persistent else None,
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
