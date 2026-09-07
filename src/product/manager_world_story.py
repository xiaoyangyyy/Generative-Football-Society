"""One bounded contribution story projected from a validated world navigator."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "same_simulator_chapter_descriptive_chain_not_outcome_causality"
)


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, min(100_000, int(number))) if math.isfinite(number) else 0


def _unavailable_story() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "story_state": "unavailable",
        "source": None,
        "stages": [],
        "metrics": {
            "influenced_decisions": 0,
            "locally_attributable_action_changes": 0,
            "result_available": False,
            "persistent_state_available": False,
        },
        "same_chapter_evidence": True,
        "outcome_improvement_authorized": False,
        "causal_effect_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_manager_world_story(
    navigator: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compress one validated chapter without inventing a causal funnel."""
    unavailable = _unavailable_story()
    if not isinstance(navigator, Mapping) or navigator.get("available") is not True:
        return unavailable

    current = navigator.get("current_chapter")
    current = current if isinstance(current, Mapping) else {}
    history = navigator.get("history_chapters")
    history = history if isinstance(history, list) else []
    latest = next((row for row in history if isinstance(row, Mapping)), None)
    stages: list[dict[str, str]]
    metrics = dict(unavailable["metrics"])

    if latest is None:
        current_stages = current.get("stages")
        current_stages = current_stages if isinstance(current_stages, list) else []
        stage_status = {
            str(row.get("stage_id")): str(row.get("status"))
            for row in current_stages
            if isinstance(row, Mapping)
        }
        intervention = (
            "complete"
            if current.get("frozen_decision_identity")
            else "action_required"
            if current.get("workflow_state") == "decision_required"
            else "pending"
        )
        review_status = stage_status.get("record_manager_review")
        future = (
            "selected"
            if review_status == "review_recorded"
            else "available"
            if stage_status.get("inspect_local_mechanism")
            in {"scenario_evidence_available", "aggregate_only"}
            else "pending"
        )
        stages = [
            {"stage_id": "manager_intervention", "status": intervention},
            {"stage_id": "counterfactual_review", "status": future},
            {"stage_id": "official_action_adoption", "status": "pending"},
            {"stage_id": "persistent_world_transition", "status": "pending"},
            {"stage_id": "outcome_evidence", "status": "pending"},
        ]
        fixture = current.get("fixture")
        fixture = fixture if isinstance(fixture, Mapping) else {}
        source = {
            "season_id": str(navigator.get("season_id") or ""),
            "navigator_identity": str(navigator.get("navigator_identity") or ""),
            "chapter_identity": None,
            "fixture_id": str(fixture.get("fixture_id") or ""),
            "matchday": _bounded_count(fixture.get("matchday")),
        }
        story_state = "awaiting_official_world"
    else:
        reviewed = latest.get("reviewed_future_context")
        reviewed = reviewed if isinstance(reviewed, Mapping) else {}
        adoption = latest.get("action_adoption")
        adoption = adoption if isinstance(adoption, Mapping) else {}
        world = latest.get("descriptive_world_after")
        world = world if isinstance(world, Mapping) else {}
        action_state = str(adoption.get("state") or "evidence_unavailable")
        action_status = {
            "realized_action_change": "changed",
            "probability_influence_only": "influenced_only",
            "no_nonzero_influence": "no_influence",
            "not_applicable_stable_mode": "not_applicable",
            "evidence_unavailable": "evidence_unavailable",
        }.get(action_state, "evidence_unavailable")
        stages = [
            {"stage_id": "manager_intervention", "status": "complete"},
            {
                "stage_id": "counterfactual_review",
                "status": (
                    "selected"
                    if reviewed.get("selected_for_fixture") is True
                    else "reviewed_not_selected"
                    if reviewed.get("available") is True
                    else "not_used"
                ),
            },
            {"stage_id": "official_action_adoption", "status": action_status},
            {
                "stage_id": "persistent_world_transition",
                "status": (
                    "world_persisted"
                    if latest.get("persistent_transition_identity")
                    and world.get("persistent_state_available") is True
                    else "evidence_unavailable"
                ),
            },
            {
                "stage_id": "outcome_evidence",
                "status": (
                    "descriptive_result_only"
                    if world.get("result_available") is True
                    else "evidence_unavailable"
                ),
            },
        ]
        metrics = {
            "influenced_decisions": _bounded_count(
                adoption.get("influenced_decisions")
            ),
            "locally_attributable_action_changes": _bounded_count(
                latest.get("locally_attributable_action_changes")
            ),
            "result_available": world.get("result_available") is True,
            "persistent_state_available": (
                world.get("persistent_state_available") is True
            ),
        }
        source = {
            "season_id": str(navigator.get("season_id") or ""),
            "navigator_identity": str(navigator.get("navigator_identity") or ""),
            "chapter_identity": str(latest.get("chapter_identity") or ""),
            "fixture_id": str(latest.get("fixture_id") or ""),
            "matchday": _bounded_count(latest.get("matchday")),
        }
        story_state = {
            "realized_action_change": "local_action_change_observed",
            "probability_influence_only": "probability_influence_only",
            "no_nonzero_influence": "no_nonzero_influence",
            "not_applicable_stable_mode": "not_applicable",
        }.get(action_state, "evidence_gap")

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "story_state": story_state,
        "source": source,
        "stages": stages,
        "metrics": metrics,
        "same_chapter_evidence": True,
        "outcome_improvement_authorized": False,
        "causal_effect_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def validate_manager_world_story(
    story: Mapping[str, Any], *, navigator: Mapping[str, Any] | None,
) -> None:
    """Replay the complete read model and reject any projection drift."""
    expected = build_manager_world_story(navigator)
    if dict(story) != expected:
        raise ValueError("manager world story replay mismatch")


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA_VERSION",
    "build_manager_world_story",
    "validate_manager_world_story",
]
