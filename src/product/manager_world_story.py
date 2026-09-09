"""One bounded contribution story projected from a validated world navigator."""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.product.manager_world_dossier import (
    CLAIM_BOUNDARY,
    build_manager_world_dossier,
)


SCHEMA_VERSION = 6


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, min(100_000, int(number))) if math.isfinite(number) else 0


def _unavailable_story() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "view_mode": "unavailable",
        "story_state": "unavailable",
        "source": None,
        "previous_completed": None,
        "chapter_dossier": None,
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


def _active_projection(
    navigator: Mapping[str, Any], current: Mapping[str, Any],
) -> dict[str, Any]:
    current_stages = current.get("stages")
    current_stages = current_stages if isinstance(current_stages, list) else []
    stage_by_id = {
        str(row.get("stage_id")): row
        for row in current_stages if isinstance(row, Mapping)
    }
    stage_status = {
        stage_id: str(row.get("status"))
        for stage_id, row in stage_by_id.items()
    }
    intervention = (
        "complete" if current.get("frozen_decision_identity") else
        "action_required" if current.get("workflow_state") == "decision_required"
        else "pending"
    )
    review_status = stage_status.get("record_manager_review")
    future = (
        "selected" if review_status == "review_recorded" else
        "available" if stage_status.get("inspect_local_mechanism")
        in {"scenario_evidence_available", "aggregate_only"} else "pending"
    )
    fixture = current["fixture"]
    frozen = stage_by_id.get("freeze_intervention")
    frozen = frozen if isinstance(frozen, Mapping) else {}
    review_stage = stage_by_id.get("record_manager_review")
    review_stage = review_stage if isinstance(review_stage, Mapping) else {}
    evidence = current.get("evidence_summary")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    dossier = build_manager_world_dossier(
        phase="active",
        choice={
            "decision_identity": current.get("frozen_decision_identity"),
            "selected_tactic": frozen.get("tactic"),
            "rotation": frozen.get("rotation"),
            "applied_tactic": None,
            "runtime_verified": False,
            "runtime_matches_selection": False,
        },
        review={
            **evidence,
            "intent": review_stage.get("intent"),
            "evidence_level": None,
            "timing_sensitivity_observed": None,
        },
        review_status=future,
        action={},
        action_status="pending",
        world={},
        world_status="pending",
        continuity_gaps=current.get("continuity_gaps"),
    )
    return {
        "story_state": "awaiting_official_world",
        "source": {
            "season_id": str(navigator.get("season_id") or ""),
            "navigator_identity": str(navigator.get("navigator_identity") or ""),
            "chapter_identity": str(current.get("current_chapter_identity") or ""),
            "chapter_state": "active",
            "navigable": False,
            "fixture_id": str(fixture.get("fixture_id") or ""),
            "matchday": _bounded_count(fixture.get("matchday")),
        },
        "stages": [
            {"stage_id": "manager_intervention", "status": intervention},
            {"stage_id": "counterfactual_review", "status": future},
            {"stage_id": "official_action_adoption", "status": "pending"},
            {"stage_id": "persistent_world_transition", "status": "pending"},
            {"stage_id": "outcome_evidence", "status": "pending"},
        ],
        "metrics": {
            "influenced_decisions": 0,
            "locally_attributable_action_changes": 0,
            "result_available": False,
            "persistent_state_available": False,
        },
        "chapter_dossier": dossier,
    }


def _completed_projection(
    navigator: Mapping[str, Any], chapter: Mapping[str, Any],
) -> dict[str, Any]:
    reviewed = chapter.get("reviewed_future_context")
    reviewed = reviewed if isinstance(reviewed, Mapping) else {}
    adoption = chapter.get("action_adoption")
    adoption = adoption if isinstance(adoption, Mapping) else {}
    world = chapter.get("descriptive_world_after")
    world = world if isinstance(world, Mapping) else {}
    action_state = str(adoption.get("state") or "evidence_unavailable")
    action_status = {
        "realized_action_change": "changed",
        "probability_influence_only": "influenced_only",
        "no_nonzero_influence": "no_influence",
        "not_applicable_stable_mode": "not_applicable",
        "evidence_unavailable": "evidence_unavailable",
    }.get(action_state, "evidence_unavailable")
    review_status = (
        "selected" if reviewed.get("selected_for_fixture") is True else
        "reviewed_not_selected" if reviewed.get("available") is True
        else "not_used"
    )
    world_status = (
        "world_persisted" if chapter.get("persistent_transition_identity")
        and world.get("persistent_state_available") is True
        else "evidence_unavailable"
    )
    dossier = build_manager_world_dossier(
        phase="completed",
        choice=chapter.get("manager_choice"),
        review=reviewed,
        review_status=review_status,
        action=adoption,
        action_status=action_status,
        world=world,
        world_status=world_status,
        continuity_gaps=chapter.get("continuity_gaps"),
    )
    return {
        "story_state": {
            "realized_action_change": "local_action_change_observed",
            "probability_influence_only": "probability_influence_only",
            "no_nonzero_influence": "no_nonzero_influence",
            "not_applicable_stable_mode": "not_applicable",
        }.get(action_state, "evidence_gap"),
        "source": {
            "season_id": str(navigator.get("season_id") or ""),
            "navigator_identity": str(navigator.get("navigator_identity") or ""),
            "chapter_identity": str(chapter.get("chapter_identity") or ""),
            "chapter_state": "completed",
            "navigable": True,
            "fixture_id": str(chapter.get("fixture_id") or ""),
            "matchday": _bounded_count(chapter.get("matchday")),
        },
        "stages": [
            {"stage_id": "manager_intervention", "status": "complete"},
            {"stage_id": "counterfactual_review", "status": review_status},
            {"stage_id": "official_action_adoption", "status": action_status},
            {"stage_id": "persistent_world_transition", "status": world_status},
            {"stage_id": "outcome_evidence", "status": (
                "descriptive_result_only" if world.get("result_available") is True
                else "evidence_unavailable"
            )},
        ],
        "metrics": {
            "influenced_decisions": _bounded_count(
                adoption.get("influenced_decisions")
            ),
            "locally_attributable_action_changes": _bounded_count(
                chapter.get("locally_attributable_action_changes")
            ),
            "result_available": world.get("result_available") is True,
            "persistent_state_available": world.get(
                "persistent_state_available"
            ) is True,
        },
        "chapter_dossier": dossier,
    }


def build_manager_world_story(
    navigator: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project the active chapter first and retain one completed predecessor."""
    unavailable = _unavailable_story()
    if not isinstance(navigator, Mapping) or navigator.get("available") is not True:
        return unavailable

    current = navigator.get("current_chapter")
    current = current if isinstance(current, Mapping) else {}
    history = navigator.get("history_chapters")
    history = history if isinstance(history, list) else []
    latest = next((row for row in history if isinstance(row, Mapping)), None)
    active_fixture = current.get("fixture")
    active = bool(
        isinstance(active_fixture, Mapping)
        and active_fixture.get("fixture_id")
        and current.get("workflow_state") != "season_complete"
    )

    if active:
        primary = _active_projection(navigator, current)
        completed = _completed_projection(navigator, latest) if latest else None
        previous_completed = ({
            "source": completed["source"],
            "story_state": completed["story_state"],
            "action_status": completed["stages"][2]["status"],
            "persistent_world_status": completed["stages"][3]["status"],
            "outcome_status": completed["stages"][4]["status"],
            "metrics": completed["metrics"],
            "chapter_dossier": completed["chapter_dossier"],
            "same_chapter_evidence": True,
        } if completed else None)
        view_mode = "active_chapter"
    elif latest is not None:
        primary = _completed_projection(navigator, latest)
        previous_completed = None
        view_mode = "latest_completed_chapter"
    else:
        return unavailable

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "view_mode": view_mode,
        "story_state": primary["story_state"],
        "source": primary["source"],
        "previous_completed": previous_completed,
        "stages": primary["stages"],
        "metrics": primary["metrics"],
        "chapter_dossier": primary["chapter_dossier"],
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
