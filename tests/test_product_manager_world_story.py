import copy

import pytest

from scripts.verify_product_recovery import CODE_IDENTITY_FILES
from src.product.manager_world_story import (
    CLAIM_BOUNDARY,
    build_manager_world_story,
    validate_manager_world_story,
)


def _completed_navigator() -> dict:
    return {
        "available": True,
        "season_id": "season-1",
        "navigator_identity": "a" * 64,
        "current_chapter": {},
        "history_chapters": [{
            "chapter_identity": "b" * 64,
            "fixture_id": "fixture-3",
            "matchday": 3,
            "reviewed_future_context": {
                "available": True,
                "selected_for_fixture": True,
                "intent": "keep_after_review",
                "evidence_level": "scenario_evidence",
                "fixed_scenario_budget": 3,
                "eligible_scenarios": 2,
                "action_divergence_scenarios": 2,
                "local_attribution_scenarios": 1,
                "descriptive_future_difference_scenarios": 1,
                "timing_sensitivity_observed": True,
            },
            "manager_choice": {
                "decision_identity": "d" * 64,
                "selected_tactic": "gegenpress",
                "rotation": "balanced",
                "applied_tactic": "gegenpress",
                "runtime_verified": True,
                "runtime_matches_selection": True,
            },
            "action_adoption": {
                "state": "realized_action_change",
                "retained_records": 6,
                "influenced_decisions": 4,
                "attribution_eligible_decisions": 3,
                "locally_attributable_action_changes": 2,
                "direct_ball_event_links": 3,
                "manager_record_coverage_complete": True,
            },
            "descriptive_world_after": {
                "result_available": True,
                "outcome": "win",
                "points_earned": 3,
                "persistent_state_available": True,
                "recovery_complete": True,
                "metrics_delta": {
                    "team_fatigue_ema": 0.08,
                    "squad_morale_ema": 0.02,
                    "team_media_pressure": 0.01,
                    "injured_players": 1.0,
                    "suspended_players": 0.0,
                    "unavailable_players": 1.0,
                },
                "transition_summary": {
                    "changed_players": 7,
                    "new_injuries": 1,
                    "injuries_cleared": 0,
                    "new_suspensions": 0,
                    "suspensions_cleared": 0,
                },
                "society_transition": {
                    "available": True,
                    "memory_record_delta": 3,
                    "cognitive_memory_delta": 2,
                    "belief_delta": 1,
                    "reflection_delta": 0,
                    "changed_state_fields": [
                        "emotion_profile", "tactical_controls",
                    ],
                },
            },
            "locally_attributable_action_changes": 2,
            "persistent_transition_identity": "c" * 64,
            "continuity_gaps": [],
        }],
    }


def test_world_story_uses_one_latest_chapter_and_stays_noncausal():
    navigator = _completed_navigator()
    original = copy.deepcopy(navigator)

    story = build_manager_world_story(navigator)
    validate_manager_world_story(story, navigator=navigator)

    assert navigator == original
    assert story["schema_version"] == 3
    assert story["view_mode"] == "latest_completed_chapter"
    assert story["story_state"] == "local_action_change_observed"
    assert story["source"] == {
        "season_id": "season-1",
        "navigator_identity": "a" * 64,
        "chapter_identity": "b" * 64,
        "chapter_state": "completed",
        "navigable": True,
        "fixture_id": "fixture-3",
        "matchday": 3,
    }
    assert story["previous_completed"] is None
    assert [row["status"] for row in story["stages"]] == [
        "complete",
        "selected",
        "changed",
        "world_persisted",
        "descriptive_result_only",
    ]
    assert story["metrics"] == {
        "influenced_decisions": 4,
        "locally_attributable_action_changes": 2,
        "result_available": True,
        "persistent_state_available": True,
    }
    assert story["same_chapter_evidence"] is True
    assert story["outcome_improvement_authorized"] is False
    assert story["causal_effect_authorized"] is False
    assert story["claim_boundary"] == CLAIM_BOUNDARY
    dossier = story["chapter_dossier"]
    assert dossier["phase"] == "completed"
    assert dossier["manager_choice"] == {
        "available": True,
        "decision_identity": "d" * 64,
        "selected_tactic": "gegenpress",
        "rotation": "balanced",
        "applied_tactic": "gegenpress",
        "runtime_verified": True,
        "runtime_matches_selection": True,
    }
    assert dossier["future_review"]["registered_scenarios"] == 3
    assert dossier["future_review"]["local_attribution_scenarios"] == 1
    assert dossier["future_review"]["ranking_performed"] is False
    assert dossier["official_action"]["retained_records"] == 6
    assert dossier["official_action"]["influenced_decisions"] == 4
    assert dossier["official_action"][
        "locally_attributable_action_changes"
    ] == 2
    assert dossier["world_after"]["outcome"] == "win"
    assert dossier["world_after"]["metrics_delta"]["team_fatigue_ema"] == 0.08
    assert dossier["world_after"]["transition_summary"]["changed_players"] == 7
    assert dossier["world_after"]["society_transition"][
        "changed_state_fields"
    ] == ["emotion_profile", "tactical_controls"]
    assert dossier["outcome_improvement_authorized"] is False
    assert dossier["causal_effect_authorized"] is False


def test_world_story_exposes_current_review_without_fake_result():
    navigator = {
        "available": True,
        "season_id": "season-2",
        "navigator_identity": "d" * 64,
        "history_chapters": [],
        "current_chapter": {
            "current_chapter_identity": "f" * 64,
            "workflow_state": "review_recorded_decision_refrozen",
            "frozen_decision_identity": "e" * 64,
            "fixture": {
                "fixture_id": "fixture-1",
                "matchday": 1,
            },
            "stages": [{
                "stage_id": "freeze_intervention",
                "status": "decision_frozen",
                "tactic": "possession",
                "rotation": "fresh",
            }, {
                "stage_id": "record_manager_review",
                "status": "review_recorded",
                "intent": "revise_after_review",
            }],
            "evidence_summary": {
                "registered_scenarios": 3,
                "eligible_scenarios": 2,
                "action_divergence_scenarios": 1,
                "local_attribution_scenarios": 1,
                "descriptive_future_difference_scenarios": 1,
            },
            "continuity_gaps": [],
        },
    }

    story = build_manager_world_story(navigator)

    assert story["story_state"] == "awaiting_official_world"
    assert story["view_mode"] == "active_chapter"
    assert story["source"]["chapter_identity"] == "f" * 64
    assert story["source"]["chapter_state"] == "active"
    assert story["source"]["navigable"] is False
    assert story["previous_completed"] is None
    assert [row["status"] for row in story["stages"]] == [
        "complete", "selected", "pending", "pending", "pending",
    ]
    assert story["metrics"]["result_available"] is False
    dossier = story["chapter_dossier"]
    assert dossier["phase"] == "active"
    assert dossier["manager_choice"]["selected_tactic"] == "possession"
    assert dossier["manager_choice"]["rotation"] == "fresh"
    assert dossier["future_review"]["status"] == "selected"
    assert dossier["future_review"]["intent"] == "revise_after_review"
    assert dossier["future_review"]["registered_scenarios"] == 3
    assert dossier["official_action"]["status"] == "pending"
    assert dossier["world_after"]["status"] == "pending"
    assert story["outcome_improvement_authorized"] is False


def test_world_story_unavailable_state_is_fresh_and_fail_closed():
    first = build_manager_world_story(None)
    second = build_manager_world_story({"available": False})

    first["metrics"]["influenced_decisions"] = 99

    assert second["available"] is False
    assert second["view_mode"] == "unavailable"
    assert second["previous_completed"] is None
    assert second["chapter_dossier"] is None
    assert second["metrics"]["influenced_decisions"] == 0
    assert second["outcome_improvement_authorized"] is False
    assert second["causal_effect_authorized"] is False


def test_world_story_validation_rejects_projection_drift():
    navigator = _completed_navigator()
    story = build_manager_world_story(navigator)
    story["outcome_improvement_authorized"] = True

    with pytest.raises(ValueError, match="replay mismatch"):
        validate_manager_world_story(story, navigator=navigator)


def test_world_story_is_bound_into_product_recovery_identity():
    assert "src/product/manager_world_story.py" in CODE_IDENTITY_FILES


@pytest.mark.parametrize((
    "action_state", "stage_status", "story_state",
), (
    ("realized_action_change", "changed", "local_action_change_observed"),
    ("probability_influence_only", "influenced_only", "probability_influence_only"),
    ("no_nonzero_influence", "no_influence", "no_nonzero_influence"),
    ("not_applicable_stable_mode", "not_applicable", "not_applicable"),
    ("evidence_unavailable", "evidence_unavailable", "evidence_gap"),
))
def test_world_story_preserves_every_official_action_state(
    action_state, stage_status, story_state,
):
    navigator = _completed_navigator()
    navigator["history_chapters"][0]["action_adoption"]["state"] = action_state

    story = build_manager_world_story(navigator)

    assert story["stages"][2]["status"] == stage_status
    assert story["story_state"] == story_state
    assert story["outcome_improvement_authorized"] is False


@pytest.mark.parametrize(("reviewed", "status"), (
    ({"available": True, "selected_for_fixture": True}, "selected"),
    ({"available": True, "selected_for_fixture": False}, "reviewed_not_selected"),
    ({"available": False, "selected_for_fixture": False}, "not_used"),
))
def test_world_story_keeps_review_use_distinct_from_review_availability(
    reviewed, status,
):
    navigator = _completed_navigator()
    navigator["history_chapters"][0]["reviewed_future_context"] = reviewed

    story = build_manager_world_story(navigator)

    assert story["stages"][1]["status"] == status


def test_world_story_bounds_public_counts_and_never_infers_missing_world():
    navigator = _completed_navigator()
    chapter = navigator["history_chapters"][0]
    chapter["matchday"] = float("inf")
    chapter["action_adoption"]["influenced_decisions"] = 1_000_001
    chapter["action_adoption"]["locally_attributable_action_changes"] = -8
    chapter["locally_attributable_action_changes"] = -8
    chapter["persistent_transition_identity"] = None
    chapter["descriptive_world_after"] = {
        "result_available": False,
        "persistent_state_available": False,
    }

    story = build_manager_world_story(navigator)

    assert story["source"]["matchday"] == 0
    assert story["metrics"]["influenced_decisions"] == 100_000
    assert story["metrics"]["locally_attributable_action_changes"] == 0
    assert story["stages"][3]["status"] == "evidence_unavailable"
    assert story["stages"][4]["status"] == "evidence_unavailable"
    assert story["chapter_dossier"]["official_action"][
        "influenced_decisions"
    ] == 100_000
    assert story["chapter_dossier"]["official_action"][
        "locally_attributable_action_changes"
    ] == 0
    assert story["chapter_dossier"]["world_after"]["metrics_delta"] is None


def test_world_story_prioritizes_active_chapter_and_binds_previous_world():
    navigator = _completed_navigator()
    navigator["current_chapter"] = {
        "current_chapter_identity": "d" * 64,
        "workflow_state": "ready_to_explore",
        "frozen_decision_identity": "e" * 64,
        "fixture": {
            "fixture_id": "fixture-4",
            "matchday": 4,
        },
        "stages": [],
    }
    original = copy.deepcopy(navigator)

    story = build_manager_world_story(navigator)
    validate_manager_world_story(story, navigator=navigator)

    assert navigator == original
    assert story["view_mode"] == "active_chapter"
    assert story["source"]["fixture_id"] == "fixture-4"
    assert story["source"]["chapter_identity"] == "d" * 64
    assert story["source"]["navigable"] is False
    assert story["story_state"] == "awaiting_official_world"
    assert [row["status"] for row in story["stages"]] == [
        "complete", "pending", "pending", "pending", "pending",
    ]
    previous = story["previous_completed"]
    assert previous["source"]["fixture_id"] == "fixture-3"
    assert previous["source"]["chapter_identity"] == "b" * 64
    assert previous["source"]["navigable"] is True
    assert previous["story_state"] == "local_action_change_observed"
    assert previous["action_status"] == "changed"
    assert previous["persistent_world_status"] == "world_persisted"
    assert previous["outcome_status"] == "descriptive_result_only"
    assert previous["metrics"]["locally_attributable_action_changes"] == 2
    assert previous["chapter_dossier"]["manager_choice"][
        "selected_tactic"
    ] == "gegenpress"
    assert previous["same_chapter_evidence"] is True


def test_world_story_empty_completed_season_has_no_invented_chapter():
    navigator = {
        "available": True,
        "season_id": "empty-season",
        "navigator_identity": "1" * 64,
        "current_chapter": {
            "workflow_state": "season_complete",
            "fixture": None,
        },
        "history_chapters": [],
    }

    story = build_manager_world_story(navigator)

    assert story["available"] is False
    assert story["view_mode"] == "unavailable"
    assert story["source"] is None
