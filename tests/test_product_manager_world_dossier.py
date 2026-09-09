import copy

import pytest

from scripts.verify_product_recovery import CODE_IDENTITY_FILES
from src.product.manager_world_dossier import (
    CLAIM_BOUNDARY,
    build_manager_world_dossier,
)


def _completed_sources() -> dict:
    return {
        "phase": "completed",
        "choice": {
            "decision_identity": "d" * 64,
            "selected_tactic": "gegenpress",
            "rotation": "balanced",
            "applied_tactic": "gegenpress",
            "runtime_verified": True,
            "runtime_matches_selection": True,
        },
        "review": {
            "available": True,
            "intent": "keep_after_review",
            "evidence_level": "scenario_evidence",
            "fixed_scenario_budget": 3,
            "eligible_scenarios": 2,
            "action_divergence_scenarios": 2,
            "local_attribution_scenarios": 1,
            "descriptive_future_difference_scenarios": 1,
            "timing_sensitivity_observed": True,
        },
        "review_status": "selected",
        "action": {
            "retained_records": 6,
            "influenced_decisions": 4,
            "attribution_eligible_decisions": 3,
            "locally_attributable_action_changes": 2,
            "direct_ball_event_links": 3,
            "manager_record_coverage_complete": True,
        },
        "action_status": "changed",
        "world": {
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
        "world_status": "world_persisted",
        "continuity_gaps": [],
    }


def test_dossier_unifies_one_chapter_without_mutating_or_authorizing_effect():
    sources = _completed_sources()
    original = copy.deepcopy(sources)

    dossier = build_manager_world_dossier(**sources)

    assert sources == original
    assert dossier["schema_version"] == 1
    assert dossier["phase"] == "completed"
    assert dossier["manager_choice"]["selected_tactic"] == "gegenpress"
    assert dossier["future_review"]["registered_scenarios"] == 3
    assert dossier["official_action"]["locally_attributable_action_changes"] == 2
    assert dossier["world_after"]["metrics_delta"]["squad_morale_ema"] == 0.02
    assert dossier["world_after"]["society_transition"][
        "changed_state_fields"
    ] == ["emotion_profile", "tactical_controls"]
    assert dossier["same_chapter_evidence"] is True
    assert dossier["outcome_improvement_authorized"] is False
    assert dossier["causal_effect_authorized"] is False
    assert dossier["claim_boundary"] == CLAIM_BOUNDARY


def test_dossier_bounds_invalid_public_values_without_inventing_evidence():
    sources = _completed_sources()
    sources["choice"]["selected_tactic"] = "x" * 161
    sources["action"].update({
        "retained_records": float("inf"),
        "influenced_decisions": 1_000_001,
        "locally_attributable_action_changes": True,
    })
    sources["world"]["metrics_delta"]["team_fatigue_ema"] = float("nan")
    sources["world"]["society_transition"]["changed_state_fields"] = [
        "valid", "x" * 161, 42,
    ]
    sources["continuity_gaps"] = ["known_gap", "", 42]

    dossier = build_manager_world_dossier(**sources)

    assert dossier["manager_choice"]["available"] is False
    assert dossier["official_action"]["retained_records"] == 0
    assert dossier["official_action"]["influenced_decisions"] == 100_000
    assert dossier["official_action"][
        "locally_attributable_action_changes"
    ] == 0
    assert dossier["world_after"]["metrics_delta"]["team_fatigue_ema"] is None
    assert dossier["world_after"]["society_transition"][
        "changed_state_fields"
    ] == ["valid"]
    assert dossier["continuity_gaps"] == ["known_gap"]


def test_dossier_rejects_unknown_chapter_phase():
    sources = _completed_sources()
    sources["phase"] = "causal_claim"

    with pytest.raises(ValueError, match="phase is invalid"):
        build_manager_world_dossier(**sources)


def test_dossier_is_bound_into_product_recovery_identity():
    assert "src/product/manager_world_dossier.py" in CODE_IDENTITY_FILES
