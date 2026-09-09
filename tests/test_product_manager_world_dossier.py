import copy

import pytest

from scripts.verify_product_recovery import CODE_IDENTITY_FILES
from src.product.manager_world_dossier import (
    CLAIM_BOUNDARY,
    build_manager_world_dossier,
)


def _local_transition_matrix() -> dict:
    actions = ("hold", "pass", "cross", "shot", "none")
    matrix = {
        before: {after: 0 for after in actions}
        for before in actions
    }
    matrix["hold"]["pass"] = 1
    matrix["pass"]["cross"] = 1
    return matrix


def _player_changes() -> list[dict]:
    return [{
        "player_id": "p01",
        "name": "Alex One",
        "change": "updated",
        "fields": {
            "matches_played": {"before": 4, "after": 5},
            "injury_matches_left": {"before": 0, "after": 2},
        },
    }, {
        "player_id": "p02",
        "name": "Blake Two",
        "change": "updated",
        "fields": {
            "form_ema": {"before": 0.1, "after": 0.2},
        },
    }]


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
            "retained_record_semantics": {
                "schema_version": 3,
                "locally_attributable_action_transition_counts": (
                    _local_transition_matrix()
                ),
                "expected_change_estimator": (
                    "shared_uniform_inverse_cdf_overlap_v1"
                ),
                "expected_counterfactual_action_changes": 1.75,
                "full_source_distribution_authorized": True,
                "full_source_expectation_authorized": True,
            },
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
                "changed_players": 2,
                "new_injuries": 1,
                "injuries_cleared": 0,
                "new_suspensions": 0,
                "suspensions_cleared": 0,
            },
            "players_changed": _player_changes(),
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
    assert dossier["schema_version"] == 3
    assert dossier["phase"] == "completed"
    assert dossier["manager_choice"]["selected_tactic"] == "gegenpress"
    assert dossier["future_review"]["registered_scenarios"] == 3
    assert dossier["official_action"]["locally_attributable_action_changes"] == 2
    transitions = dossier["official_action"]["transition_evidence"]
    assert transitions == {
        "available": True,
        "reason": None,
        "schema_version": 3,
        "transitions": [{
            "transition_id": "hold_to_pass",
            "baseline_action": "hold",
            "actual_action": "pass",
            "count": 1,
        }, {
            "transition_id": "pass_to_cross",
            "baseline_action": "pass",
            "actual_action": "cross",
            "count": 1,
        }],
        "observed_transition_total": 2,
        "counts_match_official": True,
        "expectation_available": True,
        "expected_counterfactual_action_changes": 1.75,
        "full_source_transition_distribution_authorized": True,
        "full_source_expectation_authorized": True,
        "same_chapter_cooccurrence_only": True,
        "outcome_attribution_authorized": False,
    }
    assert dossier["world_after"]["metrics_delta"]["squad_morale_ema"] == 0.02
    player_changes = dossier["world_after"]["player_changes"]
    assert player_changes["available"] is True
    assert player_changes["total_changed"] == 2
    assert player_changes["players"][0]["name"] == "Alex One"
    assert player_changes["players"][0]["fields"] == [{
        "field": "matches_played", "before": 4, "after": 5,
    }, {
        "field": "injury_matches_left", "before": 0, "after": 2,
    }]
    assert player_changes["manager_or_action_effect_authorized"] is False
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
    assert dossier["official_action"]["transition_evidence"][
        "available"
    ] is False
    assert dossier["official_action"]["transition_evidence"][
        "reason"
    ] == "transition_total_mismatch"
    assert dossier["world_after"]["metrics_delta"]["team_fatigue_ema"] is None
    assert dossier["world_after"]["society_transition"][
        "changed_state_fields"
    ] == ["valid"]
    assert dossier["continuity_gaps"] == ["known_gap"]


def test_dossier_keeps_missing_player_detail_distinct_from_zero_change():
    sources = _completed_sources()
    sources["world"].pop("players_changed")

    dossier = build_manager_world_dossier(**sources)

    player_changes = dossier["world_after"]["player_changes"]
    assert player_changes["available"] is False
    assert player_changes["reason"] == (
        "legacy_or_missing_player_change_evidence"
    )
    assert player_changes["total_changed"] == 2
    assert player_changes["players"] == []


def test_dossier_fails_closed_on_invalid_transition_matrix():
    sources = _completed_sources()
    matrix = sources["action"]["retained_record_semantics"][
        "locally_attributable_action_transition_counts"
    ]
    matrix["hold"]["hold"] = 1

    dossier = build_manager_world_dossier(**sources)

    evidence = dossier["official_action"]["transition_evidence"]
    assert evidence["available"] is False
    assert evidence["reason"] == "invalid_or_incomplete_transition_matrix"
    assert evidence["transitions"] == []
    assert evidence["counts_match_official"] is False
    assert evidence["outcome_attribution_authorized"] is False


def test_dossier_keeps_legacy_transition_absence_explicit():
    sources = _completed_sources()
    sources["action"].pop("retained_record_semantics")

    dossier = build_manager_world_dossier(**sources)

    evidence = dossier["official_action"]["transition_evidence"]
    assert evidence["available"] is False
    assert evidence["reason"] == "legacy_or_missing_transition_semantics"
    assert evidence["expected_counterfactual_action_changes"] is None


def test_dossier_rejects_unknown_chapter_phase():
    sources = _completed_sources()
    sources["phase"] = "causal_claim"

    with pytest.raises(ValueError, match="phase is invalid"):
        build_manager_world_dossier(**sources)


def test_dossier_is_bound_into_product_recovery_identity():
    assert "src/product/manager_world_dossier.py" in CODE_IDENTITY_FILES
