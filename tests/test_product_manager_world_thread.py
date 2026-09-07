import copy
import hashlib
import json

import pytest

from src.product.manager_world_thread import (
    build_manager_world_evolution_thread,
    manager_world_evolution_summary,
    validate_manager_world_evolution_thread,
)


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _action_transition_matrices(multiplier=1):
    actions = ("hold", "pass", "cross", "shot", "none")
    all_rows = {baseline: {actual: 0 for actual in actions} for baseline in actions}
    local_rows = copy.deepcopy(all_rows)
    for baseline, actual in (
        ("hold", "pass"), ("pass", "cross"),
        ("cross", "shot"), ("shot", "hold"),
    ):
        all_rows[baseline][actual] = multiplier
    local_rows["hold"]["pass"] = multiplier
    local_rows["pass"]["cross"] = multiplier
    return all_rows, local_rows


def _reviewed_scenarios():
    rows = [
        (900.0, "no_realized_action_divergence", 0, 0, 0, False, "1"),
        (1800.0, "local_action_divergence_with_descriptive_future_difference", 1, 1, 1, True, "2"),
        (2700.0, "local_action_divergence_without_measured_future_difference", 1, 1, 0, True, "3"),
    ]
    scenarios = []
    for branch, status, changed, local, differences, attributed, marker in rows:
        payload = {
            "schema_version": 3,
            "source_scenario_identity": marker * 64,
            "branch_at_sec": branch,
            "branch_minute": branch / 60.0,
            "future_status": status,
            "eligible": True,
            "anchor_verified": True,
            "branch_state_identity": marker * 64,
            "changed_actions": changed,
            "locally_attributable_changes": local,
            "descriptive_future_difference_count": differences,
            "simulator_local_action_attribution": attributed,
            "outcome_causality_authorized": False,
            "real_football_causality_authorized": False,
        }
        payload["archive_identity"] = _identity(payload)
        scenarios.append(payload)
    return scenarios


def _entry(*, action_available=True, stable=False, review_linked=True):
    action_examples = [
        {
            "actual_action": "cross",
            "policy_signal": {"mode": "direct_preference"},
            "reference_action_effect": {
                "received_redistributed_probability": False,
            },
            "runtime_link": {"direct_ball_event_identity": True},
        },
        {
            "actual_action": "pass",
            "policy_signal": {"mode": "suppression_only"},
            "reference_action_effect": {
                "received_redistributed_probability": False,
            },
            "runtime_link": {"direct_ball_event_identity": True},
        },
        {
            "actual_action": "hold",
            "policy_signal": {"mode": "suppression_only"},
            "reference_action_effect": {
                "received_redistributed_probability": True,
            },
            "runtime_link": {"direct_ball_event_identity": False},
        },
        {
            "actual_action": "shot",
            "policy_signal": {"mode": "none"},
            "reference_action_effect": {
                "received_redistributed_probability": False,
            },
            "runtime_link": {"direct_ball_event_identity": False},
        },
    ]
    action = (
        {
            "schema_version": 2,
            "available": True,
            "match_id": "match-1",
            "team": "A",
            "evidence_state": "locally_attributable_action_changes_observed",
            "evidence_identity": "e" * 64,
            "retained_record_evidence": {
                "records": 4,
                "resolved_action_decisions": 4,
                "influenced_decisions": 3,
                "attribution_eligible_decisions": 3,
                "locally_attributable_action_changes": 2,
                "direct_ball_event_links": 2,
                "changed_direct_ball_event_links": 2,
                "decision_only_no_trajectory": 1,
                "unresolved_or_missing_ball_event_links": 1,
                "source_match_opportunities": 4,
                "source_records_truncated": False,
                "manager_record_coverage_complete": True,
            },
            "examples": action_examples,
            "examples_truncated": False,
            "retained_record_semantics": {
                "schema_version": 2,
                "records": 4,
                "actual_action_counts": {
                    "hold": 1, "pass": 1, "cross": 1, "shot": 1,
                    "none": 0,
                },
                "primary_signal_action_counts": {
                    "hold": 0, "pass": 2, "cross": 1, "shot": 0,
                    "none": 1,
                },
                "signal_mode_counts": {
                    "direct_preference": 1, "suppression_only": 2,
                    "none": 1, "legacy_unclassified": 0,
                },
                "hold_reference_redistribution_records": 1,
                "direct_cross_ball_event_links": 1,
                "locally_attributable_cross_changes": 1,
                "retained_record_coverage_complete": True,
                "source_manager_record_coverage_complete": True,
                "full_source_distribution_authorized": True,
                "outcome_attribution_authorized": False,
                "counterfactual_action_transition_counts": (
                    _action_transition_matrices()[0]
                ),
                "locally_attributable_action_transition_counts": (
                    _action_transition_matrices()[1]
                ),
            },
        }
        if action_available else
        {
            "schema_version": 1,
            "available": False,
            "reason": (
                "world_model_not_configured_for_match"
                if stable else "world_model_action_evidence_invalid"
            ),
        }
    )
    return {
        "schema_version": 1,
        "fixture_id": "md01-fx01",
        "matchday": 1,
        "lifecycle_state": "executed_with_direct_evidence",
        "decision": {
            "team": "A", "tactic": "balanced", "rotation": "balanced",
        },
        "decision_identity": "d" * 64,
        "future_review_execution_trace": {
            "available": True,
            "trace_identity": "r" * 64,
            "terminal_review": {
                "review_identity": "q" * 64,
                "final_decision_identity": (
                    "d" * 64 if review_linked else "x" * 64
                ),
                "linked_to_final_selection": review_linked,
                "retained_mechanism_examples": 2,
                "cross_action_mechanism_examples": 1,
                "direct_preference_mechanism_examples": 1,
                "suppression_only_mechanism_examples": 1,
                "hold_reference_redistribution_examples": 1,
                "nonzero_cross_descriptive_windows": 2,
                "intent": "keep_after_review",
                "evidence_level": "scenario_evidence",
                "fixed_scenario_budget": 3,
                "eligible_scenarios": 3,
                "verified_anchor_scenarios": 3,
                "action_divergence_scenarios": 2,
                "local_attribution_scenarios": 2,
                "descriptive_future_difference_scenarios": 1,
                "timing_sensitivity_observed": True,
                "ranking_performed": False,
                "best_branch_time": None,
                "reviewed_scenarios": _reviewed_scenarios(),
            },
        },
        "execution": {
            "available": True,
            "match_id": "match-1",
            "tactical_binding": {
                "available": True,
                "binding_identity": "b" * 64,
                "applied_tactic": "balanced",
            },
            "world_model_action_execution": action,
        },
        "observed_result": {
            "score": {"home": 1, "away": 0},
            "outcome": "win",
            "points_earned": 3,
            "descriptive_only": True,
        },
        "long_term_accounting": {
            "persistent_team_state_delta": {
                "available": True,
                "transition_identity": "t" * 64,
                "recovery_complete": True,
                "match_delta": {
                    "metrics_delta": {
                        "team_fatigue_ema": 0.08,
                        "squad_morale_ema": 0.02,
                    },
                    "summary": {
                        "new_injuries": 1,
                        "new_suspensions": 0,
                        "changed_players": 7,
                    },
                },
            },
        },
    }


def test_world_evolution_thread_connects_six_stages_without_inventing_causality():
    entry = _entry()

    thread = build_manager_world_evolution_thread(entry)

    assert thread["thread_state"] == "official_world_evolution_observed"
    assert thread["official_runtime_chain_complete"] is True
    assert thread["world_model_runtime_chain_complete"] is True
    assert thread["reviewed_world_model_chain_complete"] is True
    certificate = thread["review_to_official_world"]
    assert certificate["status"] == "complete"
    assert certificate["scenario_archive_count"] == 3
    assert certificate["scenario_evidence_retained"] is True
    assert certificate["final_decision_identity_bound"] is True
    assert certificate["runtime_tactic_verified"] is True
    assert certificate["official_action_same_match"] is True
    assert certificate[
        "scenario_to_runtime_opportunity_matching_performed"
    ] is False
    assert certificate["outcome_comparison_performed"] is False
    assert certificate["causal_effect_authorized"] is False
    assert thread["continuity_gaps"] == []
    assert [stage["stage_id"] for stage in thread["stages"]] == [
        "prematch_future_review",
        "frozen_manager_decision",
        "official_tactical_runtime",
        "official_world_model_actions",
        "observed_match_result",
        "persistent_world_state",
    ]
    assert len({stage["stage_identity"] for stage in thread["stages"]}) == 6
    assert thread["stages"][3][
        "locally_attributable_action_changes"
    ] == 2
    official = thread["stages"][3]
    assert official["retained_semantic_examples"] == 4
    assert official["semantic_examples_truncated"] is False
    assert official["semantic_example_coverage_complete"] is True
    assert official["semantic_cross_action_examples"] == 1
    assert official["semantic_direct_preference_examples"] == 1
    assert official["semantic_suppression_only_examples"] == 2
    assert official["semantic_no_signal_examples"] == 1
    assert official["semantic_legacy_unclassified_examples"] == 0
    assert official[
        "semantic_hold_reference_redistribution_examples"
    ] == 1
    assert official["semantic_direct_cross_ball_event_examples"] == 1
    assert official["retained_record_semantics"][
        "actual_action_counts"
    ]["cross"] == 1
    assert official["retained_record_semantics"][
        "full_source_distribution_authorized"
    ] is True
    assert thread["stages"][0]["review_intent"] == "keep_after_review"
    assert thread["stages"][0]["cross_action_mechanism_examples"] == 1
    assert thread["stages"][0]["direct_preference_mechanism_examples"] == 1
    assert thread["stages"][0]["suppression_only_mechanism_examples"] == 1
    assert thread["stages"][0][
        "hold_reference_redistribution_examples"
    ] == 1
    assert thread["stages"][0]["nonzero_cross_descriptive_windows"] == 2
    assert thread["stages"][0]["fixed_scenario_budget"] == 3
    assert thread["stages"][0]["action_divergence_scenarios"] == 2
    assert thread["stages"][0]["local_attribution_scenarios"] == 2
    assert thread["stages"][0]["timing_sensitivity_observed"] is True
    assert thread["stages"][0]["ranking_performed"] is False
    assert thread["stages"][0]["best_branch_time"] is None
    assert thread["stages"][0]["reviewed_scenarios"] == (
        entry["future_review_execution_trace"]["terminal_review"][
            "reviewed_scenarios"
        ]
    )
    assert len(thread["stages"][0]["reviewed_scenarios"]) == 3
    assert thread["stages"][5]["metrics_delta"]["team_fatigue_ema"] == 0.08
    assert [link["status"] for link in thread["links"]] == [
        "identity_bound",
        "direct_runtime_binding",
        "same_official_match_context",
        "direct_identity_links_available",
        "engine_state_transition_recorded",
    ]
    assert all(
        link["causal_effect_authorized"] is False
        for link in thread["links"]
    )
    assert thread["links"][2]["link_type"] == (
        "same_match_context_not_causal_direction"
    )
    assert thread["links"][3][
        "downstream_result_attribution_authorized"
    ] is False
    assert thread["causal_effect_authorized"] is False
    assert thread["outcome_effect_estimate"] is None
    validate_manager_world_evolution_thread(thread, entry=entry)


def test_official_action_semantics_keep_truncation_distinct_from_coverage():
    entry = _entry()
    action = entry["execution"]["world_model_action_execution"]
    action["examples"] = action["examples"][:3]
    action["examples_truncated"] = True

    stage = build_manager_world_evolution_thread(entry)["stages"][3]

    assert stage["retained_records"] == 4
    assert stage["retained_semantic_examples"] == 3
    assert stage["semantic_examples_truncated"] is True
    assert stage["semantic_example_coverage_complete"] is False


def test_world_evolution_thread_marks_stable_mode_optional_and_real_breaks():
    stable_entry = _entry(action_available=False, stable=True)
    stable = build_manager_world_evolution_thread(stable_entry)

    assert stable["official_runtime_chain_complete"] is True
    assert stable["world_model_runtime_chain_complete"] is False
    assert stable["reviewed_world_model_chain_complete"] is False
    assert stable["review_to_official_world"]["status"] == (
        "official_action_evidence_unavailable"
    )
    assert stable["thread_state"] == (
        "official_match_observed_action_optional_or_unavailable"
    )
    assert stable["continuity_gaps"] == []
    assert stable["stages"][3]["status"] == "not_applicable_stable_mode"
    assert stable["links"][2]["status"] == "not_applicable"

    broken_entry = _entry(action_available=False, review_linked=False)
    broken = build_manager_world_evolution_thread(broken_entry)
    assert broken["continuity_gaps"] == [
        "review_to_final_decision_continuity_broken",
        "world_model_action_evidence_unavailable",
    ]
    assert broken["links"][0]["status"] == "continuity_broken"
    assert broken["world_model_runtime_chain_complete"] is False
    assert broken["reviewed_world_model_chain_complete"] is False
    assert broken["review_to_official_world"]["status"] == (
        "review_superseded"
    )

    tactical_missing_entry = _entry()
    tactical_missing_entry["execution"]["tactical_binding"] = {
        "available": False,
        "reason": "binding_not_retained",
    }
    tactical_missing = build_manager_world_evolution_thread(
        tactical_missing_entry
    )
    assert tactical_missing["reviewed_world_model_chain_complete"] is False
    assert tactical_missing["review_to_official_world"]["status"] == (
        "tactical_runtime_binding_unavailable"
    )


def test_world_evolution_thread_pending_summary_and_rehashed_tamper_fail_closed():
    pending_entry = {
        "schema_version": 1,
        "fixture_id": "md02-fx01",
        "matchday": 2,
        "lifecycle_state": "frozen_awaiting_execution",
        "decision": {
            "team": "A", "tactic": "gegenpress", "rotation": "balanced",
        },
        "decision_identity": "p" * 64,
        "future_review_execution_trace": {
            "available": False,
            "reason": "world_model_future_review_not_available",
        },
        "execution": {
            "available": False,
            "reason": "fixture_not_completed",
        },
        "observed_result": None,
        "long_term_accounting": {
            "persistent_team_state_delta": {
                "available": False,
                "reason": "per_fixture_persisted_state_snapshot_not_retained",
            },
        },
    }
    pending = build_manager_world_evolution_thread(pending_entry)
    complete_entry = _entry()
    complete = build_manager_world_evolution_thread(complete_entry)
    pending_entry["world_evolution_thread"] = pending
    complete_entry["world_evolution_thread"] = complete

    summary = manager_world_evolution_summary(
        [pending_entry, complete_entry],
    )

    assert summary["fixtures_with_world_evolution_thread"] == 2
    assert summary["awaiting_official_match"] == 1
    assert summary["official_runtime_chain_complete"] == 1
    assert summary["world_model_runtime_chain_complete"] == 1
    assert summary["reviewed_world_model_chain_complete"] == 1
    assert summary["review_to_official_world_status_counts"] == {
        "complete": 1,
        "awaiting_official_match": 0,
        "review_superseded": 0,
        "tactical_runtime_binding_unavailable": 0,
        "official_action_evidence_unavailable": 0,
        "not_reviewed": 1,
    }
    assert summary["local_action_changes"] == 2
    assert summary["causal_effect_authorized"] is False

    tampered = copy.deepcopy(complete)
    tampered["links"][2]["status"] = "caused_world_model_actions"
    tampered_link = copy.deepcopy(tampered["links"][2])
    tampered_link.pop("link_identity")
    tampered["links"][2]["link_identity"] = _identity(tampered_link)
    frozen = copy.deepcopy(tampered)
    frozen.pop("thread_identity")
    tampered["thread_identity"] = _identity(frozen)
    with pytest.raises(ValueError, match="thread replay mismatch"):
        validate_manager_world_evolution_thread(
            tampered, entry=complete_entry,
        )

    certificate_tampered = copy.deepcopy(complete)
    certificate = certificate_tampered["review_to_official_world"]
    certificate["scenario_to_runtime_opportunity_matching_performed"] = True
    frozen_certificate = copy.deepcopy(certificate)
    frozen_certificate.pop("certificate_identity")
    certificate["certificate_identity"] = _identity(frozen_certificate)
    frozen_thread = copy.deepcopy(certificate_tampered)
    frozen_thread.pop("thread_identity")
    certificate_tampered["thread_identity"] = _identity(frozen_thread)
    with pytest.raises(ValueError, match="thread replay mismatch"):
        validate_manager_world_evolution_thread(
            certificate_tampered, entry=complete_entry,
        )


def test_world_evolution_thread_keeps_society_continuity_inside_world_state():
    entry = _entry()
    society = {
        "available": True,
        "before_state_identity": "b" * 64,
        "after_state_identity": "a" * 64,
        "source_transaction_id": "season-1:fixture-1",
        "memory_record_delta": 3,
        "cognitive_memory_delta": 2,
        "belief_delta": 1,
        "reflection_delta": 0,
        "changed_state_fields": ["emotion_profile", "tactical_controls"],
        "claim_boundary": "persisted simulator state only",
    }
    entry["long_term_accounting"]["persistent_team_state_delta"][
        "match_delta"
    ]["society_transition"] = society

    thread = build_manager_world_evolution_thread(entry)
    persistent = thread["stages"][-1]

    assert persistent["stage_id"] == "persistent_world_state"
    assert persistent["society_transition"] == society
    assert len(thread["stages"]) == 6
    assert len(thread["links"]) == 5
    validate_manager_world_evolution_thread(thread, entry=entry)
    entry["world_evolution_thread"] = thread
    summary = manager_world_evolution_summary([entry])
    assert summary["society_continuity_transitions"] == 1
