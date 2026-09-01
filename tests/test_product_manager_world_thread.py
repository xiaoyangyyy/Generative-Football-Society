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


def _entry(*, action_available=True, stable=False, review_linked=True):
    action = (
        {
            "schema_version": 1,
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


def test_world_evolution_thread_marks_stable_mode_optional_and_real_breaks():
    stable_entry = _entry(action_available=False, stable=True)
    stable = build_manager_world_evolution_thread(stable_entry)

    assert stable["official_runtime_chain_complete"] is True
    assert stable["world_model_runtime_chain_complete"] is False
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
