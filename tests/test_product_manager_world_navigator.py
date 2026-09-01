import copy
import hashlib
import json

import pytest

from src.product.manager_world_navigator import (
    MAX_HISTORY_CHAPTERS,
    build_manager_world_navigator,
    validate_manager_world_navigator,
)


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _freeze(payload, field):
    frozen = copy.deepcopy(payload)
    frozen[field] = _identity(frozen)
    return frozen


def _stage(stage_id, status, **facts):
    return _freeze({
        "stage_id": stage_id,
        "status": status,
        **facts,
    }, "stage_identity")


def _reviewed_scenarios():
    rows = [
        (900.0, "no_realized_action_divergence", 0, 0, 0, False, "1"),
        (1800.0, "local_action_divergence_with_descriptive_future_difference", 1, 1, 1, True, "2"),
        (2700.0, "local_action_divergence_without_measured_future_difference", 1, 1, 0, True, "3"),
    ]
    return [
        _freeze({
            "schema_version": 1,
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
        }, "archive_identity")
        for branch, status, changed, local, differences, attributed, marker
        in rows
    ]


def _future_trace():
    terminal = {
        "review_identity": "c" * 64,
        "task_id": "future1",
        "intent": "keep_after_review",
        "final_decision_identity": "d" * 64,
        "linked_to_final_selection": True,
        "retained_mechanism_examples": 2,
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
    }
    return _freeze({
        "schema_version": 1,
        "available": True,
        "review_chain": [],
        "terminal_review": terminal,
        "final_selection": {
            "decision_identity": "d" * 64,
            "tactic": "balanced",
        },
        "runtime_binding": {"status": "verified"},
        "end_to_end_state": "reviewed_selection_runtime_verified",
        "outcome_comparison_performed": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": "test reviewed future execution trace",
    }, "trace_identity")


def _thread(fixture_id="md01-fx01", *, complete=True, gaps=None):
    trace = _future_trace()
    terminal = trace["terminal_review"]
    stages = [
        _stage(
            "prematch_future_review", "selected_for_fixture",
            source_identity=trace["trace_identity"],
            review_identity=terminal["review_identity"],
            retained_mechanism_examples=terminal[
                "retained_mechanism_examples"
            ],
            review_intent=terminal["intent"],
            evidence_level=terminal["evidence_level"],
            fixed_scenario_budget=terminal["fixed_scenario_budget"],
            eligible_scenarios=terminal["eligible_scenarios"],
            verified_anchor_scenarios=terminal[
                "verified_anchor_scenarios"
            ],
            action_divergence_scenarios=terminal[
                "action_divergence_scenarios"
            ],
            local_attribution_scenarios=terminal[
                "local_attribution_scenarios"
            ],
            descriptive_future_difference_scenarios=terminal[
                "descriptive_future_difference_scenarios"
            ],
            timing_sensitivity_observed=terminal[
                "timing_sensitivity_observed"
            ],
            ranking_performed=False,
            best_branch_time=None,
            reviewed_scenarios=terminal["reviewed_scenarios"],
            evidence_authority="simulator_mechanism_review_only",
        ),
        _stage("frozen_manager_decision", "frozen"),
        _stage("official_tactical_runtime", "runtime_verified"),
        _stage(
            "official_world_model_actions",
            "locally_attributable_action_changes_observed",
            retained_records=4,
            resolved_action_decisions=4,
            influenced_decisions=3,
            attribution_eligible_decisions=3,
            locally_attributable_action_changes=2,
            direct_ball_event_links=2,
            changed_direct_ball_event_links=2,
            decision_only_no_trajectory=1,
            unresolved_ball_event_links=1,
            source_match_opportunities=4,
            source_records_truncated=False,
            manager_record_coverage_complete=True,
            source_identity="e" * 64,
        ),
        _stage(
            "observed_match_result", "observed_descriptive",
            source_identity="a" * 64,
        ),
        _stage(
            "persistent_world_state",
            "after_recovery_recorded",
            source_identity="f" * 64,
            recovery_complete=True,
            metrics_delta={
                "team_fatigue_ema": 0.08,
                "squad_morale_ema": 0.02,
                "team_media_pressure": 0.01,
                "injured_players": 1.0,
                "suspended_players": 0.0,
                "unavailable_players": 1.0,
            },
            transition_summary={
                "changed_players": 7,
                "new_injuries": 1,
                "injuries_cleared": 0,
                "new_suspensions": 0,
                "suspensions_cleared": 0,
            },
        ),
    ]
    return _freeze({
        "schema_version": 1,
        "available": True,
        "fixture_id": fixture_id,
        "thread_state": "official_world_evolution_observed",
        "stages": stages,
        "links": [],
        "continuity_gaps": list(gaps or []),
        "official_runtime_chain_complete": complete,
        "world_model_runtime_chain_complete": complete,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
    }, "thread_identity")


def _entry(index=1, *, pending=False, gaps=None):
    fixture_id = f"md{index:02d}-fx01"
    payload = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "matchday": index,
        "lifecycle_state": (
            "frozen_awaiting_execution"
            if pending else "executed_with_direct_evidence"
        ),
        "observed_result": (
            None if pending else {
                "score": {"home": index % 3, "away": 1},
                "outcome": "win" if index % 3 > 1 else "draw",
                "points_earned": 3 if index % 3 > 1 else 1,
                "descriptive_only": True,
            }
        ),
        "long_term_accounting": {
            "persistent_team_state_delta": {
                "available": not pending,
                "transition_identity": None if pending else "f" * 64,
                "recovery_complete": not pending,
                "match_delta": None if pending else {
                    "metrics_delta": {
                        "team_fatigue_ema": 0.08,
                        "squad_morale_ema": 0.02,
                        "team_media_pressure": 0.01,
                        "injured_players": 1.0,
                        "suspended_players": 0.0,
                        "unavailable_players": 1.0,
                    },
                    "summary": {
                        "changed_players": 7,
                        "new_injuries": 1,
                        "injuries_cleared": 0,
                        "new_suspensions": 0,
                        "suspensions_cleared": 0,
                    },
                },
            },
        },
        "world_evolution_thread": _thread(
            fixture_id, complete=not pending, gaps=gaps,
        ),
        "future_review_execution_trace": _future_trace(),
    }
    if not pending:
        result_stage = next(
            row for row in payload["world_evolution_thread"]["stages"]
            if row["stage_id"] == "observed_match_result"
        )
        result_stage["source_identity"] = _identity(
            payload["observed_result"]
        )
        result_stage["score"] = copy.deepcopy(
            payload["observed_result"]["score"]
        )
        result_stage["outcome"] = payload["observed_result"]["outcome"]
        result_stage["points_earned"] = payload["observed_result"][
            "points_earned"
        ]
        result_stage["descriptive_only"] = True
        result_stage.pop("stage_identity")
        result_stage["stage_identity"] = _identity(result_stage)
        thread = payload["world_evolution_thread"]
        thread.pop("thread_identity")
        thread["thread_identity"] = _identity(thread)
    return _freeze(payload, "entry_identity")


def _workspace(workflow_state="evidence_ready_for_review"):
    allowed = {
        "freeze_decision": workflow_state == "decision_required",
        "request_future_set": workflow_state != "decision_required",
        "review_future_set": (
            workflow_state == "evidence_ready_for_review"
        ),
        "advance_official_match": workflow_state != "decision_required",
    }
    payload = {
        "schema_version": 1,
        "available": workflow_state != "season_complete",
        "workflow_state": workflow_state,
        "season_id": "season-1",
        "season_revision": 5,
        "fixture": (
            None if workflow_state == "season_complete" else {
                "fixture_id": "md03-fx01",
                "matchday": 3,
                "home": "A",
                "away": "B",
            }
        ),
        "frozen_intervention": (
            None if workflow_state in {
                "decision_required", "season_complete",
            } else {"decision_identity": "d" * 64}
        ),
        "stages": [
            _stage("freeze_intervention", "decision_frozen"),
            _stage("generate_bounded_futures", "evidence_complete"),
        ] if workflow_state != "season_complete" else [],
        "evidence_summary": {
            "future_sets": 1,
            "registered_scenarios": 3,
        },
        "allowed_actions": allowed,
        "continuity_gaps": [],
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": "bounded manager intervention workspace",
    }
    return _freeze(payload, "workspace_identity")


def _season(workflow_state="evidence_ready_for_review", entries=None):
    return {
        "season_id": "season-1",
        "revision": 5,
        "manager_intervention_workspace": _workspace(workflow_state),
        "manager_decision_ledger": {
            "schema_version": 1,
            "entries": list(entries or []),
        },
    }


def _rewrite_action(entry, status, *, source=True, **facts):
    action = next(
        row for row in entry["world_evolution_thread"]["stages"]
        if row["stage_id"] == "official_world_model_actions"
    )
    action["status"] = status
    action["source_identity"] = "e" * 64 if source else None
    action.update(facts)
    action.pop("stage_identity")
    action["stage_identity"] = _identity(action)
    thread = entry["world_evolution_thread"]
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)
    return entry


def test_navigator_joins_current_review_and_completed_world_chapters():
    season = _season(entries=[
        _entry(1),
        _entry(2, gaps=["world_model_action_evidence_unavailable"]),
        _entry(3, pending=True),
    ])

    navigator = build_manager_world_navigator(season)
    propagation = {
        "chapters": 2,
        "results_available": 2,
        "outcomes": {"win": 1, "draw": 1, "loss": 0},
        "points_earned": 4,
        "persistent_state_chapters": 2,
        "recovery_complete_chapters": 2,
        "match_metrics_delta_totals": {
            "team_fatigue_ema": 0.16,
            "squad_morale_ema": 0.04,
            "team_media_pressure": 0.02,
            "injured_players": 2.0,
            "suspended_players": 0.0,
            "unavailable_players": 2.0,
        },
        "transition_summary_totals": {
            "changed_players": 14,
            "new_injuries": 2,
            "injuries_cleared": 0,
            "new_suspensions": 0,
            "suspensions_cleared": 0,
        },
        "descriptive_cooccurrence_only": True,
        "state_effect_comparison_authorized": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
    }

    assert navigator["navigation_state"] == "active_manager_world"
    assert navigator["primary_action"]["action_id"] == (
        "review_future_evidence"
    )
    assert [
        row["fixture_id"] for row in navigator["history_chapters"]
    ] == ["md02-fx01", "md01-fx01"]
    assert navigator["history_chapters"][0][
        "locally_attributable_action_changes"
    ] == 2
    assert navigator["history_chapters"][0][
        "persistent_transition_identity"
    ] == "f" * 64
    assert navigator["summary"] == {
        "completed_world_chapters": 2,
        "visible_world_chapters": 2,
        "chapters_truncated": False,
        "official_runtime_chapters": 2,
        "world_model_runtime_chapters": 2,
        "local_action_changes": 4,
        "chapters_with_continuity_gaps": 1,
        "chapters_without_continuity_gaps": 1,
        "continuity_gap_counts": [{
            "gap": "world_model_action_evidence_unavailable",
            "chapters": 1,
            "latest_fixture_id": "md02-fx01",
            "latest_matchday": 2,
            "latest_chapter_identity": navigator["history_chapters"][0][
                "chapter_identity"
            ],
        }],
        "visible_official_runtime_chapters": 2,
        "visible_world_model_runtime_chapters": 2,
        "visible_local_action_changes": 4,
        "visible_chapters_with_continuity_gaps": 1,
        "world_model_influence_path": {
            "review_selected": 2,
                "reviewed_future_scenario_evidence": 2,
                "reviewed_future_scenario_archives": 6,
            "reviewed_future_action_divergence": 2,
            "reviewed_future_local_attribution": 2,
            "reviewed_future_timing_sensitivity": 2,
            "runtime_tactic_verified": 2,
            "official_action_evidence": 2,
            "chapters_with_local_action_changes": 2,
            "persistent_world_transitions": 2,
            "complete_world_model_runtime_chains": 2,
        },
        "world_model_action_adoption_ledger": {
            "fixtures_with_action_evidence": 2,
            "fixtures_with_complete_record_coverage": 2,
            "fixtures_with_incomplete_record_coverage": 0,
            "retained_records": 8,
            "resolved_action_decisions": 8,
            "influenced_decisions": 6,
            "attribution_eligible_decisions": 6,
            "locally_attributable_action_changes": 4,
            "direct_ball_event_links": 4,
            "changed_direct_ball_event_links": 4,
            "decision_only_no_trajectory": 2,
            "unresolved_ball_event_links": 2,
            "source_match_opportunities": 8,
            "influence_rate": 0.75,
            "realized_change_rate_among_influenced": 0.666667,
            "direct_ball_event_link_coverage": 0.666667,
            "state_counts": [{
                "state": "realized_action_change",
                "chapters": 2,
                "latest_fixture_id": "md02-fx01",
                "latest_matchday": 2,
                "latest_chapter_identity": navigator["history_chapters"][0][
                    "chapter_identity"
                ],
                "descriptive_world_after": propagation,
            }],
            "descriptive_world_after": propagation,
        },
    }
    assert navigator["causal_effect_authorized"] is False
    assert navigator["outcome_effect_estimate"] is None
    validate_manager_world_navigator(navigator, season=season)


def test_world_trajectory_is_chronological_identity_bound_and_cumulative():
    navigator = build_manager_world_navigator(_season(entries=[
        _entry(2), _entry(1),
    ]))

    trajectory = navigator["world_trajectory"]
    assert trajectory["total_points"] == 2
    assert trajectory["visible_points"] == 2
    assert trajectory["points_truncated"] is False
    assert trajectory["results_available"] == 2
    assert trajectory["persistent_state_chapters"] == 2
    assert trajectory["reviewed_future_chapters"] == 2
    assert trajectory["reviewed_future_selected_chapters"] == 2
    assert trajectory["scenario_evidence_chapters"] == 2
    assert trajectory["reviewed_scenario_archives"] == 6
    assert trajectory["identity_verified_scenario_archives"] == 6
    assert trajectory["reviewed_action_divergence_chapters"] == 2
    assert trajectory["reviewed_timing_sensitivity_chapters"] == 2
    assert trajectory["future_to_outcome_comparison_performed"] is False
    assert trajectory["descriptive_chronology_only"] is True
    assert trajectory["missing_evidence_imputed"] is False
    assert trajectory["turning_point_inference_authorized"] is False
    assert trajectory["outcome_effect_estimate"] is None
    assert trajectory["causal_effect_authorized"] is False
    first, second = trajectory["points"]
    assert [first["matchday"], second["matchday"]] == [1, 2]
    assert [first["sequence"], second["sequence"]] == [1, 2]
    assert first["chapter_identity"] == next(
        row["chapter_identity"] for row in navigator["history_chapters"]
        if row["fixture_id"] == "md01-fx01"
    )
    assert first["reviewed_future_context"] == {
        "available": True,
        "selected_for_fixture": True,
        "review_identity": "c" * 64,
        "intent": "keep_after_review",
        "evidence_level": "scenario_evidence",
        "fixed_scenario_budget": 3,
        "eligible_scenarios": 3,
        "verified_anchor_scenarios": 3,
        "action_divergence_scenarios": 2,
        "local_attribution_scenarios": 2,
        "descriptive_future_difference_scenarios": 1,
        "retained_mechanism_examples": 2,
        "timing_sensitivity_observed": True,
        "ranking_performed": False,
        "best_branch_time": None,
        "scenarios": _reviewed_scenarios(),
        "outcome_comparison_performed": False,
        "causal_effect_authorized": False,
    }
    assert first["cumulative"]["points_earned"] == 1
    assert second["cumulative"]["points_earned"] == 4
    assert second["cumulative"][
        "locally_attributable_action_changes"
    ] == 4
    assert second["cumulative"]["match_metrics_delta_totals"][
        "team_fatigue_ema"
    ] == 0.16
    assert second["cumulative"]["transition_summary_totals"][
        "new_injuries"
    ] == 2
    assert first["world_change_markers"] == [
        "local_action_change", "new_injury",
    ]
    assert first["descriptive_chronology_only"] is True
    assert first["turning_point_inference_authorized"] is False
    assert first["causal_effect_authorized"] is False
    assert _identity({
        key: value for key, value in first.items()
        if key != "trajectory_point_identity"
    }) == first["trajectory_point_identity"]


def test_world_trajectory_marks_missing_evidence_without_imputation():
    unavailable = _entry(2, gaps=[
        "observed_result_unavailable",
        "persistent_world_state_unavailable",
    ])
    unavailable["lifecycle_state"] = "executed_evidence_unavailable"
    unavailable["observed_result"] = None
    persistent_source = unavailable["long_term_accounting"][
        "persistent_team_state_delta"
    ]
    persistent_source.update({
        "available": False,
        "transition_identity": None,
        "recovery_complete": False,
        "match_delta": None,
    })
    thread = unavailable["world_evolution_thread"]
    thread["official_runtime_chain_complete"] = False
    thread["world_model_runtime_chain_complete"] = False
    result = next(
        row for row in thread["stages"]
        if row["stage_id"] == "observed_match_result"
    )
    result.clear()
    result.update({
        "stage_id": "observed_match_result",
        "status": "result_evidence_unavailable",
        "source_identity": None,
    })
    result["stage_identity"] = _identity(result)
    persistent = next(
        row for row in thread["stages"]
        if row["stage_id"] == "persistent_world_state"
    )
    persistent.clear()
    persistent.update({
        "stage_id": "persistent_world_state",
        "status": "persistent_state_evidence_unavailable",
        "source_identity": None,
        "recovery_complete": False,
        "metrics_delta": None,
        "transition_summary": None,
    })
    persistent["stage_identity"] = _identity(persistent)
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    unavailable.pop("entry_identity")
    unavailable["entry_identity"] = _identity(unavailable)

    trajectory = build_manager_world_navigator(_season(entries=[
        _entry(1), unavailable,
    ]))["world_trajectory"]
    missing = trajectory["points"][1]
    assert trajectory["results_available"] == 1
    assert trajectory["persistent_state_chapters"] == 1
    assert missing["result_available"] is False
    assert missing["points_earned"] is None
    assert missing["persistent_state_available"] is False
    assert missing["match_metrics_delta"] is None
    assert missing["match_transition_summary"] is None
    assert missing["cumulative"]["points_earned"] == 1
    assert missing["cumulative"]["persistent_state_chapters"] == 1
    assert missing["cumulative"]["match_metrics_delta_totals"][
        "team_fatigue_ema"
    ] == 0.08
    assert missing["world_change_markers"] == [
        "local_action_change",
        "result_evidence_unavailable",
        "persistent_state_evidence_unavailable",
        "continuity_gap",
    ]


@pytest.mark.parametrize(
    ("state", "action"),
    [
        ("decision_required", "freeze_manager_decision"),
        ("ready_to_explore", "request_bounded_futures"),
        ("future_generation_in_progress", "wait_for_bounded_futures"),
        ("future_generation_interrupted", "resume_bounded_futures"),
        ("future_generation_failed", "retry_bounded_futures"),
        ("evidence_ready_for_review", "review_future_evidence"),
        ("review_recorded_decision_refrozen", "advance_official_world"),
    ],
)
def test_navigator_exposes_exactly_one_primary_next_action(state, action):
    navigator = build_manager_world_navigator(_season(state))

    assert navigator["primary_action"]["action_id"] == action
    assert action not in {
        row["action_id"] for row in navigator["secondary_actions"]
    }
    assert len({
        row["action_id"] for row in navigator["secondary_actions"]
    }) == len(navigator["secondary_actions"])
    assert all(
        row["target_element_id"]
        for row in navigator["secondary_actions"]
    )


def test_current_continuity_gaps_are_preserved_and_invalid_values_fail_closed():
    season = _season()
    workspace = season["manager_intervention_workspace"]
    workspace["continuity_gaps"] = ["future_set_projection_drift"]
    workspace.pop("workspace_identity")
    workspace["workspace_identity"] = _identity(workspace)

    navigator = build_manager_world_navigator(season)

    assert navigator["current_chapter"]["continuity_gaps"] == [
        "future_set_projection_drift"
    ]

    workspace["continuity_gaps"] = ["future_set_projection_drift", ""]
    workspace.pop("workspace_identity")
    workspace["workspace_identity"] = _identity(workspace)
    with pytest.raises(ValueError, match="current continuity gaps"):
        build_manager_world_navigator(season)

    workspace["continuity_gaps"] = ["duplicate_gap", "duplicate_gap"]
    workspace.pop("workspace_identity")
    workspace["workspace_identity"] = _identity(workspace)
    with pytest.raises(ValueError, match="current continuity gaps"):
        build_manager_world_navigator(season)


def test_completed_season_closes_current_navigation_but_keeps_history():
    navigator = build_manager_world_navigator(
        _season("season_complete", entries=[_entry(1)]),
    )

    assert navigator["navigation_state"] == "season_complete"
    assert navigator["primary_action"] is None
    assert len(navigator["history_chapters"]) == 1


def test_history_is_recent_first_and_explicitly_bounded():
    entries = [
        _entry(index) for index in range(1, MAX_HISTORY_CHAPTERS + 2)
    ]

    navigator = build_manager_world_navigator(
        _season(entries=entries),
    )

    assert len(navigator["history_chapters"]) == MAX_HISTORY_CHAPTERS
    assert navigator["history_chapters"][0]["matchday"] == (
        MAX_HISTORY_CHAPTERS + 1
    )
    assert navigator["summary"]["chapters_truncated"] is True
    assert navigator["summary"]["completed_world_chapters"] == (
        MAX_HISTORY_CHAPTERS + 1
    )
    assert navigator["summary"]["official_runtime_chapters"] == (
        MAX_HISTORY_CHAPTERS + 1
    )
    assert navigator["summary"]["visible_official_runtime_chapters"] == (
        MAX_HISTORY_CHAPTERS
    )
    assert navigator["summary"]["local_action_changes"] == (
        2 * (MAX_HISTORY_CHAPTERS + 1)
    )
    assert navigator["summary"]["visible_local_action_changes"] == (
        2 * MAX_HISTORY_CHAPTERS
    )
    trajectory = navigator["world_trajectory"]
    assert trajectory["points_truncated"] is True
    assert trajectory["total_points"] == MAX_HISTORY_CHAPTERS + 1
    assert trajectory["visible_points"] == MAX_HISTORY_CHAPTERS
    assert trajectory["points"][0]["sequence"] == 2
    assert trajectory["points"][-1]["sequence"] == (
        MAX_HISTORY_CHAPTERS + 1
    )
    assert trajectory["points"][0]["cumulative"]["points_earned"] == 4


def test_history_order_and_window_do_not_depend_on_source_ledger_order():
    entries = [
        _entry(index) for index in range(1, MAX_HISTORY_CHAPTERS + 3)
    ]

    navigator = build_manager_world_navigator(
        _season(entries=list(reversed(entries))),
    )

    assert navigator["history_chapters"][0]["matchday"] == (
        MAX_HISTORY_CHAPTERS + 2
    )
    assert navigator["history_chapters"][-1]["matchday"] == 3
    assert [
        row["matchday"] for row in navigator["world_trajectory"]["points"]
    ] == list(range(3, MAX_HISTORY_CHAPTERS + 3))


def test_truncated_old_chapter_is_still_identity_validated():
    entries = [
        _entry(index) for index in range(1, MAX_HISTORY_CHAPTERS + 3)
    ]
    oldest_thread = entries[0]["world_evolution_thread"]
    oldest_thread["world_model_runtime_chain_complete"] = False

    with pytest.raises(ValueError, match="chapter identity"):
        build_manager_world_navigator(_season(entries=entries))


def test_gap_diagnostics_count_chapters_and_sort_deterministically():
    navigator = build_manager_world_navigator(_season(entries=[
        _entry(1, gaps=[
            "z_unknown_gap", "tactical_runtime_binding_unavailable",
        ]),
        _entry(2, gaps=["tactical_runtime_binding_unavailable"]),
        _entry(3, gaps=["observed_result_unavailable"]),
        _entry(4),
    ]))

    assert navigator["summary"]["chapters_with_continuity_gaps"] == 3
    assert navigator["summary"]["chapters_without_continuity_gaps"] == 1
    identities = {
        row["fixture_id"]: row["chapter_identity"]
        for row in navigator["history_chapters"]
    }
    assert navigator["summary"]["continuity_gap_counts"] == [
        {
            "gap": "tactical_runtime_binding_unavailable",
            "chapters": 2,
            "latest_fixture_id": "md02-fx01",
            "latest_matchday": 2,
            "latest_chapter_identity": identities["md02-fx01"],
        },
        {
            "gap": "observed_result_unavailable",
            "chapters": 1,
            "latest_fixture_id": "md03-fx01",
            "latest_matchday": 3,
            "latest_chapter_identity": identities["md03-fx01"],
        },
        {
            "gap": "z_unknown_gap",
            "chapters": 1,
            "latest_fixture_id": "md01-fx01",
            "latest_matchday": 1,
            "latest_chapter_identity": identities["md01-fx01"],
        },
    ]


def test_action_adoption_ledger_distinguishes_probability_and_realized_states():
    probability_only = _rewrite_action(
        _entry(2),
        "probability_influence_without_realized_action_change",
        locally_attributable_action_changes=0,
        changed_direct_ball_event_links=0,
        source_records_truncated=True,
        manager_record_coverage_complete=False,
    )
    no_influence = _rewrite_action(
        _entry(3),
        "no_nonzero_action_influence_observed",
        influenced_decisions=0,
        locally_attributable_action_changes=0,
        changed_direct_ball_event_links=0,
    )
    unavailable = _rewrite_action(
        _entry(4), "action_evidence_unavailable", source=False,
        retained_records=0,
        resolved_action_decisions=0,
        influenced_decisions=0,
        attribution_eligible_decisions=0,
        locally_attributable_action_changes=0,
        direct_ball_event_links=0,
        changed_direct_ball_event_links=0,
        decision_only_no_trajectory=0,
        unresolved_ball_event_links=0,
        source_match_opportunities=0,
        source_records_truncated=False,
        manager_record_coverage_complete=False,
    )
    navigator = build_manager_world_navigator(_season(entries=[
        _entry(1), probability_only, no_influence, unavailable,
    ]))

    ledger = navigator["summary"]["world_model_action_adoption_ledger"]
    assert ledger["fixtures_with_action_evidence"] == 3
    assert ledger["fixtures_with_complete_record_coverage"] == 2
    assert ledger["fixtures_with_incomplete_record_coverage"] == 1
    assert ledger["retained_records"] == 12
    assert ledger["influenced_decisions"] == 6
    assert ledger["locally_attributable_action_changes"] == 2
    assert ledger["influence_rate"] == 0.5
    assert ledger["realized_change_rate_among_influenced"] == 0.333333
    assert [row["state"] for row in ledger["state_counts"]] == [
        "realized_action_change",
        "probability_influence_only",
        "no_nonzero_influence",
        "evidence_unavailable",
    ]
    assert all(
        row["latest_chapter_identity"] in {
            chapter["chapter_identity"]
            for chapter in navigator["history_chapters"]
        }
        for row in ledger["state_counts"]
    )


def test_invalid_action_adoption_partition_and_status_fail_closed_when_rehashed():
    invalid_partition = _rewrite_action(
        _entry(1),
        "locally_attributable_action_changes_observed",
        direct_ball_event_links=3,
    )
    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(_season(entries=[invalid_partition]))

    contradictory_status = _rewrite_action(
        _entry(2),
        "probability_influence_without_realized_action_change",
    )
    with pytest.raises(ValueError, match="action adoption"):
        build_manager_world_navigator(_season(entries=[contradictory_status]))

    unknown_status = _rewrite_action(
        _entry(3), "silently_adopted", source=False,
        retained_records=0,
        resolved_action_decisions=0,
        influenced_decisions=0,
        attribution_eligible_decisions=0,
        locally_attributable_action_changes=0,
        direct_ball_event_links=0,
        changed_direct_ball_event_links=0,
        decision_only_no_trajectory=0,
        unresolved_ball_event_links=0,
        source_match_opportunities=0,
        source_records_truncated=False,
        manager_record_coverage_complete=False,
    )
    with pytest.raises(ValueError, match="action adoption"):
        build_manager_world_navigator(_season(entries=[unknown_status]))

    unavailable_with_counts = _rewrite_action(
        _entry(4), "action_evidence_unavailable", source=False,
    )
    with pytest.raises(ValueError, match="action adoption"):
        build_manager_world_navigator(_season(entries=[unavailable_with_counts]))


def test_invalid_descriptive_result_and_world_transition_fail_closed():
    invalid_result = _entry(1)
    invalid_result["observed_result"]["points_earned"] = 3
    invalid_result.pop("entry_identity")
    invalid_result["entry_identity"] = _identity(invalid_result)
    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(_season(entries=[invalid_result]))

    invalid_transition = _entry(2)
    persistent = next(
        row for row in invalid_transition["world_evolution_thread"]["stages"]
        if row["stage_id"] == "persistent_world_state"
    )
    persistent["metrics_delta"]["team_fatigue_ema"] = True
    persistent.pop("stage_identity")
    persistent["stage_identity"] = _identity(persistent)
    thread = invalid_transition["world_evolution_thread"]
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    invalid_transition.pop("entry_identity")
    invalid_transition["entry_identity"] = _identity(invalid_transition)
    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(_season(entries=[invalid_transition]))


def test_impossible_reviewed_future_partition_fails_closed_when_rehashed():
    entry = _entry(1)
    trace = entry["future_review_execution_trace"]
    terminal = trace["terminal_review"]
    terminal["action_divergence_scenarios"] = 1
    trace.pop("trace_identity")
    trace["trace_identity"] = _identity(trace)
    thread = entry["world_evolution_thread"]
    review_stage = next(
        row for row in thread["stages"]
        if row["stage_id"] == "prematch_future_review"
    )
    review_stage["source_identity"] = trace["trace_identity"]
    review_stage["action_divergence_scenarios"] = 1
    review_stage.pop("stage_identity")
    review_stage["stage_identity"] = _identity(review_stage)
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)

    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(_season(entries=[entry]))

    boolean_count = _entry(2)
    trace = boolean_count["future_review_execution_trace"]
    trace["terminal_review"]["eligible_scenarios"] = True
    trace.pop("trace_identity")
    trace["trace_identity"] = _identity(trace)
    thread = boolean_count["world_evolution_thread"]
    review_stage = next(
        row for row in thread["stages"]
        if row["stage_id"] == "prematch_future_review"
    )
    review_stage["source_identity"] = trace["trace_identity"]
    review_stage["eligible_scenarios"] = 1
    review_stage.pop("stage_identity")
    review_stage["stage_identity"] = _identity(review_stage)
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    boolean_count.pop("entry_identity")
    boolean_count["entry_identity"] = _identity(boolean_count)
    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(_season(entries=[boolean_count]))


def test_rehashed_reviewed_scenario_time_tamper_fails_closed():
    entry = _entry(1)
    trace = entry["future_review_execution_trace"]
    terminal = trace["terminal_review"]
    scenario = terminal["reviewed_scenarios"][1]
    scenario["branch_minute"] = 99.0
    scenario.pop("archive_identity")
    scenario["archive_identity"] = _identity(scenario)
    trace.pop("trace_identity")
    trace["trace_identity"] = _identity(trace)

    thread = entry["world_evolution_thread"]
    review_stage = next(
        row for row in thread["stages"]
        if row["stage_id"] == "prematch_future_review"
    )
    review_stage["source_identity"] = trace["trace_identity"]
    review_stage["reviewed_scenarios"] = copy.deepcopy(
        terminal["reviewed_scenarios"]
    )
    review_stage.pop("stage_identity")
    review_stage["stage_identity"] = _identity(review_stage)
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)

    with pytest.raises(ValueError, match="scenario archive values"):
        build_manager_world_navigator(_season(entries=[entry]))


def test_reviewed_future_descriptive_difference_is_not_a_false_funnel():
    entry = _entry(1)
    trace = entry["future_review_execution_trace"]
    terminal = trace["terminal_review"]
    terminal.update({
        "eligible_scenarios": 1,
        "verified_anchor_scenarios": 1,
        "action_divergence_scenarios": 2,
        "local_attribution_scenarios": 0,
        "descriptive_future_difference_scenarios": 2,
    })
    replacements = [
        (False, False, None, "descriptive_only_ineligible", 1, 0, 1),
        (True, True, "8" * 64,
         "action_divergence_without_local_attribution", 1, 0, 1),
        (False, False, None, "descriptive_only_ineligible", 0, 0, 0),
    ]
    for scenario, values in zip(
        terminal["reviewed_scenarios"], replacements, strict=True,
    ):
        (
            eligible, anchor, state_identity, status,
            changed, local, differences,
        ) = values
        scenario.update({
            "eligible": eligible,
            "anchor_verified": anchor,
            "branch_state_identity": state_identity,
            "future_status": status,
            "changed_actions": changed,
            "locally_attributable_changes": local,
            "descriptive_future_difference_count": differences,
            "simulator_local_action_attribution": False,
        })
        scenario.pop("archive_identity")
        scenario["archive_identity"] = _identity(scenario)
    trace.pop("trace_identity")
    trace["trace_identity"] = _identity(trace)
    thread = entry["world_evolution_thread"]
    review_stage = next(
        row for row in thread["stages"]
        if row["stage_id"] == "prematch_future_review"
    )
    for field in (
        "eligible_scenarios",
        "verified_anchor_scenarios",
        "action_divergence_scenarios",
        "local_attribution_scenarios",
        "descriptive_future_difference_scenarios",
    ):
        review_stage[field] = terminal[field]
    review_stage["reviewed_scenarios"] = copy.deepcopy(
        terminal["reviewed_scenarios"]
    )
    review_stage["source_identity"] = trace["trace_identity"]
    review_stage.pop("stage_identity")
    review_stage["stage_identity"] = _identity(review_stage)
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)

    chapter = build_manager_world_navigator(
        _season(entries=[entry]),
    )["history_chapters"][0]
    assert chapter["reviewed_future_context"][
        "descriptive_future_difference_scenarios"
    ] == 2
    assert chapter["reviewed_future_context"][
        "local_attribution_scenarios"
    ] == 0


def test_duplicate_historical_gap_fails_closed():
    season = _season(entries=[_entry(
        1, gaps=[
            "observed_result_unavailable",
            "observed_result_unavailable",
        ],
    )])

    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(season)


def test_invalid_historical_matchday_fails_closed_even_when_rehashed():
    season = _season(entries=[_entry(1)])
    entry = season["manager_decision_ledger"]["entries"][0]
    entry["matchday"] = True
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)

    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(season)


def test_duplicate_fixture_identity_fails_closed_even_when_rehashed():
    first = _entry(1)
    duplicate = copy.deepcopy(first)
    duplicate["matchday"] = 2
    duplicate.pop("entry_identity")
    duplicate["entry_identity"] = _identity(duplicate)

    with pytest.raises(ValueError, match="invalid or duplicate"):
        build_manager_world_navigator(_season(entries=[first, duplicate]))


def test_duplicate_completed_matchday_fails_closed_even_when_rehashed():
    first = _entry(1)
    duplicate_matchday = _entry(2)
    duplicate_matchday["matchday"] = 1
    duplicate_matchday.pop("entry_identity")
    duplicate_matchday["entry_identity"] = _identity(duplicate_matchday)

    with pytest.raises(ValueError, match="completed matchdays are duplicate"):
        build_manager_world_navigator(
            _season(entries=[first, duplicate_matchday]),
        )


def test_rehashed_navigation_or_source_thread_tamper_fails_closed():
    season = _season(entries=[_entry(1)])
    navigator = build_manager_world_navigator(season)
    tampered = copy.deepcopy(navigator)
    tampered["primary_action"]["label"] = "自动选择最佳战术"
    frozen = copy.deepcopy(tampered)
    frozen.pop("navigator_identity")
    tampered["navigator_identity"] = _identity(frozen)

    with pytest.raises(ValueError, match="navigator replay mismatch"):
        validate_manager_world_navigator(tampered, season=season)

    corrupt_source = copy.deepcopy(season)
    thread = corrupt_source["manager_decision_ledger"]["entries"][0][
        "world_evolution_thread"
    ]
    thread["world_model_runtime_chain_complete"] = False
    with pytest.raises(ValueError, match="chapter identity"):
        build_manager_world_navigator(corrupt_source)


def test_unknown_workflow_lifecycle_and_invalid_action_count_fail_closed():
    bad_workflow = _season()
    workspace = bad_workflow["manager_intervention_workspace"]
    workspace["workflow_state"] = "auto_pick_best_world"
    workspace.pop("workspace_identity")
    workspace["workspace_identity"] = _identity(workspace)
    with pytest.raises(ValueError, match="workflow state"):
        build_manager_world_navigator(bad_workflow)

    bad_lifecycle = _season(entries=[_entry(1)])
    entry = bad_lifecycle["manager_decision_ledger"]["entries"][0]
    entry["lifecycle_state"] = "silently_promoted"
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)
    with pytest.raises(ValueError, match="ledger entries"):
        build_manager_world_navigator(bad_lifecycle)

    bad_count = _season(entries=[_entry(1)])
    entry = bad_count["manager_decision_ledger"]["entries"][0]
    thread = entry["world_evolution_thread"]
    action = thread["stages"][3]
    action["locally_attributable_action_changes"] = True
    action.pop("stage_identity")
    action["stage_identity"] = _identity(action)
    thread.pop("thread_identity")
    thread["thread_identity"] = _identity(thread)
    entry.pop("entry_identity")
    entry["entry_identity"] = _identity(entry)
    with pytest.raises(ValueError, match="chapter facts"):
        build_manager_world_navigator(bad_count)
