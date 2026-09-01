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


def _thread(fixture_id="md01-fx01", *, complete=True, gaps=None):
    stages = [
        _stage("prematch_future_review", "selected_for_fixture"),
        _stage("frozen_manager_decision", "frozen"),
        _stage("official_tactical_runtime", "runtime_verified"),
        _stage(
            "official_world_model_actions",
            "locally_attributable_action_changes_observed",
            locally_attributable_action_changes=2,
        ),
        _stage("observed_match_result", "observed_descriptive"),
        _stage(
            "persistent_world_state",
            "after_recovery_recorded",
            source_identity="f" * 64,
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
                "descriptive_only": True,
            }
        ),
        "world_evolution_thread": _thread(
            fixture_id, complete=not pending, gaps=gaps,
        ),
    }
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


def test_navigator_joins_current_review_and_completed_world_chapters():
    season = _season(entries=[
        _entry(1),
        _entry(2, gaps=["world_model_action_evidence_unavailable"]),
        _entry(3, pending=True),
    ])

    navigator = build_manager_world_navigator(season)

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
    }
    assert navigator["causal_effect_authorized"] is False
    assert navigator["outcome_effect_estimate"] is None
    validate_manager_world_navigator(navigator, season=season)


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


def test_duplicate_fixture_identity_fails_closed_even_when_rehashed():
    first = _entry(1)
    duplicate = copy.deepcopy(first)
    duplicate["matchday"] = 2
    duplicate.pop("entry_identity")
    duplicate["entry_identity"] = _identity(duplicate)

    with pytest.raises(ValueError, match="invalid or duplicate"):
        build_manager_world_navigator(_season(entries=[first, duplicate]))


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
