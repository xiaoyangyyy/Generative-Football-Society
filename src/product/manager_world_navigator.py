"""Season-level navigation across current intervention and completed worlds."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_HISTORY_CHAPTERS = 64
_WORKFLOW_STATES = {
    "decision_required",
    "ready_to_explore",
    "future_generation_in_progress",
    "future_generation_interrupted",
    "future_generation_failed",
    "evidence_ready_for_review",
    "review_recorded_decision_refrozen",
    "season_complete",
}
_LIFECYCLE_STATES = {
    "frozen_awaiting_execution",
    "executed_with_direct_evidence",
    "executed_evidence_unavailable",
}
_ACTION_EVIDENCE_STATES = {
    "locally_attributable_action_changes_observed",
    "probability_influence_without_realized_action_change",
    "no_nonzero_action_influence_observed",
}
_COMPLETED_ACTION_STATES = _ACTION_EVIDENCE_STATES | {
    "not_applicable_stable_mode",
    "legacy_evidence_unavailable",
    "action_evidence_unavailable",
}
_ACTION_ADOPTION_STATE_ORDER = (
    "realized_action_change",
    "probability_influence_only",
    "no_nonzero_influence",
    "not_applicable_stable_mode",
    "evidence_unavailable",
)
_BOUNDARY = (
    "navigation over the current manager intervention session and replayable "
    "completed simulator-world chapters only; navigation state is not an "
    "outcome-effect estimate, intervention ranking or causal conclusion"
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _is_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _identity_matches(payload: Mapping[str, Any], field: str) -> bool:
    return (
        _is_identity(payload.get(field))
        and payload[field] == _identity({
            key: value for key, value in payload.items() if key != field
        })
    )


def _primary_action(workflow_state: str) -> dict[str, Any] | None:
    actions = {
        "decision_required": (
            "freeze_manager_decision", "冻结本场经理决策",
            "manager-decision-form",
        ),
        "ready_to_explore": (
            "request_bounded_futures", "生成身份绑定的有界未来",
            "request-manager-future-set",
        ),
        "future_generation_in_progress": (
            "wait_for_bounded_futures", "等待固定场景预算完成", None,
        ),
        "future_generation_interrupted": (
            "resume_bounded_futures", "从已验证进度恢复未来生成",
            "manager-future-set-list",
        ),
        "future_generation_failed": (
            "retry_bounded_futures", "检查失败后重新生成有界未来",
            "request-manager-future-set",
        ),
        "evidence_ready_for_review": (
            "review_future_evidence", "复核未来证据并保留或修改方案",
            "manager-future-set-list",
        ),
        "review_recorded_decision_refrozen": (
            "advance_official_world", "推进正式比赛世界",
            "play-matchday",
        ),
    }
    selected = actions.get(workflow_state)
    if selected is None:
        return None
    action_id, label, target = selected
    return {
        "action_id": action_id,
        "label": label,
        "target_element_id": target,
    }


def _secondary_actions(
    workflow_state: str,
    allowed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    primary = _primary_action(workflow_state)
    primary_id = primary.get("action_id") if primary else None
    candidates = []
    if allowed.get("request_future_set") is True:
        candidates.append({
            "action_id": "request_bounded_futures",
            "label": "再生成一组有界未来",
            "target_element_id": "request-manager-future-set",
        })
    if allowed.get("review_future_set") is True:
        candidates.append({
            "action_id": "review_future_evidence",
            "label": "复核当前未来证据",
            "target_element_id": "manager-future-set-list",
        })
    if allowed.get("advance_official_match") is True:
        candidates.append({
            "action_id": "advance_official_world",
            "label": "跳过可选探索并推进正式比赛",
            "target_element_id": "play-matchday",
        })
    return [
        row for row in candidates if row["action_id"] != primary_id
    ]


def _history_chapter(entry: Mapping[str, Any]) -> dict[str, Any]:
    thread = entry.get("world_evolution_thread")
    if not isinstance(thread, Mapping):
        raise ValueError("manager world navigator chapter thread is unavailable")
    if (
        not _identity_matches(entry, "entry_identity")
        or not _identity_matches(thread, "thread_identity")
        or entry.get("fixture_id") != thread.get("fixture_id")
        or thread.get("causal_effect_authorized") is not False
        or thread.get("outcome_effect_estimate") is not None
    ):
        raise ValueError("manager world navigator chapter identity is invalid")
    stages = thread.get("stages")
    if not isinstance(stages, list):
        raise ValueError("manager world navigator chapter stages are invalid")
    stage_by_id = {
        row.get("stage_id"): row
        for row in stages if isinstance(row, Mapping)
    }
    required = {
        "prematch_future_review",
        "frozen_manager_decision",
        "official_tactical_runtime",
        "official_world_model_actions",
        "observed_match_result",
        "persistent_world_state",
    }
    if set(stage_by_id) != required or any(
        not _identity_matches(row, "stage_identity")
        for row in stage_by_id.values()
    ):
        raise ValueError("manager world navigator chapter stages are incomplete")
    observed = entry.get("observed_result")
    observed = observed if isinstance(observed, Mapping) else {}
    action = stage_by_id["official_world_model_actions"]
    persistent = stage_by_id["persistent_world_state"]
    action_count_fields = (
        "retained_records",
        "resolved_action_decisions",
        "influenced_decisions",
        "attribution_eligible_decisions",
        "locally_attributable_action_changes",
        "direct_ball_event_links",
        "changed_direct_ball_event_links",
        "decision_only_no_trajectory",
        "unresolved_ball_event_links",
        "source_match_opportunities",
    )
    action_counts = {
        field: action.get(field, 0) for field in action_count_fields
    }
    local_changes = action_counts["locally_attributable_action_changes"]
    action_status = action.get("status")
    action_source_identity = action.get("source_identity")
    transition_identity = persistent.get("source_identity")
    gaps = thread.get("continuity_gaps")
    matchday = entry.get("matchday")
    if (
        isinstance(matchday, bool)
        or not isinstance(matchday, int)
        or matchday < 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in action_counts.values()
        )
        or not isinstance(action.get("manager_record_coverage_complete"), bool)
        or not isinstance(action.get("source_records_truncated"), bool)
        or (
            action_status in _ACTION_EVIDENCE_STATES
            and action.get("manager_record_coverage_complete") is not (
                not action.get("source_records_truncated")
            )
        )
        or action_counts["influenced_decisions"]
        > action_counts["retained_records"]
        or action_counts["locally_attributable_action_changes"]
        > action_counts["attribution_eligible_decisions"]
        or action_counts["locally_attributable_action_changes"]
        > action_counts["influenced_decisions"]
        or action_counts["attribution_eligible_decisions"]
        > action_counts["retained_records"]
        or action_counts["resolved_action_decisions"]
        > action_counts["retained_records"]
        or action_counts["direct_ball_event_links"]
        > action_counts["retained_records"]
        or action_counts["changed_direct_ball_event_links"] > min(
            action_counts["direct_ball_event_links"], local_changes,
        )
        or (
            action_counts["direct_ball_event_links"]
            + action_counts["decision_only_no_trajectory"]
            + action_counts["unresolved_ball_event_links"]
            != action_counts["retained_records"]
        )
        or action_counts["retained_records"]
        > action_counts["source_match_opportunities"]
        or (
            action_source_identity is not None
            and not _is_identity(action_source_identity)
        )
        or (
            transition_identity is not None
            and not _is_identity(transition_identity)
        )
        or not isinstance(gaps, list)
        or any(not isinstance(gap, str) or not gap for gap in gaps)
        or len(gaps) != len(set(gaps))
    ):
        raise ValueError("manager world navigator chapter facts are invalid")
    if action_status == "locally_attributable_action_changes_observed":
        adoption_state = "realized_action_change"
    elif action_status == "probability_influence_without_realized_action_change":
        adoption_state = "probability_influence_only"
    elif action_status == "no_nonzero_action_influence_observed":
        adoption_state = "no_nonzero_influence"
    elif action_status == "not_applicable_stable_mode":
        adoption_state = "not_applicable_stable_mode"
    else:
        adoption_state = "evidence_unavailable"
    if (
        action_status not in _COMPLETED_ACTION_STATES
        or (action_status in _ACTION_EVIDENCE_STATES)
        is not (action_source_identity is not None)
        or (
            action_status not in _ACTION_EVIDENCE_STATES
            and (
                any(action_counts.values())
                or action.get("manager_record_coverage_complete") is not False
                or action.get("source_records_truncated") is not False
            )
        )
        or (
            action_status == "locally_attributable_action_changes_observed"
            and local_changes < 1
        )
        or (
            action_status == "probability_influence_without_realized_action_change"
            and (
                action_counts["influenced_decisions"] < 1
                or local_changes != 0
            )
        )
        or (
            action_status == "no_nonzero_action_influence_observed"
            and action_counts["influenced_decisions"] != 0
        )
    ):
        raise ValueError("manager world navigator action adoption is invalid")
    payload = {
        "fixture_id": entry.get("fixture_id"),
        "matchday": matchday,
        "lifecycle_state": entry.get("lifecycle_state"),
        "source_entry_identity": entry.get("entry_identity"),
        "source_thread_identity": thread.get("thread_identity"),
        "thread_state": thread.get("thread_state"),
        "official_runtime_chain_complete": (
            thread.get("official_runtime_chain_complete") is True
        ),
        "world_model_runtime_chain_complete": (
            thread.get("world_model_runtime_chain_complete") is True
        ),
        "stage_statuses": {
            stage_id: stage_by_id[stage_id].get("status")
            for stage_id in (
                "prematch_future_review",
                "frozen_manager_decision",
                "official_tactical_runtime",
                "official_world_model_actions",
                "observed_match_result",
                "persistent_world_state",
            )
        },
        "score": copy.deepcopy(observed.get("score")),
        "outcome": observed.get("outcome"),
        "locally_attributable_action_changes": local_changes,
        "action_adoption": {
            "state": adoption_state,
            **copy.deepcopy(action_counts),
            "manager_record_coverage_complete": action.get(
                "manager_record_coverage_complete"
            ),
            "source_records_truncated": action.get(
                "source_records_truncated"
            ),
        },
        "official_action_evidence_identity": action_source_identity,
        "persistent_transition_identity": transition_identity,
        "continuity_gaps": copy.deepcopy(gaps),
        "causal_effect_authorized": False,
    }
    payload["chapter_identity"] = _identity(payload)
    return payload


def build_manager_world_navigator(
    season: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the current manager session and completed ledger chapters."""
    season_id = str(season.get("season_id") or "")
    revision = season.get("revision")
    workspace = season.get("manager_intervention_workspace")
    ledger = season.get("manager_decision_ledger")
    if (
        not season_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(workspace, Mapping)
        or not _identity_matches(workspace, "workspace_identity")
        or workspace.get("season_id") != season_id
        or workspace.get("season_revision") != revision
        or not isinstance(ledger, Mapping)
        or not isinstance(ledger.get("entries"), list)
    ):
        raise ValueError("manager world navigator sources are invalid")

    if any(
        not isinstance(row, Mapping)
        or row.get("lifecycle_state") not in _LIFECYCLE_STATES
        for row in ledger["entries"]
    ):
        raise ValueError("manager world navigator ledger entries are invalid")
    fixture_ids = [row.get("fixture_id") for row in ledger["entries"]]
    if (
        any(not isinstance(fixture_id, str) or not fixture_id
            for fixture_id in fixture_ids)
        or len(fixture_ids) != len(set(fixture_ids))
    ):
        raise ValueError(
            "manager world navigator fixture identities are invalid or duplicate"
        )
    completed_entries = [
        row for row in ledger["entries"]
        if row.get("lifecycle_state") != "frozen_awaiting_execution"
    ]
    all_chapters = [_history_chapter(row) for row in completed_entries]
    chapters = list(reversed(all_chapters[-MAX_HISTORY_CHAPTERS:]))
    workflow_state = str(workspace.get("workflow_state") or "")
    if workflow_state not in _WORKFLOW_STATES:
        raise ValueError("manager world navigator workflow state is invalid")
    allowed = workspace.get("allowed_actions")
    allowed = allowed if isinstance(allowed, Mapping) else {}
    current_gaps = workspace.get("continuity_gaps")
    if (
        not isinstance(current_gaps, list)
        or any(not isinstance(gap, str) or not gap for gap in current_gaps)
        or len(current_gaps) != len(set(current_gaps))
    ):
        raise ValueError(
            "manager world navigator current continuity gaps are invalid"
        )
    current = {
        "workflow_state": workflow_state,
        "workspace_identity": workspace.get("workspace_identity"),
        "fixture": copy.deepcopy(workspace.get("fixture")),
        "frozen_decision_identity": (
            (workspace.get("frozen_intervention") or {}).get(
                "decision_identity"
            )
            if isinstance(
                workspace.get("frozen_intervention"), Mapping,
            ) else None
        ),
        "stages": copy.deepcopy(workspace.get("stages") or []),
        "evidence_summary": copy.deepcopy(
            workspace.get("evidence_summary")
        ),
        "continuity_gaps": copy.deepcopy(current_gaps),
    }
    current["current_chapter_identity"] = _identity(current)
    primary = _primary_action(workflow_state)
    influence_path = {
        "review_selected": sum(
            row["stage_statuses"]["prematch_future_review"]
            == "selected_for_fixture"
            for row in all_chapters
        ),
        "runtime_tactic_verified": sum(
            row["stage_statuses"]["official_tactical_runtime"]
            == "runtime_verified"
            for row in all_chapters
        ),
        "official_action_evidence": sum(
            row["official_action_evidence_identity"] is not None
            for row in all_chapters
        ),
        "chapters_with_local_action_changes": sum(
            row["locally_attributable_action_changes"] > 0
            for row in all_chapters
        ),
        "persistent_world_transitions": sum(
            row["persistent_transition_identity"] is not None
            for row in all_chapters
        ),
        "complete_world_model_runtime_chains": sum(
            row["world_model_runtime_chain_complete"]
            for row in all_chapters
        ),
    }
    adoption_totals = {
        field: sum(
            row["action_adoption"][field] for row in all_chapters
        )
        for field in (
            "retained_records",
            "resolved_action_decisions",
            "influenced_decisions",
            "attribution_eligible_decisions",
            "locally_attributable_action_changes",
            "direct_ball_event_links",
            "changed_direct_ball_event_links",
            "decision_only_no_trajectory",
            "unresolved_ball_event_links",
            "source_match_opportunities",
        )
    }
    adoption_state_facts: dict[str, dict[str, Any]] = {}
    for chapter in all_chapters:
        state = chapter["action_adoption"]["state"]
        facts = adoption_state_facts.setdefault(state, {
            "chapters": 0,
            "latest_fixture_id": None,
            "latest_matchday": 0,
            "latest_chapter_identity": None,
        })
        facts["chapters"] += 1
        current_key = (
            facts["latest_matchday"], str(facts["latest_fixture_id"] or ""),
        )
        chapter_key = (chapter["matchday"], chapter["fixture_id"])
        if chapter_key > current_key:
            facts["latest_fixture_id"] = chapter["fixture_id"]
            facts["latest_matchday"] = chapter["matchday"]
            facts["latest_chapter_identity"] = chapter["chapter_identity"]
    action_adoption_ledger = {
        "fixtures_with_action_evidence": sum(
            row["official_action_evidence_identity"] is not None
            for row in all_chapters
        ),
        "fixtures_with_complete_record_coverage": sum(
            row["official_action_evidence_identity"] is not None
            and row["action_adoption"]["manager_record_coverage_complete"]
            for row in all_chapters
        ),
        "fixtures_with_incomplete_record_coverage": sum(
            row["official_action_evidence_identity"] is not None
            and not row["action_adoption"]["manager_record_coverage_complete"]
            for row in all_chapters
        ),
        **adoption_totals,
        "influence_rate": (
            round(
                adoption_totals["influenced_decisions"]
                / adoption_totals["retained_records"], 6,
            )
            if adoption_totals["retained_records"] else None
        ),
        "realized_change_rate_among_influenced": (
            round(
                adoption_totals["locally_attributable_action_changes"]
                / adoption_totals["influenced_decisions"], 6,
            )
            if adoption_totals["influenced_decisions"] else None
        ),
        "direct_ball_event_link_coverage": (
            round(
                adoption_totals["direct_ball_event_links"] / (
                    adoption_totals["direct_ball_event_links"]
                    + adoption_totals["unresolved_ball_event_links"]
                ), 6,
            )
            if (
                adoption_totals["direct_ball_event_links"]
                + adoption_totals["unresolved_ball_event_links"]
            ) else None
        ),
        "state_counts": [
            {"state": state, **adoption_state_facts[state]}
            for state in _ACTION_ADOPTION_STATE_ORDER
            if state in adoption_state_facts
        ],
    }
    gap_facts: dict[str, dict[str, Any]] = {}
    for chapter in all_chapters:
        for gap in chapter["continuity_gaps"]:
            facts = gap_facts.setdefault(gap, {
                "chapters": 0,
                "latest_fixture_id": None,
                "latest_matchday": 0,
                "latest_chapter_identity": None,
            })
            facts["chapters"] += 1
            current_key = (
                facts["latest_matchday"],
                str(facts["latest_fixture_id"] or ""),
            )
            chapter_key = (chapter["matchday"], chapter["fixture_id"])
            if chapter_key > current_key:
                facts["latest_fixture_id"] = chapter["fixture_id"]
                facts["latest_matchday"] = chapter["matchday"]
                facts["latest_chapter_identity"] = chapter["chapter_identity"]
    continuity_gap_counts = [
        {"gap": gap, **facts}
        for gap, facts in sorted(
            gap_facts.items(),
            key=lambda item: (-item[1]["chapters"], item[0]),
        )
    ]
    summary = {
        "completed_world_chapters": len(all_chapters),
        "visible_world_chapters": len(chapters),
        "chapters_truncated": (
            len(all_chapters) > MAX_HISTORY_CHAPTERS
        ),
        "official_runtime_chapters": sum(
            row["official_runtime_chain_complete"] for row in all_chapters
        ),
        "world_model_runtime_chapters": sum(
            row["world_model_runtime_chain_complete"] for row in all_chapters
        ),
        "local_action_changes": sum(
            row["locally_attributable_action_changes"] for row in all_chapters
        ),
        "chapters_with_continuity_gaps": sum(
            bool(row["continuity_gaps"]) for row in all_chapters
        ),
        "chapters_without_continuity_gaps": sum(
            not row["continuity_gaps"] for row in all_chapters
        ),
        "continuity_gap_counts": continuity_gap_counts,
        "visible_official_runtime_chapters": sum(
            row["official_runtime_chain_complete"] for row in chapters
        ),
        "visible_world_model_runtime_chapters": sum(
            row["world_model_runtime_chain_complete"] for row in chapters
        ),
        "visible_local_action_changes": sum(
            row["locally_attributable_action_changes"] for row in chapters
        ),
        "visible_chapters_with_continuity_gaps": sum(
            bool(row["continuity_gaps"]) for row in chapters
        ),
        "world_model_influence_path": influence_path,
        "world_model_action_adoption_ledger": action_adoption_ledger,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "season_id": season_id,
        "season_revision": revision,
        "navigation_state": (
            "season_complete"
            if workflow_state == "season_complete" else "active_manager_world"
        ),
        "current_chapter": current,
        "primary_action": primary,
        "secondary_actions": _secondary_actions(
            workflow_state, allowed,
        ),
        "history_chapters": chapters,
        "summary": summary,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": _BOUNDARY,
    }
    payload["navigator_identity"] = _identity(payload)
    return payload


def validate_manager_world_navigator(
    navigator: Mapping[str, Any],
    *,
    season: Mapping[str, Any],
) -> None:
    """Replay the navigator from the current session and decision ledger."""
    expected = build_manager_world_navigator(season)
    if dict(navigator) != expected:
        raise ValueError("manager world navigator replay mismatch")


__all__ = [
    "MAX_HISTORY_CHAPTERS",
    "SCHEMA_VERSION",
    "build_manager_world_navigator",
    "validate_manager_world_navigator",
]
