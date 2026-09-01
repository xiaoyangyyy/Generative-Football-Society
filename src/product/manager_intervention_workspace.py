"""Canonical manager intervention session derived from existing authorities."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.product.season import ManagerDecision


SCHEMA_VERSION = 1
MAX_VISIBLE_EXPLORATIONS = 50
_TASK_STATES = {"queued", "running", "interrupted", "completed", "failed"}
_BOUNDARY = (
    "one product workflow derived from the frozen manager decision, bounded "
    "simulator futures and explicit review receipts; it does not create a "
    "second season state, rank interventions, predict scores or establish "
    "match-outcome or real-football causality"
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


def _stage(stage_id: str, status: str, **facts: Any) -> dict[str, Any]:
    payload = {
        "stage_id": stage_id,
        "status": status,
        **copy.deepcopy(facts),
    }
    payload["stage_identity"] = _identity(payload)
    return payload


def _decision_identity(decision: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    normalized = ManagerDecision.from_payload(decision).as_dict()
    return normalized, _identity(normalized)


def _normalized_exploration(
    item: Mapping[str, Any],
    *,
    season_id: str,
    revision: int,
    fixture_id: str,
    decision_identity: str,
    review_by_task: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    gaps: list[str] = []
    task_id = item.get("task_id")
    context = item.get("manager_context")
    state = item.get("state")
    branch_times = item.get("branch_times_sec")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not task_id.isalnum()
        or len(task_id) > 64
        or not isinstance(context, Mapping)
        or state not in _TASK_STATES
        or not isinstance(branch_times, list)
        or not 2 <= len(branch_times) <= 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 5400
            or (
                index
                and float(value) <= float(branch_times[index - 1])
            )
            for index, value in enumerate(branch_times)
        )
    ):
        return None, ["invalid_future_set_projection"]
    context_identity = context.get("context_identity")
    source_decision_identity = context.get("decision_identity")
    if not _is_identity(context_identity) or not _is_identity(
        source_decision_identity
    ):
        return None, ["invalid_future_set_context_identity"]
    same_fixture = (
        context.get("season_id") == season_id
        and context.get("fixture_id") == fixture_id
    )
    current_binding = bool(
        same_fixture
        and context.get("season_revision") == revision
        and source_decision_identity == decision_identity
    )
    if bool(item.get("binding_current")) is not current_binding:
        gaps.append("future_set_binding_projection_mismatch")
    scenarios = item.get("scenario_evidence")
    scenarios = scenarios if isinstance(scenarios, list) else []
    scenario_identities = [
        row.get("scenario_identity")
        for row in scenarios
        if isinstance(row, Mapping) and _is_identity(
            row.get("scenario_identity")
        )
    ]
    if len(scenario_identities) != len(scenarios):
        gaps.append("future_set_scenario_identity_unavailable")
    if state == "completed" and not scenarios:
        gaps.append("completed_future_set_without_scenario_evidence")
    receipt = item.get("review_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else review_by_task.get(
        task_id
    )
    review_identity = (
        receipt.get("review_identity")
        if isinstance(receipt, Mapping) else None
    )
    if review_identity is not None and not _is_identity(review_identity):
        gaps.append("future_set_review_identity_invalid")
        review_identity = None
    counts = {}
    for name in (
        "action_divergence_scenarios",
        "local_attribution_scenarios",
        "descriptive_future_difference_scenarios",
    ):
        value = item.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            gaps.append("future_set_aggregate_count_invalid")
            counts[name] = 0
        else:
            counts[name] = value
    budget = len(branch_times)
    if any(value > budget for value in counts.values()):
        gaps.append("future_set_aggregate_count_invalid")
        counts = {name: min(value, budget) for name, value in counts.items()}
    payload = {
        "task_id": task_id,
        "state": state,
        "fixture_id": context.get("fixture_id"),
        "context_identity": context_identity,
        "source_revision": context.get("season_revision"),
        "source_decision_identity": source_decision_identity,
        "current_binding": current_binding,
        "branch_minutes": [
            float(value) / 60.0 for value in branch_times
        ],
        "scenario_identities": scenario_identities,
        "scenario_evidence_available": bool(scenarios),
        **counts,
        "reviewed": review_identity is not None,
        "review_identity": review_identity,
    }
    payload["exploration_identity"] = _identity(payload)
    return payload, gaps


def build_manager_intervention_workspace(
    season: Mapping[str, Any],
    manager_future_sets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one read-only state machine for the manager intervention journey."""
    season_id = str(season.get("season_id") or "")
    revision = season.get("revision")
    managed = season.get("next_manager_fixture")
    if (
        not season_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
    ):
        raise ValueError("manager intervention season identity is invalid")
    if not isinstance(manager_future_sets, Sequence) or isinstance(
        manager_future_sets, (str, bytes)
    ):
        raise ValueError("manager intervention future sets are invalid")
    if len(manager_future_sets) > MAX_VISIBLE_EXPLORATIONS:
        raise ValueError("manager intervention future sets exceed visible limit")

    if not isinstance(managed, Mapping):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "workflow_state": "season_complete",
            "season_id": season_id,
            "season_revision": revision,
            "fixture": None,
            "frozen_intervention": None,
            "stages": [],
            "explorations": [],
            "evidence_summary": None,
            "allowed_actions": {
                "freeze_decision": False,
                "request_future_set": False,
                "review_future_set": False,
                "advance_official_match": False,
            },
            "continuity_gaps": [],
            "outcome_effect_estimate": None,
            "causal_effect_authorized": False,
            "claim_boundary": _BOUNDARY,
        }
        payload["workspace_identity"] = _identity(payload)
        return payload

    fixture_id = str(managed.get("fixture_id") or "")
    fixture = {
        "fixture_id": fixture_id,
        "matchday": managed.get("matchday"),
        "home": managed.get("home"),
        "away": managed.get("away"),
    }
    if (
        not fixture_id
        or not isinstance(fixture["matchday"], int)
        or not all(
            isinstance(fixture[name], str) and fixture[name]
            for name in ("home", "away")
        )
    ):
        raise ValueError("manager intervention fixture is invalid")
    raw_decision = managed.get("manager_decision")
    if not isinstance(raw_decision, Mapping):
        stages = [
            _stage("freeze_intervention", "decision_required"),
            _stage("generate_bounded_futures", "blocked_by_decision"),
            _stage("inspect_local_mechanism", "blocked_by_decision"),
            _stage("record_manager_review", "blocked_by_decision"),
            _stage("advance_official_world", "blocked_by_decision"),
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "available": True,
            "workflow_state": "decision_required",
            "season_id": season_id,
            "season_revision": revision,
            "fixture": fixture,
            "frozen_intervention": None,
            "stages": stages,
            "explorations": [],
            "evidence_summary": {
                "future_sets": 0,
                "current_future_sets": 0,
                "completed_future_sets": 0,
                "reviewed_future_sets": 0,
                "registered_scenarios": 0,
                "action_divergence_scenarios": 0,
                "local_attribution_scenarios": 0,
                "descriptive_future_difference_scenarios": 0,
            },
            "allowed_actions": {
                "freeze_decision": True,
                "request_future_set": False,
                "review_future_set": False,
                "advance_official_match": False,
            },
            "continuity_gaps": [],
            "outcome_effect_estimate": None,
            "causal_effect_authorized": False,
            "claim_boundary": _BOUNDARY,
        }
        payload["workspace_identity"] = _identity(payload)
        return payload

    decision, decision_identity = _decision_identity(raw_decision)
    reviews = managed.get("manager_future_reviews")
    reviews = reviews if isinstance(reviews, list) else []
    review_by_task = {
        str(row.get("task_id")): row
        for row in reviews
        if isinstance(row, Mapping)
        and isinstance(row.get("task_id"), str)
        and _is_identity(row.get("review_identity"))
    }
    normalized_sets = []
    gaps = []
    for item in manager_future_sets:
        if not isinstance(item, Mapping):
            gaps.append("invalid_future_set_projection")
            continue
        normalized, item_gaps = _normalized_exploration(
            item,
            season_id=season_id,
            revision=revision,
            fixture_id=fixture_id,
            decision_identity=decision_identity,
            review_by_task=review_by_task,
        )
        gaps.extend(item_gaps)
        if normalized is not None and normalized["fixture_id"] == fixture_id:
            normalized_sets.append(normalized)

    current_sets = [
        row for row in normalized_sets if row["current_binding"]
    ]
    completed_current = [
        row for row in current_sets
        if row["state"] == "completed" and not row["reviewed"]
    ]
    active_current = [
        row for row in current_sets if row["state"] in {"queued", "running"}
    ]
    interrupted_current = [
        row for row in current_sets if row["state"] == "interrupted"
    ]
    failed_current = [
        row for row in current_sets if row["state"] == "failed"
    ]
    terminal_review = reviews[-1] if reviews and isinstance(
        reviews[-1], Mapping
    ) else None
    review_matches_decision = bool(
        terminal_review
        and terminal_review.get("final_decision_identity")
        == decision_identity
        and _is_identity(terminal_review.get("review_identity"))
    )

    if completed_current:
        workflow_state = "evidence_ready_for_review"
    elif active_current:
        workflow_state = "future_generation_in_progress"
    elif interrupted_current:
        workflow_state = "future_generation_interrupted"
    elif failed_current:
        workflow_state = "future_generation_failed"
    elif review_matches_decision:
        workflow_state = "review_recorded_decision_refrozen"
    else:
        workflow_state = "ready_to_explore"

    completed_sets = [
        row for row in normalized_sets if row["state"] == "completed"
    ]
    reviewed_sets = [row for row in normalized_sets if row["reviewed"]]
    registered_scenarios = sum(
        len(row["branch_minutes"]) for row in normalized_sets
    )
    evidence_summary = {
        "future_sets": len(normalized_sets),
        "current_future_sets": len(current_sets),
        "completed_future_sets": len(completed_sets),
        "reviewed_future_sets": len(reviewed_sets),
        "registered_scenarios": registered_scenarios,
        "action_divergence_scenarios": sum(
            row["action_divergence_scenarios"] for row in completed_sets
        ),
        "local_attribution_scenarios": sum(
            row["local_attribution_scenarios"] for row in completed_sets
        ),
        "descriptive_future_difference_scenarios": sum(
            row["descriptive_future_difference_scenarios"]
            for row in completed_sets
        ),
    }
    generation_status = (
        "evidence_complete" if completed_current else
        "running" if active_current else
        "interrupted" if interrupted_current else
        "failed" if failed_current else
        "available"
    )
    mechanism_status = (
        "scenario_evidence_available"
        if any(
            row["scenario_evidence_available"] for row in completed_current
        )
        else "aggregate_only"
        if completed_current else "awaiting_future_set"
    )
    review_status = (
        "review_recorded" if review_matches_decision else
        "review_available" if completed_current else
        "optional_not_started"
    )
    stages = [
        _stage(
            "freeze_intervention", "decision_frozen",
            source_identity=decision_identity,
            tactic=decision["tactic"],
            rotation=decision["rotation"],
        ),
        _stage(
            "generate_bounded_futures", generation_status,
            current_future_sets=len(current_sets),
            registered_scenarios=sum(
                len(row["branch_minutes"]) for row in current_sets
            ),
        ),
        _stage(
            "inspect_local_mechanism", mechanism_status,
            action_divergence_scenarios=sum(
                row["action_divergence_scenarios"]
                for row in completed_current
            ),
            local_attribution_scenarios=sum(
                row["local_attribution_scenarios"]
                for row in completed_current
            ),
        ),
        _stage(
            "record_manager_review", review_status,
            source_identity=(
                terminal_review.get("review_identity")
                if review_matches_decision else None
            ),
            intent=(
                terminal_review.get("intent")
                if review_matches_decision else None
            ),
        ),
        _stage(
            "advance_official_world", "ready_for_official_match",
            official_match_is_only_season_advance_authority=True,
        ),
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "workflow_state": workflow_state,
        "season_id": season_id,
        "season_revision": revision,
        "fixture": fixture,
        "frozen_intervention": {
            "decision": decision,
            "decision_identity": decision_identity,
        },
        "stages": stages,
        "explorations": normalized_sets,
        "evidence_summary": evidence_summary,
        "allowed_actions": {
            "freeze_decision": False,
            "request_future_set": True,
            "review_future_set": bool(completed_current),
            "advance_official_match": True,
        },
        "continuity_gaps": sorted(set(gaps)),
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": _BOUNDARY,
    }
    payload["workspace_identity"] = _identity(payload)
    return payload


def validate_manager_intervention_workspace(
    workspace: Mapping[str, Any],
    *,
    season: Mapping[str, Any],
    manager_future_sets: Sequence[Mapping[str, Any]],
) -> None:
    """Reject edited workflow state even if its outer identity was rehashed."""
    expected = build_manager_intervention_workspace(
        season, manager_future_sets,
    )
    if dict(workspace) != expected:
        raise ValueError("manager intervention workspace replay mismatch")


__all__ = [
    "MAX_VISIBLE_EXPLORATIONS",
    "SCHEMA_VERSION",
    "build_manager_intervention_workspace",
    "validate_manager_intervention_workspace",
]
