"""Replayable review receipts for manager-bound world-model future sets."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from src.product.manager_future import validate_manager_future_context_shape
from src.product.match_plan import WorldModelForkSetPlan
from src.product.season import ManagerDecision
from src.product.world_model_fork_set import (
    summarize_fork_set_scenario_evidence,
    validate_fork_set_scenario_evidence,
)


LEGACY_SCENARIO_REVIEW_SCHEMA_VERSION = 2
MECHANISM_REVIEW_SCHEMA_VERSION = 3
REVIEW_SCHEMA_VERSION = 4
REVIEW_INTENTS = {"keep_after_review", "revise_after_review"}
MAX_REVIEWS_PER_FIXTURE = 16
_BOUNDARY = (
    "records an explicit manager interaction with simulator future-set evidence "
    "and the resulting frozen decision only; it does not recommend a best time, "
    "predict a score, estimate decision quality, or establish causality"
)
_AUTHORITY = {
    "descriptive_simulator_timing_sensitivity": True,
    "best_time_recommendation": False,
    "match_outcome_causality": False,
    "population_inference": False,
    "real_football_causality": False,
    "promotion_authorized": False,
}
_SUMMARY_FIELDS = {
    "fixed_scenario_budget",
    "eligible_scenarios",
    "verified_anchor_scenarios",
    "action_divergence_scenarios",
    "local_attribution_scenarios",
    "descriptive_future_difference_scenarios",
    "timing_sensitivity_observed",
    "ranking_performed",
    "best_branch_time",
}
_FUTURE_STATUSES = {
    "descriptive_only_ineligible",
    "no_realized_action_divergence",
    "action_divergence_without_local_attribution",
    "local_action_divergence_with_descriptive_future_difference",
    "local_action_divergence_without_measured_future_difference",
}


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _is_hex_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _decision_identity(value: Mapping[str, Any]) -> str:
    decision = ManagerDecision.from_payload(value)
    return _identity(decision.as_dict())


def _validated_task_evidence(
    task_id: str,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[
    dict[str, Any], WorldModelForkSetPlan, dict[str, Any],
    list[dict[str, Any]] | None,
]:
    if (
        not isinstance(task_id, str)
        or not task_id.isalnum()
        or not 1 <= len(task_id) <= 64
    ):
        raise ValueError("manager future review task identity is invalid")
    if not isinstance(request, Mapping) or not isinstance(result, Mapping):
        raise ValueError("manager future review task evidence is invalid")
    context = validate_manager_future_context_shape(
        request.get("manager_context")
    )
    plan = WorldModelForkSetPlan.from_payload(request.get("plan"))
    fixture = context["fixture"]
    if (
        request.get("home") != fixture["home"]
        or request.get("away") != fixture["away"]
        or request.get("fast") is not context["fast"]
        or plan.seed != context["match_seed"]
        or plan.home_tactic != context["home_tactic"]
        or plan.away_tactic != context["away_tactic"]
    ):
        raise ValueError("manager future review request controls are inconsistent")
    expected_digest = {
        "season_id": context["season_id"],
        "season_revision": context["season_revision"],
        "fixture_id": fixture["fixture_id"],
        "matchday": fixture["matchday"],
        "manager_team": context["manager_team"],
        "decision_identity": context["decision_identity"],
        "context_identity": context["context_identity"],
        "claim_boundary": context["claim_boundary"],
    }
    aggregate = result.get("aggregate")
    authority = result.get("claim_authority")
    if (
        result.get("fork_set_id") != task_id
        or result.get("status") != "complete"
        or result.get("manager_context") != expected_digest
        or not isinstance(aggregate, Mapping)
        or not isinstance(authority, Mapping)
        or dict(authority) != _AUTHORITY
        or set(aggregate) != {
            "eligible_scenarios",
            "verified_anchor_scenarios",
            "action_divergence_scenarios",
            "local_attribution_scenarios",
            "descriptive_future_difference_scenarios",
            "timing_sensitivity_observed",
            "status_counts",
            "ranking_performed",
            "best_branch_time",
        }
        or aggregate.get("ranking_performed") is not False
        or aggregate.get("best_branch_time") is not None
        or not isinstance(aggregate.get("timing_sensitivity_observed"), bool)
    ):
        raise ValueError("manager future review completion evidence is invalid")
    budget = len(plan.branch_times_sec)
    count_fields = (
        "eligible_scenarios",
        "verified_anchor_scenarios",
        "action_divergence_scenarios",
        "local_attribution_scenarios",
        "descriptive_future_difference_scenarios",
    )
    if any(
        isinstance(aggregate.get(field), bool)
        or not isinstance(aggregate.get(field), int)
        or not 0 <= int(aggregate[field]) <= budget
        for field in count_fields
    ):
        raise ValueError("manager future review aggregate counts are invalid")
    status_counts = aggregate.get("status_counts")
    if (
        not isinstance(status_counts, Mapping)
        or not status_counts
        or set(status_counts) - _FUTURE_STATUSES
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in status_counts.values()
        )
        or sum(status_counts.values()) != budget
    ):
        raise ValueError("manager future review status counts are invalid")
    summary = {
        "fixed_scenario_budget": budget,
        **{field: int(aggregate[field]) for field in count_fields},
        "timing_sensitivity_observed": aggregate[
            "timing_sensitivity_observed"
        ],
        "ranking_performed": False,
        "best_branch_time": None,
    }
    scenario_evidence = result.get("scenario_evidence")
    validated_scenarios = (
        None
        if scenario_evidence is None else
        validate_fork_set_scenario_evidence(
            scenario_evidence, plan.branch_times_sec,
        )
    )
    if validated_scenarios is not None:
        scenario_versions = {
            row.get("schema_version") for row in validated_scenarios
        }
        if len(scenario_versions) != 1:
            raise ValueError(
                "manager future review scenario versions are mixed"
            )
        rebuilt = summarize_fork_set_scenario_evidence(
            validated_scenarios, plan.branch_times_sec,
        )
        if any(aggregate.get(key) != value for key, value in rebuilt.items()):
            raise ValueError(
                "manager future review scenario aggregate is inconsistent"
            )
    return context, plan, summary, validated_scenarios


def build_manager_future_review(
    *,
    task_id: str,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    final_decision: Mapping[str, Any],
    intent: str,
) -> dict[str, Any]:
    """Bind an explicit keep/revise interaction to completed bounded evidence."""
    if intent not in REVIEW_INTENTS:
        raise ValueError("manager future review intent is invalid")
    context, plan, summary, scenarios = _validated_task_evidence(
        task_id, request, result,
    )
    normalized_decision = ManagerDecision.from_payload(final_decision).as_dict()
    final_identity = _decision_identity(normalized_decision)
    reviewed_identity = str(context["decision_identity"])
    changed = final_identity != reviewed_identity
    if (
        intent == "keep_after_review" and changed
        or intent == "revise_after_review" and not changed
    ):
        raise ValueError("manager future review intent does not match decision")
    fixture = context["fixture"]
    receipt_version = 1
    if scenarios is not None:
        scenario_version = scenarios[0]["schema_version"]
        receipt_version = {
            1: LEGACY_SCENARIO_REVIEW_SCHEMA_VERSION,
            2: MECHANISM_REVIEW_SCHEMA_VERSION,
            3: REVIEW_SCHEMA_VERSION,
        }[scenario_version]
    payload = {
        "schema_version": receipt_version,
        "task_id": task_id,
        "season_id": context["season_id"],
        "source_season_revision": context["season_revision"],
        "fixture_id": fixture["fixture_id"],
        "matchday": fixture["matchday"],
        "manager_team": context["manager_team"],
        "context_identity": context["context_identity"],
        "reviewed_decision_identity": reviewed_identity,
        "final_decision_identity": final_identity,
        "intent": intent,
        "decision_changed": changed,
        "branch_times_sec": list(plan.branch_times_sec),
        "evidence_summary": summary,
        "claim_authority": dict(_AUTHORITY),
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": _BOUNDARY,
    }
    if scenarios is not None:
        payload["scenario_evidence"] = scenarios
    return {**payload, "review_identity": _identity(payload)}


def validate_manager_future_review(
    review: Mapping[str, Any],
    *,
    season_id: str,
    fixture_id: str,
    matchday: int,
    manager_team: str,
) -> None:
    """Validate a persisted receipt without depending on the mutable task queue."""
    base_required = {
        "schema_version",
        "task_id",
        "season_id",
        "source_season_revision",
        "fixture_id",
        "matchday",
        "manager_team",
        "context_identity",
        "reviewed_decision_identity",
        "final_decision_identity",
        "intent",
        "decision_changed",
        "branch_times_sec",
        "evidence_summary",
        "claim_authority",
        "outcome_effect_estimate",
        "causal_effect_authorized",
        "claim_boundary",
        "review_identity",
    }
    schema_version = review.get("schema_version") if isinstance(
        review, Mapping,
    ) else None
    required = (
        base_required | {"scenario_evidence"}
        if schema_version in {
            LEGACY_SCENARIO_REVIEW_SCHEMA_VERSION,
            MECHANISM_REVIEW_SCHEMA_VERSION,
            REVIEW_SCHEMA_VERSION,
        } else base_required
    )
    if (
        not isinstance(review, Mapping)
        or schema_version not in {
            1,
            LEGACY_SCENARIO_REVIEW_SCHEMA_VERSION,
            MECHANISM_REVIEW_SCHEMA_VERSION,
            REVIEW_SCHEMA_VERSION,
        }
        or set(review) != required
    ):
        raise ValueError("manager future review fields are invalid")
    task_id = review.get("task_id")
    branch_times = review.get("branch_times_sec")
    summary = review.get("evidence_summary")
    if (
        not isinstance(task_id, str)
        or not task_id.isalnum()
        or not 1 <= len(task_id) <= 64
        or review.get("season_id") != season_id
        or review.get("fixture_id") != fixture_id
        or review.get("matchday") != matchday
        or review.get("manager_team") != manager_team
        or isinstance(review.get("source_season_revision"), bool)
        or not isinstance(review.get("source_season_revision"), int)
        or int(review["source_season_revision"]) < 0
        or not _is_hex_identity(review.get("context_identity"))
        or not _is_hex_identity(review.get("reviewed_decision_identity"))
        or not _is_hex_identity(review.get("final_decision_identity"))
        or review.get("intent") not in REVIEW_INTENTS
        or not isinstance(review.get("decision_changed"), bool)
        or review.get("decision_changed")
        is not (
            review.get("reviewed_decision_identity")
            != review.get("final_decision_identity")
        )
        or (
            review.get("intent") == "keep_after_review"
            and review.get("decision_changed") is not False
        )
        or (
            review.get("intent") == "revise_after_review"
            and review.get("decision_changed") is not True
        )
        or not isinstance(branch_times, list)
        or not 2 <= len(branch_times) <= 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 5400
            or (
                index > 0
                and float(value) <= float(branch_times[index - 1])
            )
            for index, value in enumerate(branch_times)
        )
        or not isinstance(summary, Mapping)
        or set(summary) != _SUMMARY_FIELDS
        or summary.get("fixed_scenario_budget") != len(branch_times)
        or not isinstance(summary.get("timing_sensitivity_observed"), bool)
        or summary.get("ranking_performed") is not False
        or summary.get("best_branch_time") is not None
        or review.get("claim_authority") != _AUTHORITY
        or review.get("outcome_effect_estimate") is not None
        or review.get("causal_effect_authorized") is not False
        or review.get("claim_boundary") != _BOUNDARY
    ):
        raise ValueError("manager future review values are invalid")
    count_fields = _SUMMARY_FIELDS - {
        "fixed_scenario_budget",
        "timing_sensitivity_observed",
        "ranking_performed",
        "best_branch_time",
    }
    budget = len(branch_times)
    if any(
        isinstance(summary.get(field), bool)
        or not isinstance(summary.get(field), int)
        or not 0 <= int(summary[field]) <= budget
        for field in count_fields
    ):
        raise ValueError("manager future review summary is invalid")
    if schema_version in {
        LEGACY_SCENARIO_REVIEW_SCHEMA_VERSION,
        MECHANISM_REVIEW_SCHEMA_VERSION,
        REVIEW_SCHEMA_VERSION,
    }:
        scenarios = validate_fork_set_scenario_evidence(
            review.get("scenario_evidence"), branch_times,
        )
        expected_scenario_version = {
            LEGACY_SCENARIO_REVIEW_SCHEMA_VERSION: 1,
            MECHANISM_REVIEW_SCHEMA_VERSION: 2,
            REVIEW_SCHEMA_VERSION: 3,
        }[schema_version]
        if any(
            row.get("schema_version") != expected_scenario_version
            for row in scenarios
        ):
            raise ValueError(
                "manager future review scenario version mismatch"
            )
        rebuilt = summarize_fork_set_scenario_evidence(
            scenarios, branch_times,
        )
        if (
            any(summary.get(field) != rebuilt[field] for field in count_fields)
            or summary.get("timing_sensitivity_observed")
            is not rebuilt["timing_sensitivity_observed"]
        ):
            raise ValueError(
                "manager future review scenario summary mismatch"
            )
    frozen = dict(review)
    observed = frozen.pop("review_identity")
    if observed != _identity(frozen):
        raise ValueError("manager future review identity mismatch")


__all__ = [
    "MAX_REVIEWS_PER_FIXTURE",
    "REVIEW_INTENTS",
    "build_manager_future_review",
    "validate_manager_future_review",
]
