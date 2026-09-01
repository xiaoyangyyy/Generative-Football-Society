"""One replayable evidence thread across a managed fixture's world lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _official_action_semantic_examples(
    action: Mapping[str, Any], *, retained_records: int,
) -> dict[str, Any]:
    """Summarize only the bounded official examples, never all records."""
    raw_examples = action.get("examples")
    examples = (
        [row for row in raw_examples if isinstance(row, Mapping)]
        if isinstance(raw_examples, list) else []
    )
    modes = [
        str((row.get("policy_signal") or {}).get("mode") or "legacy_unclassified")
        if isinstance(row.get("policy_signal"), Mapping)
        else "legacy_unclassified"
        for row in examples
    ]
    truncated = action.get("examples_truncated") is True
    retained = len(examples)
    return {
        "retained_semantic_examples": retained,
        "semantic_examples_truncated": truncated,
        "semantic_example_coverage_complete": bool(
            retained_records > 0
            and retained == retained_records
            and not truncated
        ),
        "semantic_cross_action_examples": sum(
            row.get("actual_action") == "cross" for row in examples
        ),
        "semantic_direct_preference_examples": modes.count(
            "direct_preference"
        ),
        "semantic_suppression_only_examples": modes.count(
            "suppression_only"
        ),
        "semantic_no_signal_examples": modes.count("none"),
        "semantic_legacy_unclassified_examples": modes.count(
            "legacy_unclassified"
        ),
        "semantic_hold_reference_redistribution_examples": sum(
            isinstance(row.get("reference_action_effect"), Mapping)
            and row["reference_action_effect"].get(
                "received_redistributed_probability"
            ) is True
            for row in examples
        ),
        "semantic_direct_cross_ball_event_examples": sum(
            row.get("actual_action") == "cross"
            and isinstance(row.get("runtime_link"), Mapping)
            and row["runtime_link"].get("direct_ball_event_identity") is True
            for row in examples
        ),
    }


def _stage(stage_id: str, status: str, **facts: Any) -> dict[str, Any]:
    payload = {
        "stage_id": stage_id,
        "status": status,
        **copy.deepcopy(facts),
    }
    payload["stage_identity"] = _identity(payload)
    return payload


def _link(
    source: str, target: str, status: str, link_type: str, **facts: Any,
) -> dict[str, Any]:
    payload = {
        "source_stage": source,
        "target_stage": target,
        "status": status,
        "link_type": link_type,
        "causal_effect_authorized": False,
        **copy.deepcopy(facts),
    }
    payload["link_identity"] = _identity(payload)
    return payload


def build_manager_world_evolution_thread(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one connected view; never infer a missing causal edge."""
    lifecycle = str(entry.get("lifecycle_state") or "")
    pending = lifecycle == "frozen_awaiting_execution"
    decision = entry.get("decision")
    decision = decision if isinstance(decision, Mapping) else {}
    decision_identity = str(entry.get("decision_identity") or "")

    review_trace = entry.get("future_review_execution_trace")
    review_trace = review_trace if isinstance(review_trace, Mapping) else {}
    terminal_review = review_trace.get("terminal_review")
    terminal_review = (
        terminal_review if isinstance(terminal_review, Mapping) else {}
    )
    if review_trace.get("available") is not True:
        review_status = "not_reviewed"
    elif terminal_review.get("linked_to_final_selection") is True:
        review_status = "selected_for_fixture"
    else:
        review_status = "superseded_before_execution"
    review_stage = _stage(
        "prematch_future_review",
        review_status,
        source_identity=review_trace.get("trace_identity"),
        review_identity=terminal_review.get("review_identity"),
        retained_mechanism_examples=int(
            terminal_review.get("retained_mechanism_examples") or 0
        ),
        cross_action_mechanism_examples=int(
            terminal_review.get("cross_action_mechanism_examples") or 0
        ),
        direct_preference_mechanism_examples=int(
            terminal_review.get("direct_preference_mechanism_examples") or 0
        ),
        suppression_only_mechanism_examples=int(
            terminal_review.get("suppression_only_mechanism_examples") or 0
        ),
        hold_reference_redistribution_examples=int(
            terminal_review.get("hold_reference_redistribution_examples") or 0
        ),
        nonzero_cross_descriptive_windows=int(
            terminal_review.get("nonzero_cross_descriptive_windows") or 0
        ),
        review_intent=terminal_review.get("intent"),
        evidence_level=terminal_review.get("evidence_level"),
        fixed_scenario_budget=int(
            terminal_review.get("fixed_scenario_budget") or 0
        ),
        eligible_scenarios=int(
            terminal_review.get("eligible_scenarios") or 0
        ),
        verified_anchor_scenarios=int(
            terminal_review.get("verified_anchor_scenarios") or 0
        ),
        action_divergence_scenarios=int(
            terminal_review.get("action_divergence_scenarios") or 0
        ),
        local_attribution_scenarios=int(
            terminal_review.get("local_attribution_scenarios") or 0
        ),
        descriptive_future_difference_scenarios=int(
            terminal_review.get(
                "descriptive_future_difference_scenarios"
            ) or 0
        ),
        timing_sensitivity_observed=terminal_review.get(
            "timing_sensitivity_observed"
        ),
        ranking_performed=terminal_review.get("ranking_performed"),
        best_branch_time=terminal_review.get("best_branch_time"),
        reviewed_scenarios=copy.deepcopy(
            terminal_review.get("reviewed_scenarios") or []
        ),
        evidence_authority="simulator_mechanism_review_only",
    )
    decision_stage = _stage(
        "frozen_manager_decision",
        "frozen",
        source_identity=decision_identity,
        tactic=str(decision.get("tactic") or ""),
        rotation=str(decision.get("rotation") or ""),
        evidence_authority="authoritative_fixture_selection",
    )

    execution = entry.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    binding = execution.get("tactical_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    if pending:
        tactical_status = "awaiting_execution"
    elif execution.get("available") is True and binding.get("available") is True:
        tactical_status = "runtime_verified"
    else:
        tactical_status = "execution_evidence_unavailable"
    tactical_stage = _stage(
        "official_tactical_runtime",
        tactical_status,
        source_identity=binding.get("binding_identity"),
        match_id=execution.get("match_id"),
        applied_tactic=binding.get("applied_tactic"),
        evidence_authority=(
            "direct_engine_binding"
            if tactical_status == "runtime_verified" else "unavailable"
        ),
    )

    action = execution.get("world_model_action_execution")
    action = action if isinstance(action, Mapping) else {}
    action_reason = str(action.get("reason") or "")
    if pending:
        action_status = "awaiting_execution"
    elif action.get("available") is True:
        action_status = str(action.get("evidence_state") or "available")
    elif action_reason == "world_model_not_configured_for_match":
        action_status = "not_applicable_stable_mode"
    elif action_reason == "legacy_report_without_world_model_layer":
        action_status = "legacy_evidence_unavailable"
    else:
        action_status = "action_evidence_unavailable"
    action_counts = action.get("retained_record_evidence")
    action_counts = (
        action_counts if isinstance(action_counts, Mapping) else {}
    )
    retained_records = int(action_counts.get("records") or 0)
    action_semantics = _official_action_semantic_examples(
        action, retained_records=retained_records,
    )
    retained_record_semantics = action.get("retained_record_semantics")
    retained_record_semantics = (
        copy.deepcopy(dict(retained_record_semantics))
        if action.get("schema_version") == 2
        and isinstance(retained_record_semantics, Mapping)
        else None
    )
    action_stage = _stage(
        "official_world_model_actions",
        action_status,
        source_identity=action.get("evidence_identity"),
        match_id=action.get("match_id"),
        team=action.get("team"),
        retained_records=retained_records,
        **action_semantics,
        retained_record_semantics=retained_record_semantics,
        resolved_action_decisions=int(
            action_counts.get("resolved_action_decisions") or 0
        ),
        influenced_decisions=int(
            action_counts.get("influenced_decisions") or 0
        ),
        attribution_eligible_decisions=int(
            action_counts.get("attribution_eligible_decisions") or 0
        ),
        locally_attributable_action_changes=int(
            action_counts.get("locally_attributable_action_changes") or 0
        ),
        direct_ball_event_links=int(
            action_counts.get("direct_ball_event_links") or 0
        ),
        changed_direct_ball_event_links=int(
            action_counts.get("changed_direct_ball_event_links") or 0
        ),
        decision_only_no_trajectory=int(
            action_counts.get("decision_only_no_trajectory") or 0
        ),
        unresolved_ball_event_links=int(
            action_counts.get("unresolved_or_missing_ball_event_links") or 0
        ),
        source_match_opportunities=int(
            action_counts.get("source_match_opportunities") or 0
        ),
        source_records_truncated=(
            action_counts.get("source_records_truncated") is True
        ),
        manager_record_coverage_complete=(
            action_counts.get("manager_record_coverage_complete") is True
        ),
        reason=action_reason or None,
        evidence_authority=(
            "simulator_local_action_and_runtime_identity"
            if action.get("available") is True else "unavailable_or_not_applicable"
        ),
    )

    observed = entry.get("observed_result")
    observed = observed if isinstance(observed, Mapping) else {}
    if observed:
        result_status = "observed_descriptive"
    elif pending:
        result_status = "awaiting_execution"
    else:
        result_status = "result_evidence_unavailable"
    result_stage = _stage(
        "observed_match_result",
        result_status,
        source_identity=_identity(observed) if observed else None,
        score=copy.deepcopy(observed.get("score")),
        outcome=observed.get("outcome"),
        points_earned=observed.get("points_earned"),
        descriptive_only=observed.get("descriptive_only") is True,
        evidence_authority="descriptive_match_fact" if observed else "unavailable",
    )

    accounting = entry.get("long_term_accounting")
    accounting = accounting if isinstance(accounting, Mapping) else {}
    persistent = accounting.get("persistent_team_state_delta")
    persistent = persistent if isinstance(persistent, Mapping) else {}
    if persistent.get("available") is True:
        persistent_status = (
            "after_recovery_recorded"
            if persistent.get("recovery_complete") is True else
            "after_match_recorded_recovery_pending"
        )
    elif pending:
        persistent_status = "awaiting_execution"
    else:
        persistent_status = "persistent_state_evidence_unavailable"
    match_delta = persistent.get("match_delta")
    match_delta = match_delta if isinstance(match_delta, Mapping) else {}
    persistent_stage = _stage(
        "persistent_world_state",
        persistent_status,
        source_identity=persistent.get("transition_identity"),
        recovery_complete=persistent.get("recovery_complete") is True,
        metrics_delta=copy.deepcopy(match_delta.get("metrics_delta")),
        transition_summary=copy.deepcopy(match_delta.get("summary")),
        evidence_authority=(
            "authoritative_simulator_state_transition"
            if persistent.get("available") is True else "unavailable"
        ),
    )

    if review_status == "selected_for_fixture":
        review_link_status = "identity_bound"
    elif review_status == "superseded_before_execution":
        review_link_status = "continuity_broken"
    else:
        review_link_status = "not_applicable"
    if tactical_status == "runtime_verified":
        decision_runtime_status = "direct_runtime_binding"
    elif pending:
        decision_runtime_status = "awaiting_execution"
    else:
        decision_runtime_status = "execution_evidence_unavailable"
    action_available = action.get("available") is True
    same_match = bool(
        tactical_status == "runtime_verified"
        and action_available
        and isinstance(tactical_stage.get("match_id"), str)
        and bool(tactical_stage.get("match_id"))
        and tactical_stage.get("match_id") == action_stage.get("match_id")
    )
    if same_match:
        runtime_action_status = "same_official_match_context"
    elif pending:
        runtime_action_status = "awaiting_execution"
    elif action_status == "not_applicable_stable_mode":
        runtime_action_status = "not_applicable"
    else:
        runtime_action_status = "evidence_unavailable"
    direct_links = int(action_counts.get("direct_ball_event_links") or 0)
    local_changes = int(
        action_counts.get("locally_attributable_action_changes") or 0
    )
    if action_available and direct_links:
        action_event_status = "direct_identity_links_available"
    elif action_available:
        action_event_status = "decision_records_without_direct_ball_link"
    elif pending:
        action_event_status = "awaiting_execution"
    else:
        action_event_status = "not_available"
    if observed and persistent.get("available") is True:
        result_world_status = "engine_state_transition_recorded"
    elif pending:
        result_world_status = "awaiting_execution"
    else:
        result_world_status = "state_transition_unavailable"

    stages = [
        review_stage, decision_stage, tactical_stage,
        action_stage, result_stage, persistent_stage,
    ]
    links = [
        _link(
            "prematch_future_review", "frozen_manager_decision",
            review_link_status, "decision_identity_binding",
            source_identity=terminal_review.get("final_decision_identity"),
            target_identity=decision_identity,
        ),
        _link(
            "frozen_manager_decision", "official_tactical_runtime",
            decision_runtime_status, "official_engine_tactical_binding",
            selected_tactic=decision.get("tactic"),
            applied_tactic=binding.get("applied_tactic"),
        ),
        _link(
            "official_tactical_runtime", "official_world_model_actions",
            runtime_action_status, "same_match_context_not_causal_direction",
            tactical_match_id=tactical_stage.get("match_id"),
            action_match_id=action_stage.get("match_id"),
        ),
        _link(
            "official_world_model_actions", "observed_match_result",
            action_event_status, "local_action_and_event_identity_only",
            locally_attributable_action_changes=local_changes,
            direct_ball_event_links=direct_links,
            downstream_result_attribution_authorized=False,
        ),
        _link(
            "observed_match_result", "persistent_world_state",
            result_world_status, "authoritative_engine_state_transition",
            manager_or_action_policy_effect_authorized=False,
        ),
    ]

    gaps = []
    if review_status == "superseded_before_execution":
        gaps.append("review_to_final_decision_continuity_broken")
    if not pending and tactical_status != "runtime_verified":
        gaps.append("tactical_runtime_binding_unavailable")
    if (
        not pending
        and action_status not in {
            "locally_attributable_action_changes_observed",
            "probability_influence_without_realized_action_change",
            "no_nonzero_action_influence_observed",
            "not_applicable_stable_mode",
        }
    ):
        gaps.append("world_model_action_evidence_unavailable")
    if not pending and not observed:
        gaps.append("observed_result_unavailable")
    if not pending and persistent.get("available") is not True:
        gaps.append("persistent_world_state_unavailable")
    official_runtime_complete = bool(
        tactical_status == "runtime_verified"
        and observed
        and persistent.get("available") is True
    )
    world_model_runtime_complete = bool(
        official_runtime_complete and action_available and same_match
    )
    final_decision_identity_bound = review_status == "selected_for_fixture"
    if review_status == "not_reviewed":
        review_world_status = "not_reviewed"
    elif not final_decision_identity_bound:
        review_world_status = "review_superseded"
    elif pending:
        review_world_status = "awaiting_official_match"
    elif tactical_status != "runtime_verified":
        review_world_status = "tactical_runtime_binding_unavailable"
    elif not action_available or not same_match:
        review_world_status = "official_action_evidence_unavailable"
    else:
        review_world_status = "complete"
    archived_scenarios = terminal_review.get("reviewed_scenarios")
    archived_scenarios = (
        archived_scenarios if isinstance(archived_scenarios, list) else []
    )
    review_world_certificate = {
        "schema_version": 1,
        "status": review_world_status,
        "review_identity": terminal_review.get("review_identity"),
        "review_trace_identity": review_trace.get("trace_identity"),
        "decision_identity": decision_identity,
        "tactical_binding_identity": binding.get("binding_identity"),
        "official_action_evidence_identity": action.get("evidence_identity"),
        "official_match_id": action.get("match_id"),
        "scenario_archive_count": len(archived_scenarios),
        "scenario_evidence_retained": bool(archived_scenarios),
        "final_decision_identity_bound": final_decision_identity_bound,
        "runtime_tactic_verified": tactical_status == "runtime_verified",
        "official_action_same_match": same_match,
        "scenario_to_runtime_opportunity_matching_performed": False,
        "outcome_comparison_performed": False,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "review identity, final decision, official tactical binding and "
            "same-match action evidence continuity only; archived simulated "
            "scenarios are not matched to official runtime opportunities"
        ),
    }
    review_world_certificate["certificate_identity"] = _identity(
        review_world_certificate
    )
    reviewed_world_model_chain_complete = review_world_status == "complete"
    if pending:
        thread_state = "awaiting_official_match"
    elif world_model_runtime_complete:
        thread_state = "official_world_evolution_observed"
    elif official_runtime_complete:
        thread_state = "official_match_observed_action_optional_or_unavailable"
    else:
        thread_state = "official_evidence_thread_incomplete"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "fixture_id": entry.get("fixture_id"),
        "matchday": entry.get("matchday"),
        "thread_state": thread_state,
        "stages": stages,
        "links": links,
        "continuity_gaps": gaps,
        "official_runtime_chain_complete": official_runtime_complete,
        "world_model_runtime_chain_complete": world_model_runtime_complete,
        "reviewed_world_model_chain_complete": (
            reviewed_world_model_chain_complete
        ),
        "review_to_official_world": review_world_certificate,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "one evidence thread across review, selection, runtime, retained "
            "world-model actions, descriptive result and simulator state transition; "
            "only declared identity and local-action links are authorized"
        ),
    }
    payload["thread_identity"] = _identity(payload)
    return payload


def validate_manager_world_evolution_thread(
    thread: Mapping[str, Any], *, entry: Mapping[str, Any],
) -> None:
    """Replay the entire thread from its authoritative ledger components."""
    expected = build_manager_world_evolution_thread(entry)
    if dict(thread) != expected:
        raise ValueError("manager world evolution thread replay mismatch")


def manager_world_evolution_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize connected evidence coverage, never outcome effectiveness."""
    threads = [
        row.get("world_evolution_thread")
        for row in entries
        if isinstance(row.get("world_evolution_thread"), Mapping)
    ]
    stages = [
        {
            stage.get("stage_id"): stage
            for stage in thread.get("stages") or []
            if isinstance(stage, Mapping)
        }
        for thread in threads
    ]
    payload = {
        "schema_version": 1,
        "fixtures_with_world_evolution_thread": len(threads),
        "awaiting_official_match": sum(
            thread.get("thread_state") == "awaiting_official_match"
            for thread in threads
        ),
        "official_runtime_chain_complete": sum(
            thread.get("official_runtime_chain_complete") is True
            for thread in threads
        ),
        "world_model_runtime_chain_complete": sum(
            thread.get("world_model_runtime_chain_complete") is True
            for thread in threads
        ),
        "reviewed_world_model_chain_complete": sum(
            thread.get("reviewed_world_model_chain_complete") is True
            for thread in threads
        ),
        "review_to_official_world_status_counts": {
            status: sum(
                (thread.get("review_to_official_world") or {}).get("status")
                == status
                for thread in threads
            )
            for status in (
                "complete", "awaiting_official_match", "review_superseded",
                "tactical_runtime_binding_unavailable",
                "official_action_evidence_unavailable", "not_reviewed",
            )
        },
        "review_selected_for_fixture": sum(
            row.get("prematch_future_review", {}).get("status")
            == "selected_for_fixture"
            for row in stages
        ),
        "runtime_tactic_verified": sum(
            row.get("official_tactical_runtime", {}).get("status")
            == "runtime_verified"
            for row in stages
        ),
        "official_world_model_action_evidence": sum(
            row.get("official_world_model_actions", {}).get("source_identity")
            is not None
            for row in stages
        ),
        "local_action_changes": sum(
            int(
                row.get("official_world_model_actions", {}).get(
                    "locally_attributable_action_changes", 0,
                )
            )
            for row in stages
        ),
        "persistent_world_transitions": sum(
            row.get("persistent_world_state", {}).get("source_identity")
            is not None
            for row in stages
        ),
        "threads_with_continuity_gaps": sum(
            bool(thread.get("continuity_gaps")) for thread in threads
        ),
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "connected evidence coverage only; completeness does not imply "
            "decision quality, score improvement or causal effectiveness"
        ),
    }
    payload["evidence_identity"] = _identity(payload)
    return payload


__all__ = [
    "SCHEMA_VERSION", "build_manager_world_evolution_thread",
    "manager_world_evolution_summary",
    "validate_manager_world_evolution_thread",
]
