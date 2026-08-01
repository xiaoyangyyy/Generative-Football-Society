"""Audit whether the coach LLM follows the world-model deliberation agenda."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.llm_decision_brief import (
    deliberation_portfolio_objective,
    llm_decision_brief_metadata_is_valid,
    llm_decision_brief_self_is_valid,
)


LLM_DELIBERATION_FOCUS_VERSION = 7
TASK_CONTRACTS = {
    "opponent_hypothesis": (
        "opponent_hypothesis", "opponent_belief_audit",
    ),
    "opponent_change_claim": (
        "opponent_change_claim", "opponent_change_claim_audit",
    ),
    "world_model_critique": (
        "world_model_critique", "world_model_critique_audit",
    ),
    "world_model_event_hypothesis": (
        "world_model_event_hypothesis", "world_model_event_hypothesis_audit",
    ),
    "world_model_event_option": (
        "world_model_event_option", "world_model_event_option_audit",
    ),
    "world_model_contrastive_claim": (
        "world_model_contrastive_claim",
        "world_model_contrastive_explanation_audit",
    ),
    "world_model_risk_constraint": (
        "world_model_risk_constraint", "world_model_risk_certificate_audit",
    ),
    "world_model_distributional_claim": (
        "world_model_distributional_claim",
        "world_model_distributional_claim_audit",
    ),
    "world_model_risk_preference": (
        "world_model_risk_preference", "world_model_risk_preference_audit",
    ),
    "opponent_information_query": (
        "opponent_information_query", "opponent_information_query_audit",
    ),
    "opponent_information_policy": (
        "opponent_information_policy",
        "opponent_information_policy_audit",
    ),
    "opponent_information_adaptation": (
        "opponent_information_adaptation",
        "opponent_information_adaptation_audit",
    ),
    "opponent_response_hypothesis": (
        "opponent_response_hypothesis", "opponent_response_hypothesis_audit",
    ),
}


def llm_deliberation_focus_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": LLM_DELIBERATION_FOCUS_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "world_model_deliberation_agenda_selection",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-deliberation-focus:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_deliberation_focus(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list):
        return None
    tasks = [str(task).strip() for task in raw["tasks"]]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not 1 <= len(tasks) <= 3
        or len(tasks) != len(set(tasks))
        or any(task not in TASK_CONTRACTS for task in tasks)
        or not math.isfinite(confidence)
        or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "tasks": tasks,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def deliberation_task_enabled(
    plan: dict[str, Any], task: str,
    compute_allocation: dict[str, Any] | None = None,
) -> bool:
    """Legacy plans run normally; valid focus plans execute selected tasks only."""
    focus = validate_llm_deliberation_focus(
        plan.get("world_model_deliberation_focus")
    )
    if focus is None:
        return True
    if task not in focus["tasks"]:
        return False
    if compute_allocation is None:
        return True
    return bool(
        compute_allocation.get("accepted")
        and task in (compute_allocation.get("allocations") or {})
    )


def deliberation_task_skipped_audit(
    plan: dict[str, Any], task: str,
    compute_allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    focus = validate_llm_deliberation_focus(
        plan.get("world_model_deliberation_focus")
    )
    contract_key = TASK_CONTRACTS[task][0]
    selected = bool(focus and task in focus["tasks"])
    return {
        "version": LLM_DELIBERATION_FOCUS_VERSION,
        "accepted": False,
        "reason": (
            "task_not_authorized_by_compute_allocation"
            if selected and compute_allocation is not None
            else "task_not_selected_by_deliberation_focus"
        ),
        "task": task,
        "declared_focus_tasks": list((focus or {}).get("tasks") or []),
        "unfocused_contract_was_emitted": plan.get(contract_key) is not None,
        "compute_executed": False,
        "belief_or_memory_mutated": False,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_relax_downstream_validators": False,
        "causal_interpretation": False,
    }


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "llm-deliberation-focus-audit:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _audit_from_evidence(
    focus: dict[str, Any],
    brief: dict[str, Any],
    plan: dict[str, Any],
    *,
    focus_signature: str,
) -> dict[str, Any] | None:
    if not llm_decision_brief_self_is_valid(brief):
        return None
    agenda = brief.get("deliberation_agenda") or {}
    maximum = int(agenda.get("maximum_recommended_focus_tasks", 0))
    if len(focus["tasks"]) > maximum:
        return None
    agenda_tasks = {
        str(row.get("task")): row
        for row in agenda.get("tasks") or []
        if isinstance(row, dict)
    }
    selected = set(focus["tasks"])
    eligible = {
        task for task, row in agenda_tasks.items() if row.get("eligible")
    }
    emitted = {
        task for task, (contract_key, _) in TASK_CONTRACTS.items()
        if plan.get(contract_key) is not None
    }
    accepted = {
        task for task, (_, audit_key) in TASK_CONTRACTS.items()
        if (plan.get(audit_key) or {}).get("accepted")
    }
    unsupported = sorted(selected - eligible)
    unfocused = sorted(emitted - selected)
    missing = sorted(selected - emitted)
    rejected = sorted(selected - accepted)
    from src.match_engine.world_model.llm_deliberation_compute import (
        build_deliberation_compute_allocation,
        deliberation_compute_allocation_is_valid,
        deliberation_compute_limits,
    )

    expected_compute_allocation = build_deliberation_compute_allocation(
        brief, plan,
    )
    compute_allocation = plan.get(
        "world_model_deliberation_compute_allocation"
    ) or expected_compute_allocation
    compute_allocation_valid = deliberation_compute_allocation_is_valid(
        compute_allocation
    )
    compute_allocation_matches_model = bool(
        compute_allocation == expected_compute_allocation
    )
    compute_budget_mismatches = []

    def exceeds(audit: dict[str, Any], actual: str, limit: int) -> bool:
        if actual not in audit:
            return False
        try:
            return int(audit[actual]) > int(limit)
        except (TypeError, ValueError, OverflowError):
            return True

    if compute_allocation_valid:
        event_limits = deliberation_compute_limits(
            compute_allocation, "world_model_event_option",
        )
        event_audit = plan.get("world_model_event_option_audit") or {}
        for actual_key, limit_key in (
            ("member_evaluation_budget", "member_evaluation_cap"),
            ("trajectory_member_path_budget", "trajectory_member_path_cap"),
        ):
            if (
                "world_model_event_option" in accepted
                and actual_key not in event_audit
            ):
                compute_budget_mismatches.append(
                    f"world_model_event_option:missing_{actual_key}"
                )
            elif exceeds(
                event_audit, actual_key, event_limits.get(limit_key, 0),
            ):
                compute_budget_mismatches.append(
                    f"world_model_event_option:{actual_key}"
                )
        contrastive_limits = deliberation_compute_limits(
            compute_allocation, "world_model_contrastive_claim",
        )
        contrastive_audit = plan.get(
            "world_model_contrastive_explanation_audit"
        ) or {}
        if (
            "world_model_contrastive_claim" in accepted
            and "trajectory_member_path_budget" not in contrastive_audit
        ):
            compute_budget_mismatches.append(
                "world_model_contrastive_claim:"
                "missing_trajectory_member_path_budget"
            )
        elif exceeds(
            contrastive_audit, "trajectory_member_path_budget",
            contrastive_limits.get("trajectory_member_path_cap", 0),
        ):
            compute_budget_mismatches.append(
                "world_model_contrastive_claim:trajectory_member_path_budget"
            )
        repair_audit = plan.get("world_model_contrastive_repair_audit") or {}
        if (
            "world_model_contrastive_claim" in accepted
            and "total_member_trajectory_path_budget" not in repair_audit
        ):
            compute_budget_mismatches.append(
                "world_model_contrastive_claim:missing_repair_total_path_budget"
            )
        elif exceeds(
            repair_audit, "total_member_trajectory_path_budget",
            contrastive_limits.get(
                "repair_total_member_trajectory_path_cap", 0,
            ),
        ):
            compute_budget_mismatches.append(
                "world_model_contrastive_claim:repair_total_path_budget"
            )
    compute_task_outcomes = {}
    allocation_rows = compute_allocation.get("allocations") or {}
    for task, audit_key in (
        ("world_model_event_option", "world_model_event_option_audit"),
        (
            "world_model_contrastive_claim",
            "world_model_contrastive_explanation_audit",
        ),
    ):
        if task not in allocation_rows:
            continue
        task_audit = plan.get(audit_key) or {}
        reason = str(task_audit.get("reason", "accepted"))
        task_accepted = bool(task_audit.get("accepted"))
        budget_rejection = "budget_insufficient" in reason
        planned_limits = allocation_rows[task].get(
            "planned_resource_limits"
        ) or {}
        required_to_planned = []
        if task == "world_model_event_option":
            for required_key, cap_key in (
                ("member_evaluations_required", "member_evaluation_cap"),
                (
                    "trajectory_member_paths_required",
                    "trajectory_member_path_cap",
                ),
            ):
                if required_key in task_audit and cap_key in planned_limits:
                    required_to_planned.append(
                        int(task_audit[required_key])
                        <= int(planned_limits[cap_key])
                    )
        elif (
            "trajectory_member_paths_required" in task_audit
            and "trajectory_member_path_cap" in planned_limits
        ):
            required_to_planned.append(
                int(task_audit["trajectory_member_paths_required"])
                <= int(planned_limits["trajectory_member_path_cap"])
            )
        would_fit_planned = bool(
            budget_rejection and required_to_planned
            and all(required_to_planned)
        )
        if task == "world_model_event_option":
            artifact_score = float(task_audit.get(
                "conditional_gain_vs_best_fixed", 0.0,
            )) if task_accepted else 0.0
            useful = bool(task_accepted and artifact_score > 0.0)
        else:
            artifact_score = float(bool(task_audit.get(
                "directionally_faithful"
            ))) if task_accepted else 0.0
            useful = bool(task_accepted and artifact_score > 0.0)
        compute_task_outcomes[task] = {
            "accepted": task_accepted,
            "reason": reason,
            "conclusive": bool(task_accepted or budget_rejection),
            "useful_artifact": useful,
            "model_internal_artifact_score": artifact_score,
            "budget_rejection": budget_rejection,
            "would_fit_planned_budget": would_fit_planned,
            "compute_credits": int(allocation_rows[task].get(
                "compute_credits", 0,
            )),
            "compute_value_experiment": dict(allocation_rows[task].get(
                "compute_value_experiment"
            ) or {}),
            "outcome_scope": "model_internal_useful_artifact",
            "can_claim_match_outcome_causality": False,
        }
    experimental_budget_rejections = sorted(
        task for task, outcome in compute_task_outcomes.items()
        if (
            outcome.get("budget_rejection")
            and outcome.get("would_fit_planned_budget")
            and (outcome.get("compute_value_experiment") or {}).get(
                "randomized"
            )
            and (outcome.get("compute_value_experiment") or {}).get("arm")
            == "control"
        )
    )
    nonexperimental_rejected = sorted(
        set(rejected) - set(experimental_budget_rejections)
    )
    encouragement_audit = plan.get(
        "world_model_deliberation_encouragement_audit"
    ) or {}
    from src.match_engine.world_model.llm_deliberation_encouragement import (
        deliberation_encouragement_audit_is_valid,
    )

    encouragement_valid = bool(
        not encouragement_audit
        or deliberation_encouragement_audit_is_valid(
            encouragement_audit, plan, brief,
        )
    )
    encouragement = encouragement_audit.get("encouragement") or {}
    assigned_trial = bool(
        encouragement_audit.get("accepted")
        and encouragement.get("eligible")
        and encouragement.get("assigned_arm") in {"control", "treatment"}
    )
    encouragement_context = {
        "two_stage_active": bool(encouragement_audit.get("two_stage_active")),
        "audit_valid": encouragement_valid,
        "randomized_trial": assigned_trial,
        "assigned_arm": str(encouragement.get("assigned_arm", "disabled")),
        "treatment_probability": float(encouragement.get(
            "treatment_probability", 0.0,
        )),
        "assigned_focus": list(encouragement.get("assigned_focus") or []),
        "declared_focus": sorted(selected),
        "complied_with_assigned_focus": bool(
            assigned_trial
            and set(encouragement.get("assigned_focus") or []) == selected
        ),
        "useful_shadow_artifact": bool(any(
            row.get("useful_artifact")
            for row in compute_task_outcomes.values()
        )),
        "artifact_outcome_observed": assigned_trial,
        "outcome_scope": "post_action_shadow_reasoning_only",
        "can_change_current_action": False,
        "can_claim_match_outcome_causality": False,
    }
    short_circuited = sorted(
        task for task, (_, audit_key) in TASK_CONTRACTS.items()
        if (plan.get(audit_key) or {}).get("reason")
        == "task_not_selected_by_deliberation_focus"
    )
    unfocused_compute_isolated = not (
        set(unfocused) - set(short_circuited)
    )
    selected_priority = sum(
        float(agenda_tasks[task]["priority"])
        for task in selected if task in agenda_tasks
    )
    optimal = sorted((
        float(agenda_tasks[task]["priority"]) for task in eligible
    ), reverse=True)[:len(selected)]
    optimal_priority = sum(optimal)
    efficiency = (
        selected_priority / optimal_priority
        if optimal_priority > 1e-12 else 1.0
    )
    efficiency = float(np.clip(efficiency, 0.0, 1.0))
    selected_portfolio_tasks = sorted(selected & set(agenda_tasks))
    selected_portfolio_objective = (
        deliberation_portfolio_objective(
            selected_portfolio_tasks, agenda_tasks,
        ) if selected_portfolio_tasks else 0.0
    )
    optimal_portfolio = (
        agenda.get("recommended_portfolios_by_size") or {}
    ).get(str(len(selected))) or {}
    optimal_portfolio_objective = float(
        optimal_portfolio.get("objective", 0.0)
    )
    portfolio_efficiency = (
        selected_portfolio_objective / optimal_portfolio_objective
        if optimal_portfolio_objective > 1e-12 else 0.0
    )
    portfolio_efficiency = float(np.clip(
        portfolio_efficiency, 0.0, 1.0,
    ))
    recommended = list(agenda.get("recommended_focus") or [])
    payload = {
        "version": LLM_DELIBERATION_FOCUS_VERSION,
        "accepted": True,
        "reason": "model_checked_deliberation_focus",
        "focus": focus,
        "brief_digest": brief["brief_digest"],
        "source_packet_fingerprint": brief["source_packet_fingerprint"],
        "agenda_recommended_focus": recommended,
        "eligible_tasks": sorted(eligible),
        "emitted_optional_contracts": sorted(emitted),
        "accepted_optional_contracts": sorted(accepted),
        "unsupported_selected_tasks": unsupported,
        "unfocused_emitted_contracts": unfocused,
        "short_circuited_unfocused_contracts": sorted(
            set(unfocused) & set(short_circuited)
        ),
        "unfocused_compute_isolated": unfocused_compute_isolated,
        "compute_allocation": compute_allocation,
        "compute_allocation_valid": compute_allocation_valid,
        "compute_allocation_matches_model": (
            compute_allocation_matches_model
        ),
        "compute_budget_mismatches": sorted(compute_budget_mismatches),
        "compute_budget_respected": bool(
            compute_allocation_valid and compute_allocation_matches_model
            and not compute_budget_mismatches
        ),
        "compute_task_outcomes": compute_task_outcomes,
        "task_encouragement_context": encouragement_context,
        "missing_selected_contracts": missing,
        "rejected_selected_contracts": rejected,
        "experimental_budget_rejections": (
            experimental_budget_rejections
        ),
        "nonexperimental_rejected_selected_contracts": (
            nonexperimental_rejected
        ),
        "selected_priority_sum": selected_priority,
        "optimal_same_budget_priority_sum": optimal_priority,
        "focus_priority_efficiency": efficiency,
        "selected_portfolio_objective": selected_portfolio_objective,
        "optimal_same_size_portfolio_objective": (
            optimal_portfolio_objective
        ),
        "focus_portfolio_efficiency": portfolio_efficiency,
        "model_checked_consistent": bool(
            not unsupported and not unfocused and not missing
            and not nonexperimental_rejected
            and compute_allocation_valid and compute_allocation_matches_model
            and not compute_budget_mismatches
            and encouragement_valid
            and efficiency >= 0.80 - 1e-12
            and portfolio_efficiency >= 0.80 - 1e-12
        ),
        "focus_signature": str(focus_signature),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_relax_downstream_validators": False,
        "causal_interpretation": False,
    }
    return {**payload, "audit_digest": _digest(payload)}


def evaluate_llm_deliberation_focus(
    brief: dict[str, Any],
    plan: dict[str, Any],
    *,
    focus_signature: str = "llm-deliberation-focus-unspecified",
) -> dict[str, Any]:
    focus = validate_llm_deliberation_focus(
        plan.get("world_model_deliberation_focus")
    )
    base = {
        "version": LLM_DELIBERATION_FOCUS_VERSION,
        "accepted": False,
        "reason": "missing_or_invalid_deliberation_focus",
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_relax_downstream_validators": False,
        "causal_interpretation": False,
        "focus_signature": str(focus_signature),
    }
    if focus is None:
        return base
    audit = _audit_from_evidence(
        focus, brief, plan, focus_signature=focus_signature,
    )
    return audit or {**base, "reason": "compatible_deliberation_agenda_required"}


def llm_deliberation_focus_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    try:
        expected = _digest(payload)
    except (TypeError, ValueError, OverflowError):
        return False
    from src.match_engine.world_model.llm_deliberation_compute import (
        deliberation_compute_allocation_is_valid,
    )

    compute_valid = deliberation_compute_allocation_is_valid(
        payload.get("compute_allocation")
    )
    mismatches = payload.get("compute_budget_mismatches")
    outcomes = payload.get("compute_task_outcomes")
    encouragement_context = payload.get("task_encouragement_context")
    try:
        rejected = set(payload.get("rejected_selected_contracts") or [])
        experimental_rejected = set(
            payload.get("experimental_budget_rejections") or []
        )
        nonexperimental_rejected = set(
            payload.get("nonexperimental_rejected_selected_contracts") or []
        )
    except TypeError:
        return False
    return bool(
        digest == expected
        and payload.get("version") == LLM_DELIBERATION_FOCUS_VERSION
        and compute_valid == bool(payload.get("compute_allocation_valid"))
        and payload.get("compute_allocation_matches_model") is True
        and isinstance(mismatches, list)
        and isinstance(outcomes, dict)
        and isinstance(encouragement_context, dict)
        and encouragement_context.get("audit_valid") is True
        and encouragement_context.get("outcome_scope")
        == "post_action_shadow_reasoning_only"
        and (
            not encouragement_context.get("randomized_trial")
            or encouragement_context.get("two_stage_active") is True
        )
        and encouragement_context.get("can_change_current_action") is False
        and encouragement_context.get("can_claim_match_outcome_causality")
        is False
        and experimental_rejected <= rejected
        and experimental_rejected == {
            task for task, row in outcomes.items()
            if (
                row.get("budget_rejection")
                and row.get("would_fit_planned_budget")
                and (row.get("compute_value_experiment") or {}).get(
                    "randomized"
                )
                and (row.get("compute_value_experiment") or {}).get("arm")
                == "control"
            )
        }
        and nonexperimental_rejected == rejected - experimental_rejected
        and all(
            isinstance(row, dict)
            and row.get("outcome_scope")
            == "model_internal_useful_artifact"
            and row.get("can_claim_match_outcome_causality") is False
            for row in outcomes.values()
        )
        and payload.get("compute_budget_respected")
        is bool(compute_valid and not mismatches)
        and payload.get("can_change_current_action") is False
        and payload.get("can_relax_downstream_validators") is False
        and payload.get("authority_active") is False
        and payload.get("policy_mutated") is False
        and payload.get("causal_interpretation") is False
    )


def llm_deliberation_focus_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = consistent = malformed = 0
    short_circuited_unfocused = compute_budget_respected = 0
    all_unfocused_compute_isolated = True
    clustered_consistency = []
    clustered_efficiency = []
    clustered_portfolio_efficiency = []
    clustered_compute_value = []
    focus_counts: dict[str, int] = {}
    compute_credit_counts: dict[str, int] = {}
    allocated_compute_credits = 0
    signatures = set()
    opportunities = declarations = 0
    clustered_coverage = []
    for records in record_clusters:
        match_audits = []
        match_opportunities = match_declarations = 0
        for record in records:
            metadata = record.get("llm_decision_brief_context") or {}
            if llm_decision_brief_metadata_is_valid(metadata):
                opportunities += 1
                match_opportunities += 1
            audit = record.get("llm_deliberation_focus_context") or {}
            if not audit:
                continue
            declarations += 1
            match_declarations += 1
            if not llm_deliberation_focus_audit_is_valid(audit):
                malformed += 1
                continue
            accepted += 1
            consistent += bool(audit["model_checked_consistent"])
            compute_budget_respected += bool(
                audit.get("compute_budget_respected")
            )
            short_circuited_unfocused += len(
                audit.get("short_circuited_unfocused_contracts") or []
            )
            all_unfocused_compute_isolated = bool(
                all_unfocused_compute_isolated
                and audit.get("unfocused_compute_isolated") is True
            )
            signatures.add(str(audit.get("focus_signature", "")))
            match_audits.append(audit)
            allocation = audit.get("compute_allocation") or {}
            if allocation.get("accepted"):
                allocated_compute_credits += int(
                    allocation.get("allocated_compute_credits", 0)
                )
                for task, row in (
                    allocation.get("allocations") or {}
                ).items():
                    compute_credit_counts[task] = (
                        compute_credit_counts.get(task, 0)
                        + int(row.get("compute_credits", 0))
                    )
            for task in audit["focus"]["tasks"]:
                focus_counts[task] = focus_counts.get(task, 0) + 1
        if match_audits:
            clustered_consistency.append(float(np.mean([
                bool(audit["model_checked_consistent"])
                for audit in match_audits
            ])))
            clustered_efficiency.append(float(np.mean([
                float(audit["focus_priority_efficiency"])
                for audit in match_audits
            ])))
            clustered_portfolio_efficiency.append(float(np.mean([
                float(audit["focus_portfolio_efficiency"])
                for audit in match_audits
            ])))
            clustered_compute_value.append(float(np.mean([
                float((audit.get("compute_allocation") or {}).get(
                    "mean_priority_per_allocated_credit", 0.0,
                ))
                for audit in match_audits
            ])))
        if match_opportunities:
            clustered_coverage.append(min(
                1.0, match_declarations / match_opportunities,
            ))
    unspecified = {"", "llm-deliberation-focus-unspecified"}
    return {
        "version": LLM_DELIBERATION_FOCUS_VERSION,
        "accepted_focus_audits": accepted,
        "deliberation_focus_opportunities": opportunities,
        "deliberation_focus_declarations": declarations,
        "model_checked_consistent_focus_audits": consistent,
        "compute_budget_respected_focus_audits": compute_budget_respected,
        "malformed_focus_audits": malformed,
        "short_circuited_unfocused_contracts": (
            short_circuited_unfocused
        ),
        "all_unfocused_compute_isolated": bool(
            accepted > 0 and all_unfocused_compute_isolated
        ),
        "all_compute_budgets_respected": bool(
            accepted > 0 and compute_budget_respected == accepted
        ),
        "matches": len(clustered_consistency),
        "match_clustered_consistency_rate": float(np.mean(
            clustered_consistency
        )) if clustered_consistency else 0.0,
        "match_clustered_focus_priority_efficiency": float(np.mean(
            clustered_efficiency
        )) if clustered_efficiency else 0.0,
        "match_clustered_focus_portfolio_efficiency": float(np.mean(
            clustered_portfolio_efficiency
        )) if clustered_portfolio_efficiency else 0.0,
        "match_clustered_focus_declaration_rate": float(np.mean(
            clustered_coverage
        )) if clustered_coverage else 0.0,
        "selected_task_counts": dict(sorted(focus_counts.items())),
        "allocated_compute_credits": allocated_compute_credits,
        "compute_credit_counts_by_task": dict(sorted(
            compute_credit_counts.items()
        )),
        "match_clustered_mean_priority_per_compute_credit": float(np.mean(
            clustered_compute_value
        )) if clustered_compute_value else 0.0,
        "focus_signatures": sorted(signatures),
        "all_shadow_only_non_controlling": malformed == 0,
        "provenance_compatible": bool(
            len(signatures) == 1
            and next(iter(signatures), "") not in unspecified
        ),
    }
