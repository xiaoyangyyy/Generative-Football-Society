"""Audit whether the coach LLM follows the world-model deliberation agenda."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.llm_decision_brief import (
    llm_decision_brief_metadata_is_valid,
    llm_decision_brief_self_is_valid,
)


LLM_DELIBERATION_FOCUS_VERSION = 1
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
        "missing_selected_contracts": missing,
        "rejected_selected_contracts": rejected,
        "selected_priority_sum": selected_priority,
        "optimal_same_budget_priority_sum": optimal_priority,
        "focus_priority_efficiency": efficiency,
        "model_checked_consistent": bool(
            not unsupported and not unfocused and not missing and not rejected
            and efficiency >= 0.80 - 1e-12
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
    return bool(
        digest == expected
        and payload.get("version") == LLM_DELIBERATION_FOCUS_VERSION
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
    clustered_consistency = []
    clustered_efficiency = []
    focus_counts: dict[str, int] = {}
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
            signatures.add(str(audit.get("focus_signature", "")))
            match_audits.append(audit)
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
        "malformed_focus_audits": malformed,
        "matches": len(clustered_consistency),
        "match_clustered_consistency_rate": float(np.mean(
            clustered_consistency
        )) if clustered_consistency else 0.0,
        "match_clustered_focus_priority_efficiency": float(np.mean(
            clustered_efficiency
        )) if clustered_efficiency else 0.0,
        "match_clustered_focus_declaration_rate": float(np.mean(
            clustered_coverage
        )) if clustered_coverage else 0.0,
        "selected_task_counts": dict(sorted(focus_counts.items())),
        "focus_signatures": sorted(signatures),
        "all_shadow_only_non_controlling": malformed == 0,
        "provenance_compatible": bool(
            len(signatures) == 1
            and next(iter(signatures), "") not in unspecified
        ),
    }
