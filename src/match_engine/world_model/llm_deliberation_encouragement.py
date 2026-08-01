"""Safe post-action randomized encouragement between shadow reasoning tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from src.match_engine.world_model.llm_decision_brief import (
    deliberation_portfolio_objective,
    llm_decision_brief_self_is_valid,
)


DELIBERATION_ENCOURAGEMENT_VERSION = 1
SHADOW_SWAP_TASKS = {
    "world_model_event_option",
    "world_model_contrastive_claim",
}


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "llm-deliberation-encouragement:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _arm(brief_digest: str) -> str:
    digest = hashlib.sha256(
        f"{brief_digest}|shadow-task-swap-v1".encode("utf-8")
    ).digest()
    return "treatment" if digest[0] < 128 else "control"


def build_shadow_deliberation_brief(
    brief: dict[str, Any],
) -> dict[str, Any]:
    """Create a prompt-only brief; the authoritative audit brief is unchanged."""
    if not llm_decision_brief_self_is_valid(brief):
        return {}
    output = copy.deepcopy(brief)
    agenda = output["deliberation_agenda"]
    task_rows = {
        str(row["task"]): row for row in agenda["tasks"]
    }
    base = list(agenda.get("recommended_focus") or [])
    base_set = set(base)
    eligible_swap = all(
        (task_rows.get(task) or {}).get("eligible")
        for task in SHADOW_SWAP_TASKS
    )
    exactly_one = len(base_set & SHADOW_SWAP_TASKS) == 1
    alternative = sorted(
        (base_set - SHADOW_SWAP_TASKS)
        | (SHADOW_SWAP_TASKS - base_set)
    ) if exactly_one else []
    eligible = [
        row for row in task_rows.values() if row.get("eligible")
    ]
    size = len(base)
    maximum_raw = sum(sorted((
        float(row["priority"]) for row in eligible
    ), reverse=True)[:size])
    alternative_raw = sum(
        float(task_rows[task]["priority"]) for task in alternative
    ) if alternative else 0.0
    alternative_compute = sum(
        int(task_rows[task]["expected_compute_credits"])
        for task in alternative
    ) if alternative else 0
    alternative_objective = (
        deliberation_portfolio_objective(alternative, task_rows)
        if alternative else 0.0
    )
    base_objective = float((
        agenda.get("recommended_portfolios_by_size") or {}
    ).get(str(size), {}).get("objective", 0.0))
    evidence_safe = bool(
        eligible_swap and exactly_one and len(alternative) == size
        and alternative_compute <= 6
        and alternative_raw >= 0.80 * maximum_raw - 1e-12
        and alternative_objective >= 0.80 * base_objective - 1e-12
    )
    base_has_treatment = "world_model_event_option" in base_set
    treatment = base if base_has_treatment else alternative
    control = alternative if base_has_treatment else base
    assigned_arm = _arm(str(brief["brief_digest"]))
    assigned = (
        treatment if assigned_arm == "treatment" else control
    ) if evidence_safe else base
    experiment = {
        "version": DELIBERATION_ENCOURAGEMENT_VERSION,
        "eligible": evidence_safe,
        "reason": (
            "evidence_safe_post_action_shadow_task_swap"
            if evidence_safe else "compatible_shadow_swap_unavailable"
        ),
        "target_task": "world_model_event_option",
        "control_task": "world_model_contrastive_claim",
        "treatment_portfolio": list(treatment) if evidence_safe else [],
        "control_portfolio": list(control) if evidence_safe else [],
        "assigned_arm": assigned_arm if evidence_safe else "disabled",
        "treatment_probability": 0.5 if evidence_safe else 0.0,
        "assigned_focus": list(assigned),
        "base_focus": base,
        "action_frozen_before_assignment": True,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_relax_downstream_validators": False,
        "outcome_scope": "post_action_shadow_reasoning_only",
        "can_claim_match_outcome_causality": False,
    }
    agenda["recommended_focus"] = list(assigned)
    if str(size) in agenda["recommended_portfolios_by_size"]:
        portfolio = agenda["recommended_portfolios_by_size"][str(size)]
        portfolio["tasks"] = list(assigned)
        portfolio["objective"] = deliberation_portfolio_objective(
            assigned, task_rows,
        )
        portfolio["raw_priority_sum"] = sum(
            float(task_rows[task]["priority"]) for task in assigned
        )
        portfolio["expected_compute_credits"] = sum(
            int(task_rows[task]["expected_compute_credits"])
            for task in assigned
        )
        portfolio["reasoning_domains"] = sorted({
            str(task_rows[task]["reasoning_domain"]) for task in assigned
        })
    output["shadow_deliberation_phase"] = True
    output["shadow_task_encouragement"] = experiment
    payload = dict(output)
    payload.pop("shadow_brief_digest", None)
    output["shadow_brief_digest"] = _digest(payload)
    return output


def shadow_deliberation_brief_is_valid(
    shadow: Any, authoritative_brief: dict[str, Any],
) -> bool:
    if not isinstance(shadow, dict):
        return False
    try:
        return shadow == build_shadow_deliberation_brief(authoritative_brief)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def deliberation_encouragement_audit_is_valid(
    audit: Any,
    plan: dict[str, Any],
    authoritative_brief: dict[str, Any],
) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    shadow = build_shadow_deliberation_brief(authoritative_brief)
    focus = plan.get("world_model_deliberation_focus") or {}
    return bool(
        shadow_deliberation_brief_is_valid(shadow, authoritative_brief)
        and audit.get("version") == DELIBERATION_ENCOURAGEMENT_VERSION
        and audit.get("two_stage_active") is True
        and audit.get("encouragement")
        == shadow.get("shadow_task_encouragement")
        and audit.get("shadow_brief_digest")
        == shadow.get("shadow_brief_digest")
        and audit.get("declared_focus_tasks") == list(focus.get("tasks") or [])
        and str(audit.get("action_frozen"))
        == str(plan.get("world_model_action", "none"))
        and audit.get("action_preserved") is True
        and audit.get("tactical_controls_preserved") is True
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_relax_downstream_validators") is False
    )


def _posterior(successes: int, samples: int) -> tuple[float, float]:
    alpha = successes + 1.0
    beta = samples - successes + 1.0
    mean = alpha / (alpha + beta)
    variance = alpha * beta / (
        (alpha + beta) ** 2 * (alpha + beta + 1.0)
    )
    return mean, variance


def _task_selection_rows(records) -> tuple[dict[str, dict[str, int]], dict]:
    arms = {
        "control": {"trials": 0, "complied": 0, "useful": 0},
        "treatment": {"trials": 0, "complied": 0, "useful": 0},
    }
    excluded = {"invalid_focus_audit": 0, "not_randomized": 0}
    for record in records:
        audit = record.get("llm_deliberation_focus_context") or {}
        if not audit:
            excluded["not_randomized"] += 1
            continue
        from src.match_engine.world_model.llm_deliberation_focus import (
            llm_deliberation_focus_audit_is_valid,
        )

        if not llm_deliberation_focus_audit_is_valid(audit):
            excluded["invalid_focus_audit"] += 1
            continue
        context = audit.get("task_encouragement_context") or {}
        arm = str(context.get("assigned_arm", ""))
        if not context.get("randomized_trial") or arm not in arms:
            excluded["not_randomized"] += 1
            continue
        arms[arm]["trials"] += 1
        arms[arm]["complied"] += int(bool(
            context.get("complied_with_assigned_focus")
        ))
        arms[arm]["useful"] += int(bool(
            context.get("useful_shadow_artifact")
        ))
    return arms, excluded


def build_task_selection_value_memory(
    records,
    *,
    checkpoint_signature: str,
    environment_signature: str,
    as_of_t_sec: float,
    min_per_arm: int = 4,
) -> dict[str, Any]:
    prior = []
    excluded_time = excluded_provenance = 0
    for record in records:
        try:
            created = float(record.get("created_t_sec", math.inf))
        except (TypeError, ValueError, OverflowError):
            created = math.inf
        if not math.isfinite(created) or created >= as_of_t_sec - 1e-9:
            excluded_time += 1
            continue
        if (
            str(record.get("checkpoint_signature"))
            != str(checkpoint_signature)
            or str(record.get("environment_signature"))
            != str(environment_signature)
        ):
            excluded_provenance += 1
            continue
        prior.append(record)
    arms, excluded = _task_selection_rows(prior)
    control_mean, control_var = _posterior(
        arms["control"]["useful"], arms["control"]["trials"],
    )
    treatment_mean, treatment_var = _posterior(
        arms["treatment"]["useful"], arms["treatment"]["trials"],
    )
    effect = treatment_mean - control_mean
    radius = 1.645 * math.sqrt(control_var + treatment_var)
    lower = max(-1.0, effect - radius)
    upper = min(1.0, effect + radius)
    threshold = max(2, int(min_per_arm))
    enough = all(arms[arm]["trials"] >= threshold for arm in arms)
    status = "insufficient_randomized_encouragement"
    if enough and lower > 0.0:
        status = "validated_event_option_encouragement_value"
    elif enough and upper < 0.0:
        status = "validated_contrastive_encouragement_value"
    elif enough:
        status = "encouragement_effect_uncertain"
    payload = {
        "version": DELIBERATION_ENCOURAGEMENT_VERSION,
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "as_of_t_sec": float(as_of_t_sec),
        "status": status,
        "arms": arms,
        "event_option_minus_contrastive_itt": effect,
        "itt_lower_90": lower,
        "itt_upper_90": upper,
        "minimum_samples_per_arm": threshold,
        "excluded_records": {
            **excluded,
            "future_or_current": excluded_time,
            "incompatible_provenance": excluded_provenance,
        },
        "target_task": "world_model_event_option",
        "control_task": "world_model_contrastive_claim",
        "outcome_scope": "post_action_shadow_reasoning_only",
        "intention_to_treat": True,
        "can_change_current_action": False,
        "can_claim_match_outcome_causality": False,
    }
    return {**payload, "memory_digest": _digest(payload)}


def task_selection_value_memory_is_valid(memory: Any) -> bool:
    if not isinstance(memory, dict):
        return False
    payload = dict(memory)
    digest = payload.pop("memory_digest", None)
    try:
        expected = _digest(payload)
        arms = payload["arms"]
        effect = float(payload["event_option_minus_contrastive_itt"])
        lower = float(payload["itt_lower_90"])
        upper = float(payload["itt_upper_90"])
        threshold = int(payload["minimum_samples_per_arm"])
        as_of = float(payload["as_of_t_sec"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if set(arms) != {"control", "treatment"}:
        return False
    parsed = {}
    for arm, row in arms.items():
        try:
            trials = int(row["trials"])
            complied = int(row["complied"])
            useful = int(row["useful"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not 0 <= complied <= trials or not 0 <= useful <= trials:
            return False
        parsed[arm] = (trials, useful)
    control_mean, control_var = _posterior(
        parsed["control"][1], parsed["control"][0],
    )
    treatment_mean, treatment_var = _posterior(
        parsed["treatment"][1], parsed["treatment"][0],
    )
    expected_effect = treatment_mean - control_mean
    radius = 1.645 * math.sqrt(control_var + treatment_var)
    expected_lower = max(-1.0, expected_effect - radius)
    expected_upper = min(1.0, expected_effect + radius)
    enough = all(parsed[arm][0] >= threshold for arm in parsed)
    expected_status = "insufficient_randomized_encouragement"
    if enough and expected_lower > 0.0:
        expected_status = "validated_event_option_encouragement_value"
    elif enough and expected_upper < 0.0:
        expected_status = "validated_contrastive_encouragement_value"
    elif enough:
        expected_status = "encouragement_effect_uncertain"
    return bool(
        digest == expected
        and payload.get("version") == DELIBERATION_ENCOURAGEMENT_VERSION
        and threshold >= 2
        and math.isfinite(as_of)
        and all(math.isfinite(value) for value in (effect, lower, upper))
        and lower <= effect <= upper
        and abs(effect - expected_effect) <= 1e-12
        and abs(lower - expected_lower) <= 1e-12
        and abs(upper - expected_upper) <= 1e-12
        and payload.get("status") == expected_status
        and payload.get("outcome_scope")
        == "post_action_shadow_reasoning_only"
        and payload.get("intention_to_treat") is True
        and payload.get("can_change_current_action") is False
        and payload.get("can_claim_match_outcome_causality") is False
    )


def task_encouragement_diagnostics(record_clusters) -> dict[str, Any]:
    records = []
    matches = 0
    for cluster in record_clusters:
        cluster = list(cluster)
        records.extend(cluster)
        cluster_arms, _ = _task_selection_rows(cluster)
        if sum(row["trials"] for row in cluster_arms.values()) > 0:
            matches += 1
    arms, excluded = _task_selection_rows(records)
    randomized = sum(row["trials"] for row in arms.values())
    balanced = all(row["trials"] >= 4 for row in arms.values())
    return {
        "version": DELIBERATION_ENCOURAGEMENT_VERSION,
        "randomized_encouragement_trials": randomized,
        "matches": matches,
        "arms": arms,
        "balanced_randomized_evidence": balanced,
        "overall_compliance_rate": (
            sum(row["complied"] for row in arms.values()) / randomized
            if randomized else 0.0
        ),
        "excluded_records": excluded,
        "malformed_focus_audits": excluded["invalid_focus_audit"],
        "all_trials_post_action_shadow_only": randomized > 0,
        "can_change_current_action": False,
        "can_claim_match_outcome_causality": False,
    }
