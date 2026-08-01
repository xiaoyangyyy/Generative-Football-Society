"""Learn whether extra shadow deliberation compute produces useful evidence."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


DELIBERATION_COMPUTE_VALUE_VERSION = 1
TRAJECTORY_TASKS = {
    "world_model_event_option",
    "world_model_contrastive_claim",
}


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "llm-deliberation-compute-value:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def compute_value_experiment_arm(brief_digest: str, task: str) -> str:
    digest = hashlib.sha256(
        f"{brief_digest}|{task}|compute-value-v1".encode("utf-8")
    ).digest()
    return "treatment" if digest[0] < 128 else "control"


def _posterior(successes: int, samples: int) -> tuple[float, float]:
    alpha = successes + 1.0
    beta = samples - successes + 1.0
    mean = alpha / (alpha + beta)
    variance = alpha * beta / (
        (alpha + beta) ** 2 * (alpha + beta + 1.0)
    )
    return mean, variance


def build_deliberation_compute_value_memory(
    records: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
    as_of_t_sec: float,
    min_per_arm: int = 4,
) -> dict[str, Any]:
    """Estimate randomized marginal compute yield from strictly prior records."""
    groups = {
        task: {
            "control": {"samples": 0, "useful_artifacts": 0},
            "treatment": {"samples": 0, "useful_artifacts": 0},
        }
        for task in sorted(TRAJECTORY_TASKS)
    }
    excluded = {
        "future_or_current": 0,
        "incompatible_provenance": 0,
        "invalid_focus_audit": 0,
        "not_randomized_or_inconclusive": 0,
    }
    for record in records:
        try:
            created = float(record.get("created_t_sec", math.inf))
        except (TypeError, ValueError, OverflowError):
            created = math.inf
        if not math.isfinite(created) or created >= as_of_t_sec - 1e-9:
            excluded["future_or_current"] += 1
            continue
        if (
            str(record.get("checkpoint_signature"))
            != str(checkpoint_signature)
            or str(record.get("environment_signature"))
            != str(environment_signature)
        ):
            excluded["incompatible_provenance"] += 1
            continue
        audit = record.get("llm_deliberation_focus_context") or {}
        from src.match_engine.world_model.llm_deliberation_focus import (
            llm_deliberation_focus_audit_is_valid,
        )

        if not llm_deliberation_focus_audit_is_valid(audit):
            excluded["invalid_focus_audit"] += 1
            continue
        allocation = audit.get("compute_allocation") or {}
        outcomes = audit.get("compute_task_outcomes") or {}
        used = False
        for task in TRAJECTORY_TASKS:
            row = (allocation.get("allocations") or {}).get(task) or {}
            experiment = row.get("compute_value_experiment") or {}
            outcome = outcomes.get(task) or {}
            arm = str(experiment.get("arm", ""))
            if (
                not experiment.get("randomized")
                or arm not in {"control", "treatment"}
                or not outcome.get("conclusive")
            ):
                continue
            groups[task][arm]["samples"] += 1
            groups[task][arm]["useful_artifacts"] += int(bool(
                outcome.get("useful_artifact")
            ))
            used = True
        if not used:
            excluded["not_randomized_or_inconclusive"] += 1

    task_values = {}
    threshold = max(2, int(min_per_arm))
    for task, arms in groups.items():
        control = arms["control"]
        treatment = arms["treatment"]
        control_mean, control_var = _posterior(
            control["useful_artifacts"], control["samples"],
        )
        treatment_mean, treatment_var = _posterior(
            treatment["useful_artifacts"], treatment["samples"],
        )
        effect = treatment_mean - control_mean
        radius = 1.645 * math.sqrt(control_var + treatment_var)
        lower = max(-1.0, effect - radius)
        upper = min(1.0, effect + radius)
        enough = bool(
            control["samples"] >= threshold
            and treatment["samples"] >= threshold
        )
        status = "insufficient_randomized_evidence"
        if enough and lower > 0.0:
            status = "validated_positive_compute_value"
        elif enough and upper <= 0.0:
            status = "validated_no_compute_value"
        elif enough:
            status = "randomized_effect_uncertain"
        task_values[task] = {
            "status": status,
            "control": {
                **control, "posterior_useful_rate": control_mean,
            },
            "treatment": {
                **treatment, "posterior_useful_rate": treatment_mean,
            },
            "posterior_marginal_value": effect,
            "marginal_value_lower_90": lower,
            "marginal_value_upper_90": upper,
            "minimum_samples_per_arm": threshold,
            "randomized_assignment": True,
            "outcome_scope": "model_internal_useful_artifact",
            "can_claim_match_outcome_causality": False,
        }
    payload = {
        "version": DELIBERATION_COMPUTE_VALUE_VERSION,
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "as_of_t_sec": float(as_of_t_sec),
        "task_values": task_values,
        "excluded_records": excluded,
        "learning_policy": (
            "randomized_last_credit_model_internal_yield_only"
        ),
        "can_change_current_action": False,
        "can_relax_downstream_validators": False,
        "can_claim_match_outcome_causality": False,
    }
    return {**payload, "memory_digest": _digest(payload)}


def deliberation_compute_value_memory_is_valid(memory: Any) -> bool:
    if not isinstance(memory, dict):
        return False
    payload = dict(memory)
    digest = payload.pop("memory_digest", None)
    try:
        expected = _digest(payload)
        values = payload["task_values"]
        as_of = float(payload["as_of_t_sec"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if set(values) != TRAJECTORY_TASKS:
        return False
    for row in values.values():
        try:
            control_samples = int(row["control"]["samples"])
            treatment_samples = int(row["treatment"]["samples"])
            control_useful = int(row["control"]["useful_artifacts"])
            treatment_useful = int(row["treatment"]["useful_artifacts"])
            reported_control_mean = float(
                row["control"]["posterior_useful_rate"]
            )
            reported_treatment_mean = float(
                row["treatment"]["posterior_useful_rate"]
            )
            threshold = int(row["minimum_samples_per_arm"])
            bounds = (
                float(row["marginal_value_lower_90"]),
                float(row["posterior_marginal_value"]),
                float(row["marginal_value_upper_90"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        control_mean, control_var = _posterior(
            control_useful, control_samples,
        )
        treatment_mean, treatment_var = _posterior(
            treatment_useful, treatment_samples,
        )
        effect = treatment_mean - control_mean
        radius = 1.645 * math.sqrt(control_var + treatment_var)
        expected_bounds = (
            max(-1.0, effect - radius), effect,
            min(1.0, effect + radius),
        )
        enough = (
            control_samples >= threshold
            and treatment_samples >= threshold
        )
        expected_status = "insufficient_randomized_evidence"
        if enough and expected_bounds[0] > 0.0:
            expected_status = "validated_positive_compute_value"
        elif enough and expected_bounds[2] <= 0.0:
            expected_status = "validated_no_compute_value"
        elif enough:
            expected_status = "randomized_effect_uncertain"
        if (
            control_samples < 0 or treatment_samples < 0
            or not 0 <= control_useful <= control_samples
            or not 0 <= treatment_useful <= treatment_samples
            or threshold < 2
            or not all(math.isfinite(value) for value in bounds)
            or not bounds[0] <= bounds[1] <= bounds[2]
            or any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(bounds, expected_bounds)
            )
            or abs(reported_control_mean - control_mean) > 1e-12
            or abs(reported_treatment_mean - treatment_mean) > 1e-12
            or row.get("status") != expected_status
            or row.get("randomized_assignment") is not True
            or row.get("outcome_scope")
            != "model_internal_useful_artifact"
            or row.get("can_claim_match_outcome_causality") is not False
        ):
            return False
    return bool(
        digest == expected
        and payload.get("version") == DELIBERATION_COMPUTE_VALUE_VERSION
        and math.isfinite(as_of)
        and payload.get("learning_policy")
        == "randomized_last_credit_model_internal_yield_only"
        and payload.get("can_change_current_action") is False
        and payload.get("can_relax_downstream_validators") is False
        and payload.get("can_claim_match_outcome_causality") is False
    )


def deliberation_compute_value_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    arms = {
        task: {
            "control": {"trials": 0, "conclusive": 0, "useful": 0},
            "treatment": {"trials": 0, "conclusive": 0, "useful": 0},
        }
        for task in sorted(TRAJECTORY_TASKS)
    }
    malformed = learned_exploit = learned_conserve = 0
    matches = set()
    all_noncausal = True
    for match_index, records in enumerate(record_clusters):
        for record in records:
            audit = record.get("llm_deliberation_focus_context") or {}
            if not audit:
                continue
            from src.match_engine.world_model.llm_deliberation_focus import (
                llm_deliberation_focus_audit_is_valid,
            )

            if not llm_deliberation_focus_audit_is_valid(audit):
                malformed += 1
                continue
            allocation = audit.get("compute_allocation") or {}
            outcomes = audit.get("compute_task_outcomes") or {}
            for task in TRAJECTORY_TASKS:
                row = (allocation.get("allocations") or {}).get(task) or {}
                experiment = row.get("compute_value_experiment") or {}
                arm = str(experiment.get("arm", ""))
                if arm == "learned_exploit":
                    learned_exploit += 1
                elif arm == "learned_conserve":
                    learned_conserve += 1
                if not experiment.get("randomized") or arm not in {
                    "control", "treatment",
                }:
                    continue
                matches.add(match_index)
                arms[task][arm]["trials"] += 1
                outcome = outcomes.get(task) or {}
                all_noncausal = bool(
                    all_noncausal
                    and outcome.get("can_claim_match_outcome_causality")
                    is False
                )
                if outcome.get("conclusive"):
                    arms[task][arm]["conclusive"] += 1
                    arms[task][arm]["useful"] += int(bool(
                        outcome.get("useful_artifact")
                    ))
    tasks_with_balanced_evidence = sum(
        all(arms[task][arm]["conclusive"] >= 4 for arm in (
            "control", "treatment",
        ))
        for task in TRAJECTORY_TASKS
    )
    randomized_trials = sum(
        row["trials"]
        for task in arms.values() for row in task.values()
    )
    conclusive_trials = sum(
        row["conclusive"]
        for task in arms.values() for row in task.values()
    )
    return {
        "version": DELIBERATION_COMPUTE_VALUE_VERSION,
        "randomized_trials": randomized_trials,
        "conclusive_randomized_trials": conclusive_trials,
        "conclusive_trial_rate": (
            conclusive_trials / randomized_trials
            if randomized_trials else 0.0
        ),
        "matches_with_randomized_trials": len(matches),
        "task_arm_counts": arms,
        "tasks_with_balanced_randomized_evidence": (
            tasks_with_balanced_evidence
        ),
        "learned_exploit_allocations": learned_exploit,
        "learned_conserve_allocations": learned_conserve,
        "malformed_focus_audits": malformed,
        "all_outcomes_model_internal_and_noncausal": bool(
            randomized_trials > 0 and all_noncausal
        ),
        "allocation_can_change_current_action": False,
        "allocation_can_relax_downstream_validators": False,
    }
