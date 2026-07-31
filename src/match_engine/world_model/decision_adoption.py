"""Audit whether an LLM-selected world-model action appears after the decision."""

from __future__ import annotations

import hashlib
import math
from typing import Any


TRACKED_ACTIONS = {"hold", "pass", "cross", "shot"}


def _randomized_experiment_arm(
    decision_id: str,
    *,
    seed: int,
    control_rate: float,
) -> str:
    rate = min(0.5, max(0.0, float(control_rate)))
    if rate <= 0.0:
        return "treatment"
    digest = hashlib.sha256(
        f"policy-bridge-v1:{int(seed)}:{decision_id}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    return "control" if draw < rate else "treatment"


def register_coach_action_decision(
    state,
    *,
    team_id: str,
    trigger_kind: str,
    llm_selected_action: str,
    world_model_recommended_action: str,
    recommendation_confidence: float,
    horizon_s: float,
    intervention_strength: float = 0.0,
    intervention_enabled: bool = False,
    experiment_control_rate: float = 0.0,
) -> dict[str, Any] | None:
    """Register a prospective decision; never count an already-executed action."""
    selected = str(llm_selected_action).lower()
    if selected not in TRACKED_ACTIONS:
        return None
    records = getattr(state, "_wm_coach_decision_adoption", None)
    if records is None:
        records = []
        state._wm_coach_decision_adoption = records
    created = float(getattr(state, "clock_seconds", 0.0))
    horizon = max(0.1, float(horizon_s))
    for previous in records:
        if (
            not previous["resolved"]
            and str(previous["team_id"]) == str(team_id)
        ):
            previous["resolved"] = True
            previous["adopted"] = False
            previous["resolved_t_sec"] = created
            previous["resolution"] = "superseded_by_newer_coach_decision"
    strength = float(intervention_strength)
    if not math.isfinite(strength):
        strength = 0.0
    strength = min(0.5, max(0.0, strength))
    decision_id = f"{team_id}:{created:.3f}:{len(records)}"
    control_rate = min(0.5, max(0.0, float(experiment_control_rate)))
    bridge_eligible = bool(intervention_enabled and strength > 0.0)
    experiment_arm = (
        _randomized_experiment_arm(
            decision_id,
            seed=int(getattr(state, "_wm_policy_experiment_seed", 0)),
            control_rate=control_rate,
        )
        if bridge_eligible else "disabled"
    )
    record = {
        "decision_id": decision_id,
        "team_id": str(team_id),
        "trigger_kind": str(trigger_kind),
        "created_t_sec": created,
        "expires_t_sec": created + horizon,
        "llm_selected_action": selected,
        "world_model_recommended_action": str(
            world_model_recommended_action
        ).lower(),
        "recommendation_confidence": float(recommendation_confidence),
        "intervention_enabled": bridge_eligible,
        "intervention_strength": strength,
        "experiment_arm": experiment_arm,
        "experiment_control_rate": control_rate,
        "experiment_treatment_propensity": 1.0 - control_rate,
        "policy_opportunity_observed": False,
        "intervention_applied": False,
        "intervention_t_sec": None,
        "intervention_actual_action": None,
        "observed_actions": [],
        "resolved": False,
        "adopted": None,
        "resolved_t_sec": None,
        "resolution": "pending",
    }
    records.append(record)
    return record


def pending_policy_action_bias(
    state,
    *,
    team_id: str,
    feasible_actions: set[str],
    t_sec: float,
) -> dict[str, Any] | None:
    """Return the newest eligible one-shot logit bias without consuming it."""
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    for record in reversed(records):
        if record["resolved"] or record["intervention_applied"]:
            continue
        if not record.get("intervention_enabled", False):
            continue
        if str(record["team_id"]) != str(team_id):
            continue
        if now <= float(record["created_t_sec"]):
            continue
        if now > float(record["expires_t_sec"]):
            record["resolved"] = True
            record["adopted"] = False
            record["resolved_t_sec"] = now
            record["resolution"] = "horizon_expired"
            continue
        selected = str(record["llm_selected_action"])
        if selected not in feasible_actions:
            continue
        return {
            "decision_id": record["decision_id"],
            "action": selected,
            "logit_bias": (
                float(record["intervention_strength"])
                if record.get("experiment_arm") == "treatment" else 0.0
            ),
            "experiment_arm": record.get("experiment_arm", "treatment"),
        }
    return None


def record_policy_intervention_result(
    state,
    *,
    decision_id: str,
    actual_action: str,
    t_sec: float,
) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    for record in records:
        if record["decision_id"] != decision_id or record["resolved"]:
            continue
        actual = str(actual_action).lower()
        arm = str(record.get("experiment_arm", "treatment"))
        record["policy_opportunity_observed"] = True
        record["intervention_applied"] = arm == "treatment"
        record["intervention_t_sec"] = float(t_sec)
        record["intervention_actual_action"] = actual
        record["observed_actions"].append({
            "t_sec": float(t_sec), "action": actual,
        })
        record["resolved"] = True
        record["adopted"] = actual == record["llm_selected_action"]
        record["resolved_t_sec"] = float(t_sec)
        if arm == "control":
            record["resolution"] = (
                "matching_action_in_randomized_control"
                if record["adopted"]
                else "different_action_in_randomized_control"
            )
        else:
            record["resolution"] = (
                "matching_action_after_bounded_bias"
                if record["adopted"] else "different_action_after_bounded_bias"
            )
        return


def randomized_policy_effect_from_counts(
    *,
    treatment: int,
    treatment_adopted: int,
    control: int,
    control_adopted: int,
    min_per_arm: int = 2,
) -> dict[str, Any]:
    """Build a two-arm difference-in-proportions causal report."""
    treatment = max(0, int(treatment))
    control = max(0, int(control))
    treatment_adopted = min(treatment, max(0, int(treatment_adopted)))
    control_adopted = min(control, max(0, int(control_adopted)))
    treatment_rate = treatment_adopted / max(1, treatment)
    control_rate = control_adopted / max(1, control)
    effect = treatment_rate - control_rate
    # Agresti-Caffo add-two correction avoids a falsely zero standard error
    # when a small arm happens to have all successes or all failures.
    treatment_adjusted = (treatment_adopted + 1.0) / (treatment + 2.0)
    control_adjusted = (control_adopted + 1.0) / (control + 2.0)
    standard_error = math.sqrt(
        treatment_adjusted * (1.0 - treatment_adjusted) / (treatment + 2.0)
        + control_adjusted * (1.0 - control_adjusted) / (control + 2.0)
    )
    minimum = max(2, int(min_per_arm))
    ready = treatment >= minimum and control >= minimum
    return {
        "evaluation_kind": "randomized_zero_bias_policy_bridge",
        "opportunities": treatment + control,
        "treatment": treatment,
        "treatment_adopted": treatment_adopted,
        "treatment_adoption_rate": treatment_rate,
        "control": control,
        "control_adopted": control_adopted,
        "control_adoption_rate": control_rate,
        "average_treatment_effect": effect,
        "standard_error": standard_error,
        "confidence_interval": [
            max(-1.0, effect - 1.96 * standard_error),
            min(1.0, effect + 1.96 * standard_error),
        ],
        "minimum_per_arm": minimum,
        "ready": ready,
        "causal_interpretation": (
            "within_simulator_action_selection"
            if ready else "insufficient_randomized_samples"
        ),
    }


def randomized_policy_effect(
    records: list[dict[str, Any]],
    *,
    min_per_arm: int = 2,
) -> dict[str, Any]:
    """Estimate the randomized bridge effect on selected-action adoption."""
    opportunities = [
        record for record in records
        if record.get("policy_opportunity_observed")
        and record.get("experiment_arm") in {"treatment", "control"}
        and record.get("adopted") is not None
    ]
    treatment = [
        record for record in opportunities
        if record["experiment_arm"] == "treatment"
    ]
    control = [
        record for record in opportunities
        if record["experiment_arm"] == "control"
    ]
    return randomized_policy_effect_from_counts(
        treatment=len(treatment),
        treatment_adopted=sum(bool(record["adopted"]) for record in treatment),
        control=len(control),
        control_adopted=sum(bool(record["adopted"]) for record in control),
        min_per_arm=min_per_arm,
    )


def policy_bridge_reliability_factor(
    state,
    *,
    min_per_arm: int = 2,
) -> float:
    """Conservatively reduce future bias only after adverse randomized evidence."""
    records = list(getattr(state, "_wm_coach_decision_adoption", None) or [])
    effect = randomized_policy_effect(records, min_per_arm=min_per_arm)
    if not effect["ready"]:
        return 1.0
    low, high = effect["confidence_interval"]
    if high <= 0.0:
        return 0.25
    if effect["average_treatment_effect"] <= 0.0:
        return 0.60
    if low > 0.0:
        return 1.0
    return 0.85


def observe_executed_action(
    state,
    *,
    team_id: str,
    action_kind: str,
    t_sec: float,
) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    action = str(action_kind).lower()
    for record in records:
        if record["resolved"]:
            continue
        # Decisions are registered after the current tick's action; strict
        # ordering prevents that earlier action from being counted as adoption.
        if now <= float(record["created_t_sec"]):
            continue
        if now > float(record["expires_t_sec"]):
            record["resolved"] = True
            record["adopted"] = False
            record["resolved_t_sec"] = now
            record["resolution"] = "horizon_expired"
            continue
        if str(record["team_id"]) != str(team_id):
            continue
        record["observed_actions"].append({"t_sec": now, "action": action})
        if action == record["llm_selected_action"]:
            record["resolved"] = True
            record["adopted"] = True
            record["resolved_t_sec"] = now
            record["resolution"] = "matching_action_observed"


def finalize_decision_adoption(state, *, t_sec: float) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    for record in records:
        if not record["resolved"]:
            record["resolved"] = True
            record["adopted"] = False
            record["resolved_t_sec"] = now
            record["resolution"] = "match_ended"


def decision_adoption_diagnostics(state) -> dict[str, Any]:
    records = list(getattr(state, "_wm_coach_decision_adoption", None) or [])
    resolved = [record for record in records if record["resolved"]]
    adopted = [record for record in resolved if record["adopted"]]
    interventions = [
        record for record in resolved if record.get("intervention_applied")
    ]
    intervention_adopted = [
        record for record in interventions if record["adopted"]
    ]
    randomized_effect = randomized_policy_effect(records)
    return {
        "version": 2,
        "registered": len(records),
        "resolved": len(resolved),
        "adopted": len(adopted),
        "pending": len(records) - len(resolved),
        "adoption_rate": len(adopted) / max(1, len(resolved)),
        "interventions_applied": len(interventions),
        "intervention_adopted": len(intervention_adopted),
        "intervention_adoption_rate": (
            len(intervention_adopted) / max(1, len(interventions))
        ),
        "causal_ordering": "decision_then_future_team_action",
        "randomized_policy_effect": randomized_effect,
        "records": records,
        "interpretation": (
            "A bounded intervention changes one action logit but does not force "
            "the sampled action; non-intervention adoption remains associative "
            "and does not prove causal influence."
        ),
    }
