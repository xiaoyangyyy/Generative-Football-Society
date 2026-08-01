"""Audit whether an LLM-selected world-model action appears after the decision."""

from __future__ import annotations

import math
from typing import Any

from src.match_engine.world_model.policy_experiment import (
    randomized_adoption_effect,
    randomized_experiment_arm,
    randomized_multi_horizon_effects,
    randomized_outcome_effect,
)
from src.match_engine.world_model.policy_outcomes import (
    censor_overlapping_policy_outcomes,
    normalize_outcome_horizons,
)
from src.match_engine.world_model.outcome_calibration import (
    policy_outcome_calibration,
)


TRACKED_ACTIONS = {"hold", "pass", "cross", "shot"}


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
    outcome_horizons_s: tuple[float, ...] = (0.0, 60.0, 180.0),
    action_predictions: dict[str, dict[str, Any]] | None = None,
    outcome_prediction_trust_factor: float = 1.0,
    checkpoint_signature: str = "runtime_unspecified",
    environment_signature: str = "environment_unspecified",
    active_learning: dict[str, Any] | None = None,
    opponent_belief_context: dict[str, Any] | None = None,
    opponent_response_context: dict[str, Any] | None = None,
    llm_world_model_critique_context: dict[str, Any] | None = None,
    llm_semantic_event_context: dict[str, Any] | None = None,
    llm_event_option_context: dict[str, Any] | None = None,
    llm_contrastive_explanation_context: dict[str, Any] | None = None,
    llm_contrastive_repair_context: dict[str, Any] | None = None,
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
    normalized_outcome_horizons = normalize_outcome_horizons(
        outcome_horizons_s,
    )
    censor_overlapping_policy_outcomes(
        records, team_id=team_id, t_sec=created,
    )
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
        randomized_experiment_arm(
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
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "policy_utility_version": 1,
        "active_learning": dict(active_learning or {}),
        "opponent_belief_context": dict(opponent_belief_context or {}),
        "opponent_response_context": dict(opponent_response_context or {}),
        "llm_world_model_critique_context": dict(
            llm_world_model_critique_context or {}
        ),
        "llm_semantic_event_context": dict(
            llm_semantic_event_context or {}
        ),
        "llm_event_option_context": dict(llm_event_option_context or {}),
        "llm_contrastive_explanation_context": dict(
            llm_contrastive_explanation_context or {}
        ),
        "llm_contrastive_repair_context": dict(
            llm_contrastive_repair_context or {}
        ),
        "event_option_resolved_t_sec": None,
        "event_option_expected_action": None,
        "event_option_next_action": None,
        "event_option_followup_expired": False,
        "event_option_followup_anchor_t_sec": None,
        "event_option_followup_due_t_sec": None,
        "event_option_followup_baseline": None,
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
        "outcome_horizons_s": list(normalized_outcome_horizons),
        "outcome_censored_t_sec": None,
        "outcome_censor_reason": None,
        "action_outcome_predictions": action_predictions or {},
        "outcome_prediction_trust_factor": min(
            1.0, max(0.25, float(outcome_prediction_trust_factor)),
        ),
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
    outcome_baseline: dict[str, float] | None = None,
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
        record["world_model_outcome_predictions"] = (
            record.get("action_outcome_predictions", {}).get(actual, {})
        )
        record["outcome_baseline"] = outcome_baseline
        record["short_horizon_outcome"] = None
        record["multi_horizon_outcomes"] = {}
        record["multi_horizon_regime_outcomes"] = {}
        record["outcome_anchor_t_sec"] = float(t_sec)
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


def observe_executed_action(
    state,
    *,
    team_id: str,
    action_kind: str,
    t_sec: float,
    outcome_baseline: dict[str, float] | None = None,
) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    action = str(action_kind).lower()
    for record in records:
        option_resolved_t = record.get("event_option_resolved_t_sec")
        if (
            option_resolved_t is not None
            and record.get("event_option_expected_action")
            and record.get("event_option_next_action") is None
            and not record.get("event_option_followup_expired")
            and str(record.get("team_id")) == str(team_id)
            and now > float(option_resolved_t)
        ):
            if now > float(option_resolved_t) + 120.0:
                record["event_option_followup_expired"] = True
            else:
                expected = str(record["event_option_expected_action"])
                record["event_option_next_action"] = action
                for outcome in (
                    record.get("multi_horizon_regime_outcomes") or {}
                ).values():
                    evaluation = (
                        outcome.get("llm_event_option_evaluation")
                        if isinstance(outcome, dict) else None
                    )
                    if isinstance(evaluation, dict):
                        evaluation["next_action_observed"] = True
                        evaluation["observed_continuation_action"] = action
                        evaluation["expected_action_matched"] = (
                            action == expected
                        )
                        evaluation["continuation_observed_t_sec"] = now
                        calibratable = bool(evaluation.get(
                            "continuation_value_calibratable"
                        ))
                        if action == expected and calibratable:
                            baseline = dict(outcome_baseline or {})
                            if baseline:
                                horizon = max(0.1, min(60.0, float(
                                    evaluation.get(
                                        "continuation_prediction_horizon_s",
                                        10.0,
                                    )
                                )))
                                record["event_option_followup_anchor_t_sec"] = now
                                record["event_option_followup_due_t_sec"] = (
                                    now + horizon
                                )
                                record["event_option_followup_baseline"] = baseline
                                evaluation[
                                    "continuation_calibration_eligible"
                                ] = True
                                evaluation[
                                    "continuation_calibration_reason"
                                ] = "matching_natural_followup_action"
                            else:
                                evaluation[
                                    "continuation_calibration_eligible"
                                ] = False
                                evaluation[
                                    "continuation_calibration_reason"
                                ] = "followup_pre_action_baseline_unavailable"
                        else:
                            evaluation[
                                "continuation_calibration_eligible"
                            ] = False
                            evaluation[
                                "continuation_calibration_reason"
                            ] = (
                                "followup_action_mismatch"
                                if action != expected
                                else "policy_utility_prediction_unavailable"
                            )
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
    from src.match_engine.world_model.active_learning import (
        active_learning_diagnostics,
    )
    from src.match_engine.world_model.uncertainty import (
        uncertainty_decomposition_diagnostics,
    )

    records = list(getattr(state, "_wm_coach_decision_adoption", None) or [])
    resolved = [record for record in records if record["resolved"]]
    adopted = [record for record in resolved if record["adopted"]]
    interventions = [
        record for record in resolved if record.get("intervention_applied")
    ]
    intervention_adopted = [
        record for record in interventions if record["adopted"]
    ]
    randomized_effect = randomized_adoption_effect(records)
    randomized_outcome = randomized_outcome_effect([records])
    multi_horizon_outcomes = randomized_multi_horizon_effects([records])
    regime_horizon_outcomes = randomized_multi_horizon_effects(
        [records], outcome_family="regime",
    )
    outcome_calibration = policy_outcome_calibration(
        records, outcome_family="regime",
    )
    return {
        "version": 15,
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
        "randomized_outcome_effect": randomized_outcome,
        "randomized_multi_horizon_outcomes": multi_horizon_outcomes,
        "randomized_regime_horizon_outcomes": regime_horizon_outcomes,
        "world_model_outcome_calibration": outcome_calibration,
        "active_learning": active_learning_diagnostics([records]),
        "uncertainty_decomposition": (
            uncertainty_decomposition_diagnostics([records])
        ),
        "records": records,
        "interpretation": (
            "A bounded intervention changes one action logit but does not force "
            "the sampled action; non-intervention adoption remains associative "
            "and does not prove causal influence."
        ),
    }
