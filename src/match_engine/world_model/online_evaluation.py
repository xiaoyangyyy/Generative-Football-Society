"""Aggregate persisted online world-model calibration diagnostics."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.match_engine.world_model.policy_experiment import (
    policy_horizon_key,
    randomized_adoption_effect_from_counts,
    randomized_multi_horizon_effects,
    randomized_outcome_effect,
)
from src.match_engine.world_model.outcome_calibration import (
    policy_outcome_calibration,
)
from src.match_engine.world_model.residual_memory import (
    compile_contextual_residual_memory,
)
from src.match_engine.world_model.active_learning import (
    active_learning_diagnostics,
)
from src.match_engine.world_model.uncertainty import (
    uncertainty_decomposition_diagnostics,
)
from src.match_engine.world_model.opponent_belief import (
    opponent_belief_diagnostics,
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def aggregate_online_calibration(
    match_logs: Iterable[dict[str, Any]],
    *,
    min_transitions: int = 50,
    min_policy_arm: int = 8,
    require_policy_effect: bool = False,
    required_policy_horizon_s: float = 0.0,
    min_residual_samples: int = 20,
    require_outcome_calibration: bool = False,
    require_uncertainty_decomposition: bool = False,
    require_transition_ensemble: bool = False,
    require_opponent_belief: bool = False,
    require_opponent_meta_belief: bool = False,
    require_opponent_change_detection: bool = False,
) -> dict[str, Any]:
    logs = list(match_logs)
    branch_rows: dict[str, list[dict[str, Any]]] = {
        "general": [], "pass": [], "shot": [],
    }
    target_mismatches = 0
    adoption_registered = adoption_resolved = adoption_count = 0
    interventions_applied = intervention_adopted = 0
    randomized_treatment = randomized_treatment_adopted = 0
    randomized_control = randomized_control_adopted = 0
    policy_record_clusters: list[list[dict[str, Any]]] = []
    for payload in logs:
        calibration = payload.get("world_model_online_calibration") or {}
        if calibration.get("calibration_target") not in (
            None, "same_tick_next_observation",
        ):
            target_mismatches += 1
        branches = calibration.get("branches") or {}
        for branch in branch_rows:
            row = branches.get(branch)
            if isinstance(row, dict) and int(row.get("samples", 0)) > 0:
                branch_rows[branch].append(row)
        adoption = payload.get("world_model_decision_adoption") or {}
        adoption_registered += int(adoption.get("registered", 0))
        adoption_resolved += int(adoption.get("resolved", 0))
        adoption_count += int(adoption.get("adopted", 0))
        interventions_applied += int(adoption.get("interventions_applied", 0))
        intervention_adopted += int(adoption.get("intervention_adopted", 0))
        randomized = adoption.get("randomized_policy_effect") or {}
        randomized_treatment += int(randomized.get("treatment", 0))
        randomized_treatment_adopted += int(
            randomized.get("treatment_adopted", 0)
        )
        randomized_control += int(randomized.get("control", 0))
        randomized_control_adopted += int(
            randomized.get("control_adopted", 0)
        )
        records = adoption.get("records") or []
        if isinstance(records, list):
            policy_record_clusters.append(records)

    summaries = {}
    all_finite = True
    for branch, rows in branch_rows.items():
        samples = sum(int(row.get("samples", 0)) for row in rows)

        def weighted(key: str) -> float:
            if not samples:
                return 0.0
            values = [
                _finite(row.get(key)) * int(row.get("samples", 0))
                for row in rows
            ]
            return sum(values) / samples

        summary = {
            "samples": samples,
            "matches": len(rows),
            "mean_weighted_mse": weighted("mean_weighted_mse"),
            "mean_expected_weighted_mse": weighted("expected_weighted_mse"),
            "mean_skill_vs_persistence": weighted("skill_vs_persistence"),
            "mean_trust_factor": weighted("trust_factor"),
            "mean_uncertainty_error_correlation": weighted(
                "uncertainty_error_correlation"
            ),
        }
        all_finite = all_finite and all(
            math.isfinite(float(value))
            for value in summary.values()
            if isinstance(value, (int, float))
        )
        summaries[branch] = summary

    transitions = summaries["general"]["samples"]
    gates = {
        "minimum_transitions": transitions >= int(min_transitions),
        "consistent_same_target": target_mismatches == 0,
        "finite_metrics": all_finite,
    }
    randomized_effect = randomized_adoption_effect_from_counts(
        treatment=randomized_treatment,
        treatment_adopted=randomized_treatment_adopted,
        control=randomized_control,
        control_adopted=randomized_control_adopted,
        min_per_arm=min_policy_arm,
    )
    randomized_outcome = randomized_outcome_effect(
        policy_record_clusters,
        min_per_arm=min_policy_arm,
    )
    multi_horizon_outcomes = randomized_multi_horizon_effects(
        policy_record_clusters,
        min_per_arm=min_policy_arm,
    )
    regime_horizon_outcomes = randomized_multi_horizon_effects(
        policy_record_clusters,
        min_per_arm=min_policy_arm,
        outcome_family="regime",
    )
    required_horizon_key = policy_horizon_key(required_policy_horizon_s)
    required_outcome = regime_horizon_outcomes.get(required_horizon_key)
    if required_outcome is None:
        required_outcome = randomized_outcome_effect(
            policy_record_clusters,
            min_per_arm=min_policy_arm,
            horizon_key=required_horizon_key,
            outcome_family="regime",
        )
    outcome_calibration = policy_outcome_calibration(
        [
            record
            for records in policy_record_clusters
            for record in records
        ],
        min_samples=min_residual_samples,
        outcome_family="regime",
    )
    active_learning = active_learning_diagnostics(policy_record_clusters)
    uncertainty_decomposition = uncertainty_decomposition_diagnostics(
        policy_record_clusters,
    )
    opponent_belief = opponent_belief_diagnostics(logs)
    memory_scopes = sorted({
        (
            str(record.get("checkpoint_signature")),
            str(record.get(
                "environment_signature", "environment_unspecified",
            )),
        )
        for records in policy_record_clusters
        for record in records
        if record.get("checkpoint_signature")
    })
    residual_memory_profiles = {
        f"{checkpoint_signature}|{environment_signature}": (
            compile_contextual_residual_memory(
                logs,
                checkpoint_signature=checkpoint_signature,
                environment_signature=environment_signature,
                min_samples=min_residual_samples,
            ).summary()
        )
        for checkpoint_signature, environment_signature in memory_scopes
    }
    gates["randomized_policy_adoption"] = (
        randomized_effect["ready"] if require_policy_effect else True
    )
    gates["randomized_policy_outcome"] = (
        (
            required_outcome["ready"]
            and required_outcome["cluster_robust"]
        )
        if require_policy_effect else True
    )
    gates["world_model_outcome_calibration"] = (
        (
            outcome_calibration["overall"]["active"]
            and outcome_calibration["overall"]["trust_factor"] >= 0.5
        )
        if require_outcome_calibration else True
    )
    decomposition_ready = bool(
        uncertainty_decomposition["samples"] >= max(2, min_residual_samples)
        and uncertainty_decomposition["mean_composition_identity_error"]
        <= 1e-6
    )
    gates["world_model_uncertainty_decomposition"] = (
        decomposition_ready if require_uncertainty_decomposition else True
    )
    transition_ensemble_ready = bool(
        uncertainty_decomposition["transition_ensemble_predictions"]
        >= max(2, min_residual_samples)
    )
    gates["world_model_transition_ensemble"] = (
        transition_ensemble_ready if require_transition_ensemble else True
    )
    opponent_belief_ready = bool(
        opponent_belief["decision_belief_snapshots"]
        >= max(2, min_residual_samples)
        and opponent_belief["available"]
    )
    gates["opponent_belief_decision_evidence"] = (
        opponent_belief_ready if require_opponent_belief else True
    )
    opponent_meta_belief_ready = bool(
        opponent_belief["meta_prior_backed_decisions"]
        >= max(2, min_residual_samples)
    )
    opponent_change_detection_ready = bool(
        opponent_belief["change_detector_snapshots"]
        >= max(2, min_residual_samples)
        and opponent_belief["all_change_claims_non_controlling"]
    )
    gates["opponent_meta_belief_evidence"] = (
        opponent_meta_belief_ready if require_opponent_meta_belief else True
    )
    gates["opponent_change_detection_audit"] = (
        opponent_change_detection_ready
        if require_opponent_change_detection else True
    )
    return {
        "version": 7,
        "evaluation_kind": (
            "online_world_model_calibration_and_randomized_policy_bridge"
        ),
        "match_logs": len(logs),
        "branches": summaries,
        "decision_adoption": {
            "registered": adoption_registered,
            "resolved": adoption_resolved,
            "adopted": adoption_count,
            "adoption_rate": adoption_count / max(1, adoption_resolved),
            "interventions_applied": interventions_applied,
            "intervention_adopted": intervention_adopted,
            "intervention_adoption_rate": (
                intervention_adopted / max(1, interventions_applied)
            ),
            "causal_interpretation": False,
            "randomized_policy_effect": randomized_effect,
            "randomized_outcome_effect": randomized_outcome,
            "randomized_multi_horizon_outcomes": multi_horizon_outcomes,
            "randomized_regime_horizon_outcomes": regime_horizon_outcomes,
            "required_policy_horizon_key": required_horizon_key,
            "required_policy_outcome": required_outcome,
            "world_model_outcome_calibration": outcome_calibration,
            "active_learning": active_learning,
            "uncertainty_decomposition": uncertainty_decomposition,
            "opponent_belief": opponent_belief,
            "contextual_residual_memory_by_policy_environment": (
                residual_memory_profiles
            ),
        },
        "gates": gates,
        "ready": all(gates.values()),
        "policy_effect_ready": (
            randomized_effect["ready"]
            and required_outcome["ready"]
            and required_outcome["cluster_robust"]
        ),
        "uncertainty_decomposition_ready": decomposition_ready,
        "transition_ensemble_ready": transition_ensemble_ready,
        "opponent_belief_ready": opponent_belief_ready,
        "opponent_meta_belief_ready": opponent_meta_belief_ready,
        "opponent_change_detection_ready": opponent_change_detection_ready,
        "limitations": [
            "Trust factors are valid only for the configured checkpoint and simulator.",
            "Action adoption is temporal association, not causal attribution.",
            "Randomized bridge effects are causal only for action selection inside this simulator.",
            "Short-horizon outcome uncertainty is clustered by match when multiple logs exist.",
            "Outcome forecast trust is checkpoint- and policy-regime-specific.",
            "Residual memory is isolated by checkpoint and policy-environment fingerprint.",
            "Active-learning acquisition is policy-selected, not randomized; its uncertainty reduction is descriptive.",
            "Opponent-belief diagnostics audit grounding and sensitivity; hidden tactical intent has no direct truth label.",
        ],
    }
