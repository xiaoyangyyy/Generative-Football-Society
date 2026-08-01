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
from src.match_engine.world_model.opponent_response import (
    opponent_response_diagnostics,
)
from src.match_engine.world_model.trajectory_game import (
    trajectory_planning_diagnostics,
)
from src.match_engine.world_model.llm_critic_memory import (
    llm_critic_diagnostics,
)
from src.match_engine.world_model.llm_event_hypothesis import (
    semantic_event_diagnostics,
)
from src.match_engine.world_model.event_option import event_option_diagnostics
from src.match_engine.world_model.contrastive_explanation import (
    contrastive_explanation_diagnostics,
)
from src.match_engine.world_model.contrastive_repair import (
    contrastive_repair_diagnostics,
)
from src.match_engine.world_model.risk_certificate import (
    risk_certificate_diagnostics,
)
from src.match_engine.world_model.distributional_claim import (
    distributional_claim_diagnostics,
)
from src.match_engine.world_model.risk_preference_evaluation import (
    risk_preference_diagnostics,
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
    require_opponent_response_model: bool = False,
    require_two_step_trajectory_planning: bool = False,
    require_llm_semantic_critic: bool = False,
    require_llm_semantic_events: bool = False,
    require_learned_semantic_events: bool = False,
    require_llm_event_options: bool = False,
    require_llm_event_option_values: bool = False,
    require_llm_contrastive_faithfulness: bool = False,
    require_llm_contrastive_repair: bool = False,
    require_llm_risk_certificates: bool = False,
    require_llm_distributional_decisions: bool = False,
    require_llm_risk_preferences: bool = False,
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
    opponent_response = opponent_response_diagnostics(logs)
    trajectory_planning = trajectory_planning_diagnostics(logs)
    llm_semantic_critic = llm_critic_diagnostics(logs)
    llm_semantic_events = semantic_event_diagnostics(logs)
    llm_event_options = event_option_diagnostics(logs)
    llm_contrastive = contrastive_explanation_diagnostics(logs)
    llm_contrastive_repair = contrastive_repair_diagnostics(logs)
    llm_risk_certificates = risk_certificate_diagnostics(logs)
    llm_distributional = distributional_claim_diagnostics(logs)
    llm_risk_preferences = risk_preference_diagnostics(logs)
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
    opponent_response_ready = bool(
        opponent_response["realized_predictions"]
        >= max(2, min_residual_samples)
        and opponent_response["validated_learned_predictions"]
        >= max(2, min_residual_samples)
        and opponent_response["mean_composition_identity_error"] <= 1e-6
        and opponent_response["all_llm_hypotheses_non_persistent"]
    )
    gates["opponent_response_model_evidence"] = (
        opponent_response_ready if require_opponent_response_model else True
    )
    trajectory_planning_ready = bool(
        trajectory_planning["validated_predicted_state_decisions"]
        >= max(2, min_residual_samples)
        and trajectory_planning["budgets_respected"]
        and trajectory_planning["all_active_rollouts_holdout_validated"]
        and trajectory_planning["all_non_causal"]
    )
    gates["validated_two_step_trajectory_planning"] = (
        trajectory_planning_ready
        if require_two_step_trajectory_planning else True
    )
    llm_semantic_critic_ready = bool(
        llm_semantic_critic["realized_validated_corrections"]
        >= max(2, min_residual_samples)
        and llm_semantic_critic["match_clustered_corrected_mse"]
        < llm_semantic_critic["match_clustered_baseline_mse"] - 1e-12
        and llm_semantic_critic["all_non_persistent"]
        and llm_semantic_critic["all_predictions_immutable"]
        and llm_semantic_critic["all_bridge_authority_non_increasing"]
    )
    gates["validated_llm_semantic_critic"] = (
        llm_semantic_critic_ready if require_llm_semantic_critic else True
    )
    llm_semantic_events_ready = bool(
        llm_semantic_events["realized_predictions"]
        >= max(2, min_residual_samples)
        and llm_semantic_events["matches"] >= 4
        and llm_semantic_events["all_paired_same_outcome"]
        and llm_semantic_events["all_shadow_only"]
        and llm_semantic_events["all_non_causal"]
        and llm_semantic_events["provenance_compatible"]
        and not llm_semantic_events["authority_active"]
    )
    gates["paired_llm_semantic_event_evaluation"] = (
        llm_semantic_events_ready if require_llm_semantic_events else True
    )
    learned_semantic_events_ready = bool(
        llm_semantic_events["realized_learned_head_predictions"]
        >= max(2, min_residual_samples)
        and llm_semantic_events["learned_head_matches"] >= 4
        and llm_semantic_events["match_clustered_learned_head_brier"]
        < llm_semantic_events["match_clustered_projection_brier"] - 1e-12
        and llm_semantic_events["match_clustered_fused_event_brier"]
        < llm_semantic_events["match_clustered_projection_brier"] - 1e-12
        and llm_semantic_events["all_learned_authority_bounded"]
        and llm_semantic_events["provenance_compatible"]
    )
    gates["validated_learned_semantic_event_fusion"] = (
        learned_semantic_events_ready
        if require_learned_semantic_events else True
    )
    llm_event_options_ready = bool(
        llm_event_options["resolved_events"] >= max(2, min_residual_samples)
        and llm_event_options["malformed_evaluations"] == 0
        and llm_event_options["continuations_observed"]
        >= max(2, min_residual_samples)
        and llm_event_options["matches"] >= 4
        and llm_event_options["budgets_respected"]
        and llm_event_options["all_shadow_only"]
        and llm_event_options["all_non_controlling"]
        and llm_event_options["all_non_causal"]
        and llm_event_options["provenance_compatible"]
    )
    gates["shadow_llm_event_option_evaluation"] = (
        llm_event_options_ready if require_llm_event_options else True
    )
    llm_event_option_values_ready = bool(
        llm_event_options["value_predictions_realized"]
        >= max(2, min_residual_samples)
        and llm_event_options["value_calibration_matches"] >= 4
        and llm_event_options["malformed_value_evaluations"] == 0
        and llm_event_options[
            "all_value_targets_natural_matching_actions"
        ]
        and llm_event_options["match_clustered_value_skill_vs_zero"] > 0.0
        and llm_event_options["provenance_compatible"]
    )
    gates["calibrated_llm_event_option_values"] = (
        llm_event_option_values_ready
        if require_llm_event_option_values else True
    )
    llm_contrastive_ready = bool(
        llm_contrastive["claims"] >= max(4, min_residual_samples)
        and llm_contrastive["matches"] >= 4
        and llm_contrastive["malformed_claim_audits"] == 0
        and llm_contrastive[
            "match_clustered_directional_faithfulness"
        ] >= 0.60
        and llm_contrastive[
            "match_clustered_mean_absolute_factor_effect"
        ] >= 0.005
        and llm_contrastive["budgets_respected"]
        and llm_contrastive["all_shadow_only"]
        and llm_contrastive["all_non_controlling"]
        and llm_contrastive["all_non_causal"]
        and llm_contrastive["provenance_compatible"]
    )
    gates["model_checked_llm_contrastive_faithfulness"] = (
        llm_contrastive_ready
        if require_llm_contrastive_faithfulness else True
    )
    llm_contrastive_repair_ready = bool(
        llm_contrastive_repair["attempts"] >= max(4, min_residual_samples)
        and llm_contrastive_repair["matches"] >= 4
        and llm_contrastive_repair["malformed_repair_audits"] == 0
        and llm_contrastive_repair[
            "match_clustered_repair_success_rate"
        ] >= 0.50
        and llm_contrastive_repair[
            "match_clustered_directional_effect_gain"
        ] > 0.0
        and llm_contrastive_repair["all_single_call"]
        and llm_contrastive_repair["budgets_respected"]
        and llm_contrastive_repair["all_actions_immutable"]
        and llm_contrastive_repair["all_shadow_only"]
        and llm_contrastive_repair["all_non_controlling"]
        and llm_contrastive_repair["all_non_causal"]
        and llm_contrastive_repair["provenance_compatible"]
    )
    gates["one_shot_llm_contrastive_repair"] = (
        llm_contrastive_repair_ready
        if require_llm_contrastive_repair else True
    )
    llm_risk_certificates_ready = bool(
        llm_risk_certificates["realized_certificates"]
        >= max(4, min_residual_samples)
        and llm_risk_certificates["matches"] >= 4
        and llm_risk_certificates["malformed_certificate_evaluations"] == 0
        and llm_risk_certificates["unscored_eligible_certificates"] == 0
        and llm_risk_certificates["certified_outcomes"]
        >= max(4, min_residual_samples)
        and llm_risk_certificates["certified_matches"] >= 4
        and llm_risk_certificates[
            "match_clustered_certified_violation_rate"
        ] <= llm_risk_certificates[
            "match_clustered_mean_certified_threshold"
        ]
        and llm_risk_certificates["match_clustered_violation_brier"] <= 0.25
        and llm_risk_certificates["all_shadow_only"]
        and llm_risk_certificates["all_non_controlling"]
        and llm_risk_certificates["all_non_causal"]
        and llm_risk_certificates["all_interval_monitors_complete"]
        and llm_risk_certificates[
            "all_member_trajectory_identity_preserved"
        ]
        and llm_risk_certificates["provenance_compatible"]
    )
    gates["realized_llm_risk_certificates"] = (
        llm_risk_certificates_ready
        if require_llm_risk_certificates else True
    )
    llm_distributional_ready = bool(
        llm_distributional["realized_distributions"]
        >= max(4, min_residual_samples)
        and llm_distributional["matches"] >= 4
        and llm_distributional["malformed_evaluations"] == 0
        and llm_distributional["unscored_eligible_claims"] == 0
        and llm_distributional[
            "match_clustered_directional_faithfulness"
        ] >= 0.60
        and 0.55 <= llm_distributional[
            "match_clustered_central_80_coverage"
        ] <= 0.98
        and 0.25 <= llm_distributional[
            "match_clustered_below_median_rate"
        ] <= 0.75
        and llm_distributional["match_clustered_crps"]
        <= llm_distributional["match_clustered_mean_absolute_error"] + 1e-12
        and llm_distributional["all_shadow_only"]
        and llm_distributional["all_non_controlling"]
        and llm_distributional["all_non_causal"]
        and llm_distributional["all_member_identity_preserved"]
        and llm_distributional["all_predictive_distributions_calibrated"]
        and llm_distributional["provenance_compatible"]
    )
    gates["calibrated_llm_distributional_decisions"] = (
        llm_distributional_ready
        if require_llm_distributional_decisions else True
    )
    llm_risk_preferences_ready = bool(
        llm_risk_preferences["realized_preference_values"]
        >= max(4, min_residual_samples)
        and llm_risk_preferences["matches"] >= 4
        and llm_risk_preferences["malformed_preference_contexts"] == 0
        and llm_risk_preferences["malformed_preference_evaluations"] == 0
        and llm_risk_preferences[
            "unscored_eligible_preference_horizons"
        ] == 0
        and llm_risk_preferences[
            "match_clustered_prospective_consistency"
        ] >= 0.60
        and llm_risk_preferences[
            "match_clustered_prospective_robustness"
        ] >= 0.60
        and llm_risk_preferences[
            "match_clustered_prospective_temporal_robustness"
        ] >= 0.60
        and 0.55 <= llm_risk_preferences[
            "match_clustered_central_80_coverage"
        ] <= 0.98
        and llm_risk_preferences["match_clustered_preference_value_crps"]
        <= llm_risk_preferences["match_clustered_preference_value_mae"] + 1e-12
        and llm_risk_preferences["all_predictive_distributions_calibrated"]
        and llm_risk_preferences["all_shadow_only"]
        and llm_risk_preferences["all_non_controlling"]
        and llm_risk_preferences["all_non_causal"]
        and llm_risk_preferences["provenance_compatible"]
    )
    gates["calibrated_llm_risk_preferences"] = (
        llm_risk_preferences_ready if require_llm_risk_preferences else True
    )
    return {
        "version": 23,
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
            "opponent_response": opponent_response,
            "trajectory_planning": trajectory_planning,
            "llm_semantic_critic": llm_semantic_critic,
            "llm_semantic_events": llm_semantic_events,
            "llm_event_options": llm_event_options,
            "llm_contrastive_explanations": llm_contrastive,
            "llm_contrastive_repairs": llm_contrastive_repair,
            "llm_risk_certificates": llm_risk_certificates,
            "llm_distributional_decisions": llm_distributional,
            "llm_risk_preferences": llm_risk_preferences,
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
        "opponent_response_model_ready": opponent_response_ready,
        "two_step_trajectory_planning_ready": trajectory_planning_ready,
        "llm_semantic_critic_ready": llm_semantic_critic_ready,
        "llm_semantic_events_ready": llm_semantic_events_ready,
        "learned_semantic_events_ready": learned_semantic_events_ready,
        "llm_event_options_ready": llm_event_options_ready,
        "llm_event_option_values_ready": llm_event_option_values_ready,
        "llm_contrastive_faithfulness_ready": llm_contrastive_ready,
        "llm_contrastive_repair_ready": llm_contrastive_repair_ready,
        "llm_risk_certificates_ready": llm_risk_certificates_ready,
        "llm_distributional_decisions_ready": llm_distributional_ready,
        "llm_risk_preferences_ready": llm_risk_preferences_ready,
        "limitations": [
            "Trust factors are valid only for the configured checkpoint and simulator.",
            "Action adoption is temporal association, not causal attribution.",
            "Randomized bridge effects are causal only for action selection inside this simulator.",
            "Short-horizon outcome uncertainty is clustered by match when multiple logs exist.",
            "Outcome forecast trust is checkpoint- and policy-regime-specific.",
            "Residual memory is isolated by checkpoint and policy-environment fingerprint.",
            "Active-learning acquisition is policy-selected, not randomized; its uncertainty reduction is descriptive.",
            "Opponent-belief diagnostics audit grounding and sensitivity; hidden tactical intent has no direct truth label.",
            "Opponent-response diagnostics are match-clustered and observational; held-out skill does not establish causality.",
            "Predicted-state continuation search has bounded authority only after grouped two-step holdout gain.",
            "LLM semantic residual critiques remain shadow predictions until match-held-out error reduction; they never rewrite neural forecasts.",
            "LLM semantic event hypotheses and neural event probabilities are scored against the same outcome and remain non-controlling.",
            "Learned semantic event fusion is evaluated against its transparent projection baseline under bounded per-event authority.",
            "LLM event-conditioned options remain shadow-only; continuation agreement is descriptive and does not establish option value.",
            "Conditional option values are scored only after the naturally observed continuation matches the declared branch; this calibration is observational, not a counterfactual effect estimate.",
            "Contrastive explanation probes measure faithfulness to the configured world model under schema-level context neutralization; they do not establish real-football causal explanations.",
            "One-shot contrastive repair may revise only an explanation after model counterevidence; repair success cannot change the frozen action, controls, forecasts, or policy authority.",
            "LLM chance constraints are shadow certificates over transparent ensemble modes; conservative certification is observational and cannot veto or authorize an action.",
            "Distributional LLM claims are scored against realized simulator utility; calibrated member spread is descriptive and does not establish causal action value.",
            "LLM risk preferences are model-checked on frozen scenarios; only the selected action is realized, so reported preference regret is prospective rather than counterfactual ground truth.",
            "Preference robustness uses fixed local parameter and leave-one-axis-out stress tests; it is a sensitivity certificate, not proof against every possible utility function or model error.",
            "Temporal utility paths use member identity plus held-out empirical residual-rank templates when compatible history exists, otherwise an explicit comonotonic fallback; path scenario rates are not calibrated temporal probabilities.",
        ],
    }
