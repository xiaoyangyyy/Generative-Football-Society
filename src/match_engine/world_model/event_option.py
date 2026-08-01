"""Shadow-only LLM event-conditioned options verified by member trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS
from src.match_engine.world_model.state_scales import (
    FALSIFIABLE_SEMANTIC_EVENTS,
    semantic_event_member_indicators,
)
from src.match_engine.world_model.trajectory_game import (
    encode_predicted_state_action,
    evaluate_predicted_state_action,
)


EVENT_OPTION_VERSION = 2
_HORIZON_PATTERN = re.compile(r"^(transition|\d+(?:\.\d+)?s)$")


def llm_event_option_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": EVENT_OPTION_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "shadow_event_conditioned_two_branch_policy",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-event-option:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_event_option(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    first_action = str(raw.get("first_action", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    event = str(raw.get("event", "")).strip().lower()
    on_occurrence = str(raw.get("on_occurrence", "")).strip().lower()
    on_absence = str(raw.get("on_absence", "")).strip().lower()
    if (
        first_action not in RESPONSE_ACTIONS
        or not _HORIZON_PATTERN.fullmatch(horizon)
        or event not in FALSIFIABLE_SEMANTIC_EVENTS
        or on_occurrence not in RESPONSE_ACTIONS
        or on_absence not in RESPONSE_ACTIONS
        or on_occurrence == on_absence
    ):
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "first_action": first_action,
        "horizon": horizon,
        "event": event,
        "on_occurrence": on_occurrence,
        "on_absence": on_absence,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def _conditional_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Jeffreys shrinkage prevents empty member branches from overclaiming."""
    overall = float(np.mean(values))
    return float(
        (np.sum(values * weights) + 0.5 * overall)
        / (np.sum(weights) + 0.5)
    )


def evaluate_llm_event_option(
    runtime,
    state,
    packet: dict[str, Any],
    raw_option: Any,
    *,
    team_id: str,
    selected_action: str,
    option_signature: str = "event-option-contract-unspecified",
    uncertainty_penalty: float = 0.25,
    max_member_evaluations: int = 16,
    max_member_trajectory_paths: int = 144,
) -> dict[str, Any]:
    """Evaluate one proposed conditional option without changing live policy."""
    option = validate_llm_event_option(raw_option)
    base = {
        "version": EVENT_OPTION_VERSION,
        "accepted": False,
        "authority_active": False,
        "policy_mutated": False,
        "shadow_only": True,
        "causal_interpretation": False,
        "can_execute_future_action": False,
        "can_update_world_model": False,
        "option_signature": str(option_signature),
    }
    if option is None:
        return {**base, "reason": "missing_or_invalid_event_option"}
    if option["first_action"] != str(selected_action).lower():
        return {
            **base,
            "reason": "option_first_action_must_equal_selected_action",
            "option": option,
        }
    candidate = next((
        item for item in packet.get("candidates") or []
        if str(item.get("action", "")).lower() == option["first_action"]
    ), None)
    prediction = (
        (candidate.get("multi_horizon_predictions") or {}).get(
            option["horizon"]
        ) if candidate else None
    )
    event_probabilities = (
        prediction.get("semantic_event_probabilities")
        if isinstance(prediction, dict) else None
    ) or {}
    if option["event"] not in event_probabilities:
        return {
            **base,
            "reason": "option_event_horizon_not_evaluated",
            "option": option,
        }
    rollout_steps = int((prediction or {}).get("rollout_steps", 1))
    if rollout_steps not in {1, 2}:
        return {
            **base,
            "reason": "option_rollout_depth_not_validated",
            "option": option,
            "rollout_steps": rollout_steps,
        }
    planning_gate_provider = getattr(runtime, "two_step_planning_gate", None)
    try:
        planning_gate = (
            dict(planning_gate_provider())
            if callable(planning_gate_provider) else {"active": False}
        )
    except (RuntimeError, TypeError, ValueError):
        planning_gate = {
            "active": False,
            "authority": 0.0,
            "reason": "two_step_validation_contract_error",
        }
    try:
        planning_authority = finite_float(
            planning_gate.get("authority") or 0.0, 0.0,
        )
    except (TypeError, ValueError):
        planning_authority = 0.0
    planning_gate["authority"] = float(np.clip(
        planning_authority, 0.0, 0.50,
    ))
    planning_gate["active"] = bool(
        planning_gate.get("active") and planning_gate["authority"] > 0.0
    )
    if not planning_gate.get("active"):
        return {
            **base,
            "reason": "option_predicted_state_gate_closed",
            "option": option,
            "planning_gate": planning_gate,
        }
    attacking_home = str(team_id) == str(state.home.team_id)
    observation = runtime.encode_state(state, attacking_home=attacking_home)
    option_horizon_s = max(
        0.1, float((prediction or {}).get("horizon_s", 10.0)),
    )
    first_vector = encode_high_level_action(
        option["first_action"],
        target=np.asarray(candidate["target"], dtype=np.float32),
        horizon_s=option_horizon_s / rollout_steps,
    )
    first_vector[13] = float(candidate.get("physics_prior", 0.5))
    try:
        first_output = runtime.imagine(
            observation,
            first_vector,
            steps=rollout_steps,
            carry_hidden=False,
            quality_kind=(
                "shot" if option["first_action"] == "shot" else "pass"
            ),
        )
        future_members = np.asarray(
            (first_output.uncertainty_samples or {})["transition_states"],
            dtype=np.float32,
        )
    except (KeyError, RuntimeError, TypeError, ValueError, AttributeError):
        return {
            **base,
            "reason": "option_first_member_rollout_failed",
            "option": option,
            "planning_gate": planning_gate,
        }
    if future_members.ndim == 3 and future_members.shape[1] == 1:
        future_members = future_members[:, 0, :]
    if future_members.ndim != 2 or future_members.shape[1] != observation.size:
        return {
            **base,
            "reason": "option_first_member_rollout_invalid_shape",
            "option": option,
            "planning_gate": planning_gate,
        }
    future_members = np.clip(np.nan_to_num(
        future_members, nan=0.0, posinf=1.0, neginf=0.0,
    ), 0.0, 1.0)
    members = len(future_members)
    budget = max(0, min(32, int(max_member_evaluations)))
    required_evaluations = 2 * members
    transition_members = max(1, int(getattr(
        getattr(runtime, "model", None),
        "transition_member_count",
        members,
    )))
    trajectory_path_budget = max(
        0, min(144, int(max_member_trajectory_paths)),
    )
    required_trajectory_paths = (
        members + required_evaluations * transition_members
    )
    if members < 2 or required_evaluations > budget:
        return {
            **base,
            "reason": "option_member_budget_insufficient",
            "option": option,
            "member_evaluations_required": required_evaluations,
            "member_evaluation_budget": budget,
        }
    if required_trajectory_paths > trajectory_path_budget:
        return {
            **base,
            "reason": "option_nested_trajectory_budget_insufficient",
            "option": option,
            "member_evaluations_required": required_evaluations,
            "member_evaluation_budget": budget,
            "trajectory_member_paths_required": required_trajectory_paths,
            "trajectory_member_path_budget": trajectory_path_budget,
        }
    event_index = FALSIFIABLE_SEMANTIC_EVENTS.index(option["event"])
    indicators = semantic_event_member_indicators(
        observation.reshape(1, -1), future_members[:, None, :],
    )[:, 0, event_index]
    learned_member_probabilities = np.full(members, 0.5, dtype=np.float32)
    semantic_gate_provider = getattr(runtime, "semantic_event_head_gate", None)
    try:
        semantic_gate = (
            dict(semantic_gate_provider(
                option["event"], rollout_steps=rollout_steps,
            ))
            if callable(semantic_gate_provider)
            else {"active": False, "authority": 0.0}
        )
    except (RuntimeError, TypeError, ValueError):
        semantic_gate = {
            "active": False,
            "authority": 0.0,
            "reason": "semantic_event_validation_contract_error",
        }
    if semantic_gate.get("active"):
        try:
            import torch

            with torch.no_grad():
                learned_member_probabilities = torch.sigmoid(
                    runtime.model.semantic_event_logits(
                        torch.from_numpy(observation).unsqueeze(0),
                        torch.from_numpy(future_members[:, None, :]),
                        torch.from_numpy(first_vector).unsqueeze(0),
                    )
                )[:, 0, event_index].numpy()
        except (RuntimeError, TypeError, ValueError, AttributeError):
            semantic_gate = {
                **semantic_gate,
                "active": False,
                "authority": 0.0,
                "reason": "member_semantic_head_failed",
            }
    learned_member_probabilities = np.clip(np.nan_to_num(
        learned_member_probabilities, nan=0.5, posinf=1.0, neginf=0.0,
    ), 0.0, 1.0)
    try:
        semantic_authority_raw = finite_float(
            semantic_gate.get("authority") or 0.0, 0.0,
        ) if semantic_gate.get("active") else 0.0
    except (TypeError, ValueError):
        semantic_authority_raw = 0.0
    semantic_authority = float(np.clip(
        semantic_authority_raw, 0.0, 0.50,
    ))
    occurrence_weights = np.clip(
        indicators
        + semantic_authority * (learned_member_probabilities - indicators),
        0.0,
        1.0,
    )
    absence_weights = 1.0 - occurrence_weights
    branch_values: dict[str, np.ndarray] = {}
    branch_policy_utilities: dict[str, np.ndarray] = {}
    policy_utility_predictor = getattr(
        runtime, "predict_policy_utility", None,
    )
    calibratable = callable(policy_utility_predictor)
    continuation_horizon_s = max(
        0.1, min(60.0, float(packet.get("horizon_s", 10.0))),
    )
    evaluations = model_calls = 0
    try:
        for action_name in (option["on_occurrence"], option["on_absence"]):
            values = []
            policy_utilities = []
            for member_state in future_members:
                if calibratable:
                    continuation_vector = encode_predicted_state_action(
                        member_state,
                        action_name=action_name,
                        attacking_home=attacking_home,
                        horizon_s=continuation_horizon_s,
                    )
                    forecast = policy_utility_predictor(
                        member_state,
                        continuation_vector,
                        action_kind=action_name,
                        attacking_home=attacking_home,
                        horizon_s=continuation_horizon_s,
                    )
                    policy_utility = float(forecast["policy_utility"])
                    uncertainty = float(forecast.get("uncertainty", 1.0))
                    turnover = 1.0 - float(forecast.get(
                        "retention_probability", 0.0,
                    ))
                    result = {
                        "expected_value": policy_utility,
                        "uncertainty": uncertainty,
                        "turnover": turnover,
                        "model_calls": 1,
                    }
                else:
                    result = evaluate_predicted_state_action(
                        runtime,
                        member_state,
                        action_name=action_name,
                        attacking_home=attacking_home,
                        horizon_s=continuation_horizon_s,
                    )
                    policy_utility = float("nan")
                evaluations += 1
                model_calls += int(result.get("model_calls", 1))
                values.append(
                    float(result["expected_value"])
                    - float(uncertainty_penalty) * float(result["uncertainty"])
                    - 0.15 * float(result["turnover"])
                )
                policy_utilities.append(policy_utility)
            branch_values[action_name] = np.asarray(values, dtype=np.float64)
            branch_policy_utilities[action_name] = np.asarray(
                policy_utilities, dtype=np.float64,
            )
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return {
            **base,
            "reason": "option_continuation_evaluation_failed",
            "option": option,
            "member_evaluations": evaluations,
            "member_evaluation_budget": budget,
        }
    if not all(np.all(np.isfinite(values)) for values in branch_values.values()):
        return {
            **base,
            "reason": "option_continuation_value_not_finite",
            "option": option,
            "member_evaluations": evaluations,
            "member_evaluation_budget": budget,
        }
    if calibratable and not all(
        np.all(np.isfinite(values))
        for values in branch_policy_utilities.values()
    ):
        return {
            **base,
            "reason": "option_policy_utility_not_finite",
            "option": option,
            "member_evaluations": evaluations,
            "member_evaluation_budget": budget,
        }
    occur_values = branch_values[option["on_occurrence"]]
    absent_values = branch_values[option["on_absence"]]
    occur_conditional = _conditional_mean(occur_values, occurrence_weights)
    absent_conditional = _conditional_mean(absent_values, absence_weights)
    conditional_policy_utilities = {}
    if calibratable:
        conditional_policy_utilities = {
            "on_occurrence": _conditional_mean(
                branch_policy_utilities[option["on_occurrence"]],
                occurrence_weights,
            ),
            "on_absence": _conditional_mean(
                branch_policy_utilities[option["on_absence"]],
                absence_weights,
            ),
        }
    try:
        raw_event_probability = finite_float(
            event_probabilities[option["event"]], float("nan"),
        )
    except (TypeError, ValueError):
        raw_event_probability = float("nan")
    if not np.isfinite(raw_event_probability):
        return {
            **base,
            "reason": "option_event_probability_not_finite",
            "option": option,
            "member_evaluations": evaluations,
            "member_evaluation_budget": budget,
        }
    event_probability = float(np.clip(raw_event_probability, 0.0, 1.0))
    conditional_value = (
        event_probability * occur_conditional
        + (1.0 - event_probability) * absent_conditional
    )
    fixed_values = {
        option["on_occurrence"]: float(np.mean(occur_values)),
        option["on_absence"]: float(np.mean(absent_values)),
    }
    best_fixed = max(fixed_values.values())
    audit = {
        **base,
        "accepted": True,
        "reason": "shadow_member_conditioned_event_option",
        "option": option,
        "checkpoint_signature": str(packet.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(packet.get(
            "environment_signature", "environment_unspecified",
        )),
        "event_probability": event_probability,
        "rollout_steps": rollout_steps,
        "planning_gate": planning_gate,
        "semantic_event_gate": semantic_gate,
        "member_event_weights": [float(value) for value in occurrence_weights],
        "effective_occurrence_support": float(np.sum(occurrence_weights)),
        "effective_absence_support": float(np.sum(absence_weights)),
        "on_occurrence_conditional_value": occur_conditional,
        "on_absence_conditional_value": absent_conditional,
        "conditional_continuation_value": float(conditional_value),
        "fixed_branch_values": fixed_values,
        "best_fixed_branch_value": float(best_fixed),
        "conditional_gain_vs_best_fixed": float(
            conditional_value - best_fixed
        ),
        "continuation_value_calibratable": calibratable,
        "continuation_prediction_horizon_s": continuation_horizon_s,
        "conditional_policy_utility_predictions": (
            conditional_policy_utilities
        ),
        "calibration_target": (
            "simulator_policy_utility_after_matching_next_team_action"
            if calibratable else "unavailable_runtime_policy_utility"
        ),
        "first_action_value": float(candidate.get("risk_adjusted_value", 0.0)),
        "shadow_option_value": float(
            float(candidate.get("risk_adjusted_value", 0.0))
            + 0.45 * conditional_value
        ),
        "member_evaluations": evaluations,
        "member_evaluation_budget": budget,
        "trajectory_member_paths": required_trajectory_paths,
        "trajectory_member_path_budget": trajectory_path_budget,
        "model_calls": model_calls + 1,
        "jeffreys_branch_shrinkage": 0.5,
    }
    packet["llm_event_option_audit"] = audit
    return audit


def event_option_diagnostics(match_logs: list[dict[str, Any]]) -> dict[str, Any]:
    match_rows: list[list[dict[str, Any]]] = []
    for payload in match_logs:
        rows = []
        records = (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        for record in records:
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            for outcome in outcomes.values():
                evaluation = (
                    outcome.get("llm_event_option_evaluation")
                    if isinstance(outcome, dict) else None
                )
                if isinstance(evaluation, dict):
                    rows.append(evaluation)
        if rows:
            match_rows.append(rows)
    flat = [row for rows in match_rows for row in rows]
    required_flags = (
        "shadow_only", "authority_active", "policy_mutated",
        "can_execute_future_action", "causal_interpretation",
    )
    valid = []
    malformed = 0
    for row in flat:
        try:
            version = int(row["version"])
            evaluations = int(row["member_evaluations"])
            budget = int(row["member_evaluation_budget"])
            trajectory_paths = int(row["trajectory_member_paths"])
            trajectory_budget = int(row["trajectory_member_path_budget"])
            gain = float(row["conditional_gain_vs_best_fixed"])
        except (KeyError, TypeError, ValueError, OverflowError):
            malformed += 1
            continue
        if (
            version != EVENT_OPTION_VERSION
            or evaluations < 0 or budget <= 0 or evaluations > budget
            or trajectory_paths < 0 or trajectory_budget <= 0
            or trajectory_paths > trajectory_budget
            or not np.isfinite(gain)
            or any(flag not in row for flag in required_flags)
        ):
            malformed += 1
            continue
        valid.append(row)
    continuation_rows = [
        row for row in valid if row.get("next_action_observed")
    ]
    calibration_by_row: dict[int, tuple[float, float]] = {}
    calibration_malformed = 0
    for row in valid:
        if not row.get("continuation_calibration_observed"):
            continue
        try:
            predicted = float(row["expected_continuation_policy_utility"])
            observed = float(
                row["realized_continuation_outcome"]["policy_utility"]
            )
            residual = float(row["continuation_policy_utility_residual"])
            squared_error = float(
                row["continuation_policy_utility_squared_error"]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            calibration_malformed += 1
            continue
        if (
            not all(np.isfinite(value) for value in (
                predicted, observed, residual, squared_error,
            ))
            or not row.get("continuation_calibration_eligible")
            or not row.get("expected_action_matched")
            or abs(residual - (observed - predicted)) > 1e-6
            or abs(squared_error - residual ** 2) > 1e-6
        ):
            calibration_malformed += 1
            continue
        calibration_by_row[id(row)] = (predicted, observed)
    calibration_groups = []
    for rows in match_rows:
        group = [
            calibration_by_row[id(row)]
            for row in rows if id(row) in calibration_by_row
        ]
        if group:
            calibration_groups.append(group)
    match_residuals = [
        float(np.mean([observed - predicted for predicted, observed in group]))
        for group in calibration_groups
    ]
    match_mses = [
        float(np.mean([
            (observed - predicted) ** 2 for predicted, observed in group
        ]))
        for group in calibration_groups
    ]
    match_maes = [
        float(np.mean([
            abs(observed - predicted) for predicted, observed in group
        ]))
        for group in calibration_groups
    ]
    match_baseline_mses = [
        float(np.mean([observed ** 2 for _, observed in group]))
        for group in calibration_groups
    ]
    clustered_mse = float(np.mean(match_mses)) if match_mses else 0.0
    clustered_baseline_mse = (
        float(np.mean(match_baseline_mses)) if match_baseline_mses else 0.0
    )
    signatures = sorted({str(row.get("option_signature", "")) for row in valid})
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {"", "runtime_unspecified", "environment_unspecified"}
    return {
        "version": EVENT_OPTION_VERSION,
        "evaluation_kind": "shadow_event_conditioned_option_followup",
        "resolved_events": len(valid),
        "malformed_evaluations": malformed,
        "matches": len(match_rows),
        "continuations_observed": len(continuation_rows),
        "value_predictions_realized": len(calibration_by_row),
        "value_calibration_matches": len(calibration_groups),
        "malformed_value_evaluations": calibration_malformed,
        "continuation_match_rate": float(np.mean([
            float(bool(row.get("expected_action_matched")))
            for row in continuation_rows
        ])) if continuation_rows else 0.0,
        "mean_conditional_gain_vs_best_fixed": float(np.mean([
            float(row.get("conditional_gain_vs_best_fixed", 0.0))
            for row in valid
        ])) if valid else 0.0,
        "match_clustered_value_bias": (
            float(np.mean(match_residuals)) if match_residuals else 0.0
        ),
        "match_clustered_value_mae": (
            float(np.mean(match_maes)) if match_maes else 0.0
        ),
        "match_clustered_value_mse": clustered_mse,
        "match_clustered_zero_baseline_mse": clustered_baseline_mse,
        "match_clustered_value_skill_vs_zero": (
            1.0 - clustered_mse / max(clustered_baseline_mse, 1e-6)
            if calibration_groups else 0.0
        ),
        "all_value_targets_natural_matching_actions": all(
            bool(row.get("expected_action_matched"))
            for row in valid
            if row.get("continuation_calibration_observed")
        ),
        "budgets_respected": all(
            int(row.get("member_evaluations", 0))
            <= int(row.get("member_evaluation_budget", 0))
            and int(row.get("trajectory_member_paths", 0))
            <= int(row.get("trajectory_member_path_budget", 0))
            for row in valid
        ),
        "all_shadow_only": all(bool(row.get("shadow_only")) for row in valid),
        "all_non_controlling": all(
            not bool(row.get("authority_active"))
            and not bool(row.get("policy_mutated"))
            and not bool(row.get("can_execute_future_action"))
            for row in valid
        ),
        "all_non_causal": all(
            not bool(row.get("causal_interpretation")) for row in valid
        ),
        "provenance_compatible": bool(
            len(signatures) == 1 and signatures[0] not in unspecified
            and len(scopes) == 1
            and all(value not in unspecified for value in scopes[0])
        ),
        "option_signatures": signatures,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
