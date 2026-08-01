"""Model-check LLM questions about decision-relevant opponent information."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    TACTICAL_FEATURES,
    normalise_distribution,
    normalized_entropy,
    tactic_feature_vector,
)
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS
from src.match_engine.world_model.policy_experiment import (
    policy_horizon_seconds,
)


OPPONENT_INFORMATION_QUERY_VERSION = 2
_PURPOSES = {"reduce_opponent_uncertainty", "resolve_action_choice"}
_OBSERVATION_SIGMA = 0.18


def llm_opponent_information_query_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": OPPONENT_INFORMATION_QUERY_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "shadow_opponent_tactical_value_of_information",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-opponent-information-query:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_opponent_information_query(
    raw: Any,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("selected_action", "")).strip().lower()
    feature = str(raw.get("feature", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    purpose = str(raw.get("purpose", "")).strip().lower()
    action_if_high = str(raw.get("action_if_high", "")).strip().lower()
    action_if_low = str(raw.get("action_if_low", "")).strip().lower()
    try:
        confidence = float(raw.get("confidence", 0.0))
        horizon_s = float(policy_horizon_seconds(horizon))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        action not in RESPONSE_ACTIONS
        or action_if_high not in RESPONSE_ACTIONS
        or action_if_low not in RESPONSE_ACTIONS
        or feature not in TACTICAL_FEATURES
        or purpose not in _PURPOSES
        or not 10.0 <= horizon_s <= 300.0
        or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "selected_action": action,
        "feature": feature,
        "horizon": horizon,
        "purpose": purpose,
        "action_if_high": action_if_high,
        "action_if_low": action_if_low,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def _posterior(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, dict) or set(raw) != set(OPPONENT_HYPOTHESES):
        return None
    try:
        values = np.asarray([
            float(raw[name]) for name in OPPONENT_HYPOTHESES
        ], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not np.all(np.isfinite(values)) or np.any(values < 0.0)
        or abs(float(np.sum(values)) - 1.0) > 1e-6
    ):
        return None
    return normalise_distribution(values)


def _action_values(raw: Any) -> dict[str, np.ndarray] | None:
    if not isinstance(raw, dict) or set(raw) != set(RESPONSE_ACTIONS):
        return None
    output = {}
    try:
        for action in RESPONSE_ACTIONS:
            row = raw[action]
            if not isinstance(row, dict) or set(row) != set(
                OPPONENT_HYPOTHESES
            ):
                return None
            values = np.asarray([
                float(row[name]) for name in OPPONENT_HYPOTHESES
            ], dtype=np.float64)
            if (
                not np.all(np.isfinite(values))
                or np.any(np.abs(values) > 10.0)
            ):
                return None
            output[action] = values
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return output


def _high_likelihoods(feature: str) -> np.ndarray:
    index = TACTICAL_FEATURES.index(feature)
    means = np.asarray([
        tactic_feature_vector(name)[index] for name in OPPONENT_HYPOTHESES
    ], dtype=np.float64)
    z = (means - 0.5) / (_OBSERVATION_SIGMA * math.sqrt(2.0))
    probabilities = np.asarray([
        0.5 * (1.0 + math.erf(float(value))) for value in z
    ], dtype=np.float64)
    return np.clip(probabilities, 0.02, 0.98)


def _branch(
    posterior: np.ndarray,
    likelihood: np.ndarray,
    action_values: dict[str, np.ndarray],
) -> tuple[float, np.ndarray, dict[str, float], str, float]:
    probability = float(np.dot(posterior, likelihood))
    conditioned = normalise_distribution(posterior * likelihood)
    values = {
        action: float(np.dot(conditioned, action_values[action]))
        for action in RESPONSE_ACTIONS
    }
    best_action = max(RESPONSE_ACTIONS, key=lambda action: (
        values[action], -RESPONSE_ACTIONS.index(action),
    ))
    return probability, conditioned, values, best_action, values[best_action]


def _feature_report(
    feature: str,
    posterior: np.ndarray,
    action_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    high_likelihood = _high_likelihoods(feature)
    probability, high_posterior, high_values, high_action, high_best = _branch(
        posterior, high_likelihood, action_values,
    )
    low_probability, low_posterior, low_values, low_action, low_best = _branch(
        posterior, 1.0 - high_likelihood, action_values,
    )
    if abs(probability + low_probability - 1.0) > 1e-9:
        raise ValueError("binary observation branches must sum to one")
    current_values = {
        action: float(np.dot(posterior, action_values[action]))
        for action in RESPONSE_ACTIONS
    }
    current_action = max(RESPONSE_ACTIONS, key=lambda action: (
        current_values[action], -RESPONSE_ACTIONS.index(action),
    ))
    current_best = current_values[current_action]
    adaptive_value = probability * high_best + low_probability * low_best
    expected_entropy = (
        probability * normalized_entropy(high_posterior)
        + low_probability * normalized_entropy(low_posterior)
    )
    return {
        "feature": feature,
        "binary_observation": "feature_above_neutral_0_5",
        "forecast_high_rate": probability,
        "hypothesis_high_likelihoods": {
            name: float(high_likelihood[index])
            for index, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "posterior_if_high": {
            name: float(high_posterior[index])
            for index, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "posterior_if_low": {
            name: float(low_posterior[index])
            for index, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "current_action_values": current_values,
        "current_best_action": current_action,
        "current_best_value": current_best,
        "best_action_if_high": high_action,
        "best_action_if_low": low_action,
        "action_values_if_high": high_values,
        "action_values_if_low": low_values,
        "expected_adaptive_value": adaptive_value,
        "expected_decision_value_of_information": max(
            0.0, adaptive_value - current_best,
        ),
        "prior_normalized_entropy": normalized_entropy(posterior),
        "expected_posterior_normalized_entropy": expected_entropy,
        "expected_normalized_information_gain": max(
            0.0, normalized_entropy(posterior) - expected_entropy,
        ),
        "branch_action_switch": high_action != low_action,
    }


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "opponent-information-query-audit:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _audit_from_basis(
    query: dict[str, Any],
    basis: dict[str, Any],
    *,
    query_signature: str,
) -> dict[str, Any] | None:
    posterior = _posterior(basis.get("posterior"))
    values = _action_values(basis.get("action_hypothesis_values"))
    try:
        threshold = float(basis.get("observation_threshold"))
        sigma = float(basis.get("observation_sigma"))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        posterior is None or values is None
        or abs(threshold - 0.5) > 1e-12
        or abs(sigma - _OBSERVATION_SIGMA) > 1e-12
        or basis.get("posterior_source") not in {
            "selected_action_response_posterior",
            "current_opponent_belief",
        }
    ):
        return None
    leaderboard = [
        _feature_report(feature, posterior, values)
        for feature in TACTICAL_FEATURES
    ]
    if query["purpose"] == "resolve_action_choice":
        leaderboard.sort(key=lambda row: (
            -row["expected_decision_value_of_information"],
            -row["expected_normalized_information_gain"],
            row["feature"],
        ))
        ranking_objective = "decision_value_then_information_gain"
    else:
        leaderboard.sort(key=lambda row: (
            -row["expected_normalized_information_gain"],
            -row["expected_decision_value_of_information"],
            row["feature"],
        ))
        ranking_objective = "information_gain_then_decision_value"
    for rank, row in enumerate(leaderboard, start=1):
        row["model_rank"] = rank
    selected = next(
        row for row in leaderboard if row["feature"] == query["feature"]
    )
    objective_key = (
        "expected_decision_value_of_information"
        if query["purpose"] == "resolve_action_choice"
        else "expected_normalized_information_gain"
    )
    best_objective = float(leaderboard[0][objective_key])
    selected_objective = float(selected[objective_key])
    objective_efficiency = (
        selected_objective / best_objective
        if best_objective > 1e-12 else 1.0
    )
    purpose_supported = bool(
        selected["expected_normalized_information_gain"] >= 0.02
        if query["purpose"] == "reduce_opponent_uncertainty"
        else selected["branch_action_switch"]
        and selected["expected_decision_value_of_information"] >= 0.005
    )
    probability_high = float(selected["forecast_high_rate"])
    proposed_high = query["action_if_high"]
    proposed_low = query["action_if_low"]
    high_value = float(selected["action_values_if_high"][proposed_high])
    low_value = float(selected["action_values_if_low"][proposed_low])
    best_high_value = float(selected["action_values_if_high"][
        selected["best_action_if_high"]
    ])
    best_low_value = float(selected["action_values_if_low"][
        selected["best_action_if_low"]
    ])
    high_regret = max(0.0, best_high_value - high_value)
    low_regret = max(0.0, best_low_value - low_value)
    expected_policy_value = (
        probability_high * high_value
        + (1.0 - probability_high) * low_value
    )
    expected_policy_regret = max(
        0.0,
        float(selected["expected_adaptive_value"]) - expected_policy_value,
    )
    weighted_branch_regret = (
        probability_high * high_regret
        + (1.0 - probability_high) * low_regret
    )
    contingent_policy = {
        "action_if_high": proposed_high,
        "action_if_low": proposed_low,
        "model_best_action_if_high": selected["best_action_if_high"],
        "model_best_action_if_low": selected["best_action_if_low"],
        "proposed_value_if_high": high_value,
        "proposed_value_if_low": low_value,
        "regret_if_high": high_regret,
        "regret_if_low": low_regret,
        "expected_policy_value": expected_policy_value,
        "expected_policy_regret": expected_policy_regret,
        "expected_weighted_branch_regret": weighted_branch_regret,
        "expected_regret_identity_error": abs(
            expected_policy_regret - weighted_branch_regret
        ),
        "worst_branch_regret": max(high_regret, low_regret),
        "high_branch_action_aligned": (
            proposed_high == selected["best_action_if_high"]
        ),
        "low_branch_action_aligned": (
            proposed_low == selected["best_action_if_low"]
        ),
        "model_checked_consistent": bool(
            expected_policy_regret <= 0.03 + 1e-12
            and max(high_regret, low_regret) <= 0.05 + 1e-12
            and abs(expected_policy_regret - weighted_branch_regret) <= 1e-9
        ),
        "regret_thresholds_model_owned": True,
    }
    payload = {
        "version": OPPONENT_INFORMATION_QUERY_VERSION,
        "accepted": True,
        "reason": "model_checked_shadow_information_query",
        "query": query,
        "model_evidence": basis,
        "selected_query_report": selected,
        "query_leaderboard": leaderboard,
        "model_recommended_feature": leaderboard[0]["feature"],
        "model_ranking_objective": ranking_objective,
        "selected_query_model_rank": selected["model_rank"],
        "query_ranking_objective_metric": objective_key,
        "selected_query_objective_value": selected_objective,
        "model_best_query_objective_value": best_objective,
        "selected_query_objective_regret": max(
            0.0, best_objective - selected_objective,
        ),
        "selected_query_objective_efficiency": float(np.clip(
            objective_efficiency, 0.0, 1.0,
        )),
        "purpose_supported": purpose_supported,
        "contingent_policy": contingent_policy,
        "query_signature": str(query_signature),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
        "hidden_opponent_intent_observed": False,
    }
    return {**payload, "audit_digest": _digest(payload)}


def evaluate_llm_opponent_information_query(
    packet: dict[str, Any],
    raw_query: Any,
    *,
    selected_action: str,
    query_signature: str = "opponent-information-query-unspecified",
) -> dict[str, Any]:
    query = validate_llm_opponent_information_query(raw_query)
    base = {
        "version": OPPONENT_INFORMATION_QUERY_VERSION,
        "accepted": False,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
        "query_signature": str(query_signature),
    }
    if query is None:
        return {**base, "reason": "missing_or_invalid_information_query"}
    if query["selected_action"] != str(selected_action).lower():
        return {
            **base, "reason": "information_query_action_must_equal_plan",
            "query": query,
        }
    candidates = {
        str(candidate.get("action", "")).lower(): candidate
        for candidate in packet.get("candidates") or []
    }
    selected = candidates.get(query["selected_action"])
    if not packet.get("available") or selected is None:
        return {**base, "reason": "selected_action_evidence_unavailable"}
    if query["horizon"] not in (
        selected.get("multi_horizon_predictions") or {}
    ):
        return {**base, "reason": "query_horizon_not_evaluated"}
    belief = packet.get("opponent_belief") or {}
    current = _posterior(belief.get("posterior"))
    if current is None:
        return {**base, "reason": "opponent_posterior_unavailable"}
    response = _posterior(
        (selected.get("opponent_response_prediction") or {}).get(
            "response_posterior"
        )
    )
    posterior = response if response is not None else current
    values = {
        action: dict(candidates.get(action, {}).get(
            "opponent_hypothesis_values"
        ) or {})
        for action in RESPONSE_ACTIONS
    }
    basis = {
        "posterior": {
            name: float(posterior[index])
            for index, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "posterior_source": (
            "selected_action_response_posterior"
            if response is not None else "current_opponent_belief"
        ),
        "action_hypothesis_values": values,
        "observation_threshold": 0.5,
        "observation_sigma": _OBSERVATION_SIGMA,
        "checkpoint_signature": str(packet.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(packet.get(
            "environment_signature", "environment_unspecified",
        )),
    }
    audit = _audit_from_basis(
        query, basis, query_signature=query_signature,
    )
    return audit or {**base, "reason": "complete_hypothesis_values_required"}


def opponent_information_query_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    query = validate_llm_opponent_information_query(audit.get("query"))
    if query is None:
        return False
    expected = _audit_from_basis(
        query,
        audit.get("model_evidence") or {},
        query_signature=str(audit.get("query_signature", "")),
    )
    return bool(expected is not None and audit == expected)
