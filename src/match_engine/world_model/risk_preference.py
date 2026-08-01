"""Model-check bounded LLM risk preferences over multi-horizon scenarios."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS
from src.match_engine.world_model.predictive_distribution import (
    predictive_lattice_is_valid,
)


RISK_PREFERENCE_VERSION = 1
_DISTRIBUTION_SCOPES = {"epistemic_member_only", "calibrated_predictive"}


def llm_risk_preference_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": RISK_PREFERENCE_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "shadow_multi_horizon_scenario_risk_preference",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-risk-preference:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_risk_preference(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    selected = str(raw.get("selected_action", "")).strip().lower()
    scope = str(raw.get("distribution_scope", "")).strip().lower()
    weights = raw.get("horizon_weights")
    try:
        loss_aversion = float(raw.get("loss_aversion", 0.0))
        sensitivity = float(raw.get("diminishing_sensitivity", 0.0))
        max_regret = float(raw.get("max_acceptable_regret", -1.0))
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not isinstance(weights, dict) or not 1 <= len(weights) <= 4:
        return None
    normalized_weights: dict[str, float] = {}
    try:
        for raw_horizon, raw_weight in weights.items():
            horizon = str(raw_horizon).strip().lower()
            weight = float(raw_weight)
            if (
                not horizon or len(horizon) > 24 or horizon in normalized_weights
                or not np.isfinite(weight) or not 0.05 <= weight <= 1.0
            ):
                return None
            normalized_weights[horizon] = weight
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        selected not in RESPONSE_ACTIONS
        or scope not in _DISTRIBUTION_SCOPES
        or not all(np.isfinite(value) for value in (
            loss_aversion, sensitivity, max_regret, confidence,
        ))
        or not 1.0 <= loss_aversion <= 4.0
        or not 0.50 <= sensitivity <= 1.0
        or not 0.0 <= max_regret <= 0.50
        or not 0.5 <= confidence <= 1.0
        or abs(sum(normalized_weights.values()) - 1.0) > 1e-6
    ):
        return None
    return {
        "selected_action": selected,
        "distribution_scope": scope,
        "horizon_weights": dict(sorted(normalized_weights.items())),
        "reference_utility": 0.0,
        "loss_aversion": loss_aversion,
        "diminishing_sensitivity": sensitivity,
        "max_acceptable_regret": max_regret,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def prospect_value(values: Any, preference: dict[str, Any]) -> np.ndarray:
    """Apply the declared bounded prospect-value transform elementwise."""
    array = np.asarray(values, dtype=np.float64)
    alpha = float(preference["diminishing_sensitivity"])
    loss_aversion = float(preference["loss_aversion"])
    gains = np.power(np.maximum(array, 0.0), alpha)
    losses = -loss_aversion * np.power(np.maximum(-array, 0.0), alpha)
    return np.where(array >= 0.0, gains, losses)


def _scenario_evidence(distribution: dict[str, Any]) -> dict[str, Any] | None:
    try:
        values = np.asarray(
            distribution["decision_scenario_values"], dtype=np.float64,
        )
        mean = float(distribution["mean_utility"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    scope = str(distribution.get("distribution_scope", ""))
    if (
        not distribution.get("available")
        or not distribution.get("counts_trusted")
        or not distribution.get("member_identity_preserved")
        or scope not in _DISTRIBUTION_SCOPES
        or values.ndim != 1 or len(values) < 2
        or not np.all(np.isfinite(values)) or not np.isfinite(mean)
        or abs(mean - float(np.mean(values))) > 1e-6
    ):
        return None
    evidence = {
        "distribution_scope": scope,
        "decision_scenario_values": list(map(float, values)),
    }
    if scope == "calibrated_predictive":
        keys = (
            "predictive_distribution_available", "epistemic_member_values",
            "residual_scenario_offsets", "residual_calibration_samples",
            "residual_quantiles_split", "uncertainty_decomposition",
        )
        evidence.update({key: distribution.get(key) for key in keys})
        if not predictive_lattice_is_valid(evidence):
            return None
    else:
        try:
            members = np.asarray(
                distribution["member_values"], dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if (
            members.shape != values.shape
            or not np.allclose(members, values, atol=1e-9, rtol=0.0)
        ):
            return None
        evidence["member_values"] = list(map(float, members))
    return evidence


def _compute_results(
    preference: dict[str, Any],
    evidence: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    horizons = preference["horizon_weights"]
    if set(evidence) != set(horizons):
        return None
    horizon_reports: dict[str, Any] = {}
    aggregate = {action: 0.0 for action in RESPONSE_ACTIONS}
    for horizon, weight in horizons.items():
        action_evidence = evidence.get(horizon)
        if not isinstance(action_evidence, dict) or set(action_evidence) != set(
            RESPONSE_ACTIONS
        ):
            return None
        action_values: dict[str, float] = {}
        for action in RESPONSE_ACTIONS:
            item = action_evidence[action]
            if str(item.get("distribution_scope", "")) != preference[
                "distribution_scope"
            ]:
                return None
            raw_values = np.asarray(
                item.get("decision_scenario_values"), dtype=np.float64,
            )
            if (
                raw_values.ndim != 1 or len(raw_values) < 2
                or not np.all(np.isfinite(raw_values))
            ):
                return None
            if (
                preference["distribution_scope"] == "calibrated_predictive"
                and not predictive_lattice_is_valid(item)
            ):
                return None
            if preference["distribution_scope"] == "epistemic_member_only":
                member_values = np.asarray(
                    item.get("member_values"), dtype=np.float64,
                )
                if (
                    member_values.shape != raw_values.shape
                    or not np.allclose(
                        member_values, raw_values, atol=1e-9, rtol=0.0,
                    )
                ):
                    return None
            transformed = prospect_value(raw_values, preference)
            if not np.all(np.isfinite(transformed)):
                return None
            expected_value = float(np.mean(transformed))
            action_values[action] = expected_value
            aggregate[action] += float(weight) * expected_value
        best = max(action_values.values())
        selected_value = action_values[preference["selected_action"]]
        regret = max(0.0, best - selected_value)
        horizon_reports[horizon] = {
            "weight": float(weight),
            "action_expected_preference_values": action_values,
            "preferred_actions": sorted(
                action for action, value in action_values.items()
                if abs(value - best) <= 1e-12
            ),
            "selected_expected_preference_value": selected_value,
            "selected_preference_regret": regret,
            "within_declared_regret": regret <= (
                preference["max_acceptable_regret"] + 1e-12
            ),
        }
    best_aggregate = max(aggregate.values())
    selected_aggregate = aggregate[preference["selected_action"]]
    aggregate_regret = max(0.0, best_aggregate - selected_aggregate)
    return {
        "horizons": horizon_reports,
        "aggregate_action_expected_preference_values": aggregate,
        "aggregate_preferred_actions": sorted(
            action for action, value in aggregate.items()
            if abs(value - best_aggregate) <= 1e-12
        ),
        "selected_aggregate_expected_preference_value": selected_aggregate,
        "selected_aggregate_preference_regret": aggregate_regret,
        "aggregate_within_declared_regret": aggregate_regret <= (
            preference["max_acceptable_regret"] + 1e-12
        ),
        "all_horizons_within_declared_regret": all(
            report["within_declared_regret"]
            for report in horizon_reports.values()
        ),
    }


def _audit_digest(
    preference: dict[str, Any],
    evidence: dict[str, Any],
    results: dict[str, Any],
) -> str:
    encoded = json.dumps(
        {"preference": preference, "scenario_evidence": evidence,
         "results": results},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "risk-preference-audit:" + hashlib.sha256(encoded).hexdigest()[:24]


def risk_preference_audit_is_valid(audit: dict[str, Any]) -> bool:
    if not isinstance(audit, dict):
        return False
    try:
        preference = validate_llm_risk_preference(audit.get("preference"))
        evidence = audit.get("scenario_evidence")
        if preference is None or not isinstance(evidence, dict):
            return False
        expected = _compute_results(preference, evidence)
        if expected is None:
            return False
        expected_consistency = bool(
            expected["aggregate_within_declared_regret"]
            and expected["all_horizons_within_declared_regret"]
        )
        return bool(
            audit.get("accepted")
            and audit.get("shadow_only")
            and not audit.get("authority_active")
            and not audit.get("policy_mutated")
            and audit.get("results") == expected
            and bool(audit.get("preference_consistent"))
            == expected_consistency
            and audit.get("audit_digest") == _audit_digest(
                preference, evidence, expected,
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def evaluate_llm_risk_preference(
    packet: dict[str, Any],
    raw_preference: Any,
    *,
    selected_action: str,
    preference_signature: str = "risk-preference-contract-unspecified",
) -> dict[str, Any]:
    preference = validate_llm_risk_preference(raw_preference)
    base = {
        "version": RISK_PREFERENCE_VERSION,
        "accepted": False,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_selected_action": False,
        "causal_interpretation": False,
        "preference_signature": str(preference_signature),
    }
    if preference is None:
        return {**base, "reason": "missing_or_invalid_risk_preference"}
    if preference["selected_action"] != str(selected_action).lower():
        return {
            **base, "reason": "risk_preference_action_must_equal_plan",
            "preference": preference,
        }
    candidates = {
        str(candidate.get("action", "")).lower(): candidate
        for candidate in packet.get("candidates") or []
    }
    if not packet.get("available") or not all(
        action in candidates for action in RESPONSE_ACTIONS
    ):
        return {
            **base, "reason": "complete_action_set_required",
            "preference": preference,
        }
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for horizon in preference["horizon_weights"]:
        evidence[horizon] = {}
        for action in RESPONSE_ACTIONS:
            prediction = (
                candidates[action].get("multi_horizon_predictions") or {}
            ).get(horizon) or {}
            item = _scenario_evidence(
                prediction.get("distributional_policy_utility") or {}
            )
            if item is None:
                return {
                    **base, "reason": "risk_preference_scenarios_unavailable",
                    "preference": preference, "failed_horizon": horizon,
                    "failed_action": action,
                }
            evidence[horizon][action] = item
    results = _compute_results(preference, evidence)
    if results is None:
        return {
            **base, "reason": "risk_preference_scope_or_evidence_mismatch",
            "preference": preference,
        }
    audit = {
        **base,
        "accepted": True,
        "reason": "shadow_world_model_recomputed_risk_preference",
        "preference": preference,
        "scenario_evidence": evidence,
        "results": results,
        "preference_consistent": bool(
            results["aggregate_within_declared_regret"]
            and results["all_horizons_within_declared_regret"]
        ),
        "checkpoint_signature": str(packet.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(packet.get(
            "environment_signature", "environment_unspecified",
        )),
    }
    audit["audit_digest"] = _audit_digest(preference, evidence, results)
    packet["llm_risk_preference_audit"] = audit
    return audit
