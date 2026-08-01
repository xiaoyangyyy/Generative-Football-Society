"""Realized scoring and cross-match diagnostics for LLM risk preferences."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.risk_preference import (
    RISK_PREFERENCE_VERSION,
    prospect_value,
    risk_preference_audit_is_valid,
)


def _empirical_crps(values: np.ndarray, observed: float) -> float:
    return max(0.0, float(
        np.mean(np.abs(values - observed))
        - 0.5 * np.mean(np.abs(values[:, None] - values[None, :]))
    ))


def score_llm_risk_preference(
    context: dict[str, Any],
    outcome: dict[str, Any],
    *,
    horizon: str,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any] | None:
    if not risk_preference_audit_is_valid(context):
        return None
    preference = context["preference"]
    horizon = str(horizon)
    if horizon not in preference["horizon_weights"]:
        return None
    evidence = context["scenario_evidence"][horizon][
        preference["selected_action"]
    ]
    try:
        observed = float(outcome["policy_utility"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(observed):
        return None
    raw = np.asarray(evidence["decision_scenario_values"], dtype=np.float64)
    transformed = prospect_value(raw, preference)
    observed_value = float(prospect_value([observed], preference)[0])
    mean = float(np.mean(transformed))
    quantiles = np.quantile(
        transformed, [0.10, 0.50, 0.90], method="linear",
    )
    return {
        "version": RISK_PREFERENCE_VERSION,
        "horizon": horizon,
        "preference": preference,
        "audit_digest": str(context["audit_digest"]),
        "raw_scenario_values": list(map(float, raw)),
        "transformed_scenario_values": list(map(float, transformed)),
        "observed_policy_utility": observed,
        "observed_preference_value": observed_value,
        "predicted_mean_preference_value": mean,
        "preference_value_absolute_error": abs(mean - observed_value),
        "preference_value_squared_error": (mean - observed_value) ** 2,
        "preference_value_crps": _empirical_crps(
            transformed, observed_value,
        ),
        "preference_value_quantiles": {
            "q10": float(quantiles[0]), "q50": float(quantiles[1]),
            "q90": float(quantiles[2]),
        },
        "central_80_covered": bool(
            quantiles[0] <= observed_value <= quantiles[2]
        ),
        "prospective_preference_consistent": bool(
            context["preference_consistent"]
        ),
        "prospective_preference_robust": bool(
            context["preference_robust"]
        ),
        "distribution_scope": preference["distribution_scope"],
        "predictive_distribution_calibrated": (
            preference["distribution_scope"] == "calibrated_predictive"
        ),
        "paired_same_action_horizon": True,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
        "preference_signature": str(context.get("preference_signature", "")),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "issuance_evaluation_provenance_compatible": bool(
            str(context.get("checkpoint_signature", ""))
            == str(checkpoint_signature)
            and str(context.get("environment_signature", ""))
            == str(environment_signature)
        ),
    }


def _valid_score(row: dict[str, Any], context: dict[str, Any]) -> bool:
    if not risk_preference_audit_is_valid(context):
        return False
    try:
        raw = np.asarray(row["raw_scenario_values"], dtype=np.float64)
        transformed = np.asarray(
            row["transformed_scenario_values"], dtype=np.float64,
        )
        observed = float(row["observed_policy_utility"])
        observed_value = float(row["observed_preference_value"])
        mean = float(row["predicted_mean_preference_value"])
        mae = float(row["preference_value_absolute_error"])
        mse = float(row["preference_value_squared_error"])
        crps = float(row["preference_value_crps"])
        quantiles = row["preference_value_quantiles"]
        declared_quantiles = np.asarray([
            quantiles["q10"], quantiles["q50"], quantiles["q90"],
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    preference = context["preference"]
    horizon = str(row.get("horizon", ""))
    if horizon not in preference["horizon_weights"]:
        return False
    expected_raw = np.asarray(
        context["scenario_evidence"][horizon][
            preference["selected_action"]
        ]["decision_scenario_values"], dtype=np.float64,
    )
    expected_transformed = prospect_value(expected_raw, preference)
    expected_observed = float(prospect_value([observed], preference)[0])
    expected_quantiles = np.quantile(
        expected_transformed, [0.10, 0.50, 0.90], method="linear",
    )
    if (
        raw.shape != expected_raw.shape
        or transformed.shape != expected_transformed.shape
    ):
        return False
    return bool(
        int(row.get("version", 0)) == RISK_PREFERENCE_VERSION
        and np.allclose(raw, expected_raw, atol=1e-9, rtol=0.0)
        and np.allclose(
            transformed, expected_transformed, atol=1e-9, rtol=0.0,
        )
        and abs(observed_value - expected_observed) <= 1e-9
        and abs(mean - float(np.mean(transformed))) <= 1e-9
        and abs(mae - abs(mean - observed_value)) <= 1e-9
        and abs(mse - (mean - observed_value) ** 2) <= 1e-9
        and abs(crps - _empirical_crps(transformed, observed_value)) <= 1e-9
        and np.allclose(
            declared_quantiles, expected_quantiles, atol=1e-9, rtol=0.0,
        )
        and bool(row.get("central_80_covered")) == bool(
            expected_quantiles[0] <= observed_value <= expected_quantiles[2]
        )
        and row.get("audit_digest") == context.get("audit_digest")
        and bool(row.get("prospective_preference_consistent"))
        == bool(context.get("preference_consistent"))
        and bool(row.get("prospective_preference_robust"))
        == bool(context.get("preference_robust"))
        and str(row.get("distribution_scope", ""))
        == preference["distribution_scope"]
        and row.get("paired_same_action_horizon")
        and row.get("shadow_only") and not row.get("authority_active")
        and not row.get("policy_mutated")
        and not row.get("causal_interpretation")
        and row.get("issuance_evaluation_provenance_compatible")
    )


def risk_preference_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    eligible = unscored = 0
    malformed_contexts = 0
    for payload in match_logs:
        match_rows = []
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        ):
            context = record.get("llm_risk_preference_context") or {}
            if context and not risk_preference_audit_is_valid(context):
                malformed_contexts += 1
                continue
            if not context.get("accepted"):
                continue
            preference = context["preference"]
            if preference["selected_action"] != str(
                record.get("intervention_actual_action", "")
            ).lower():
                continue
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            for horizon in preference["horizon_weights"]:
                if horizon not in outcomes:
                    continue
                eligible += 1
                score = (outcomes[horizon] or {}).get(
                    "llm_risk_preference_evaluation"
                )
                if not isinstance(score, dict):
                    unscored += 1
                else:
                    match_rows.append((score, context))
        if match_rows:
            grouped.append(match_rows)
    valid_groups = []
    malformed_scores = 0
    for rows in grouped:
        valid_rows = []
        for row, context in rows:
            if _valid_score(row, context):
                valid_rows.append(row)
            else:
                malformed_scores += 1
        if valid_rows:
            valid_groups.append(valid_rows)
    valid = [row for rows in valid_groups for row in rows]

    def match_metric(key: str) -> list[float]:
        return [
            float(np.mean([float(row[key]) for row in rows]))
            for rows in valid_groups
        ]

    def match_rate(key: str) -> list[float]:
        return [
            float(np.mean([bool(row[key]) for row in rows]))
            for rows in valid_groups
        ]

    crps = match_metric("preference_value_crps")
    mae = match_metric("preference_value_absolute_error")
    coverage = match_rate("central_80_covered")
    consistent = match_rate("prospective_preference_consistent")
    robust = match_rate("prospective_preference_robust")
    signatures = sorted({
        str(row.get("preference_signature", "")) for row in valid
    })
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "risk-preference-contract-unspecified",
        "runtime_unspecified", "environment_unspecified",
    }
    return {
        "version": RISK_PREFERENCE_VERSION,
        "evaluation_kind": "realized_multi_horizon_prospect_preference",
        "realized_preference_values": len(valid),
        "eligible_preference_horizons": eligible,
        "unscored_eligible_preference_horizons": unscored,
        "matches": len(valid_groups),
        "malformed_preference_contexts": malformed_contexts,
        "malformed_preference_evaluations": malformed_scores,
        "match_clustered_preference_value_crps": (
            float(np.mean(crps)) if crps else 0.0
        ),
        "match_clustered_preference_value_mae": (
            float(np.mean(mae)) if mae else 0.0
        ),
        "match_clustered_central_80_coverage": (
            float(np.mean(coverage)) if coverage else 0.0
        ),
        "match_clustered_prospective_consistency": (
            float(np.mean(consistent)) if consistent else 0.0
        ),
        "match_clustered_prospective_robustness": (
            float(np.mean(robust)) if robust else 0.0
        ),
        "all_predictive_distributions_calibrated": all(
            row.get("predictive_distribution_calibrated") for row in valid
        ),
        "all_shadow_only": all(row.get("shadow_only") for row in valid),
        "all_non_controlling": all(
            not row.get("authority_active") and not row.get("policy_mutated")
            for row in valid
        ),
        "all_non_causal": all(
            not row.get("causal_interpretation") for row in valid
        ),
        "provenance_compatible": bool(
            len(signatures) == 1 and signatures[0] not in unspecified
            and len(scopes) == 1
            and all(value not in unspecified for value in scopes[0])
        ),
        "preference_signatures": signatures,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
