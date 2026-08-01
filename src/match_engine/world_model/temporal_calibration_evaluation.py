"""Cross-match validation for prospectively frozen temporal path forecasts."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.temporal_calibration import (
    TEMPORAL_PATH_CALIBRATION_VERSION,
    TEMPORAL_PATH_EVENT_KEYS,
    temporal_path_score_is_valid,
)


MIN_EMPIRICAL_VALIDATION_PATHS = 8
MIN_EMPIRICAL_VALIDATION_MATCHES = 4
MAX_EVENT_CALIBRATION_GAP = 0.25
MAX_SCORE_REGRET_VS_FALLBACK = 0.02


def temporal_path_calibration_diagnostics(
    match_logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str | None = None,
    environment_signature: str | None = None,
) -> dict[str, Any]:
    grouped: list[list[dict[str, Any]]] = []
    issued = eligible = missing = malformed = 0
    for payload in match_logs:
        match_rows = []
        records = (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        for record in records:
            if checkpoint_signature is not None and str(record.get(
                "checkpoint_signature", "",
            )) != str(checkpoint_signature):
                continue
            if environment_signature is not None and str(record.get(
                "environment_signature", "environment_unspecified",
            )) != str(environment_signature):
                continue
            forecast = record.get("world_model_temporal_path_forecast")
            if not isinstance(forecast, dict) or not forecast.get("available"):
                continue
            issued += 1
            horizons = (forecast.get("primary") or {}).get("horizon_keys") or []
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            if not horizons or not all(horizon in outcomes for horizon in horizons):
                continue
            eligible += 1
            score = record.get("world_model_temporal_path_evaluation")
            if not isinstance(score, dict):
                missing += 1
            elif temporal_path_score_is_valid(score, forecast):
                match_rows.append(score)
            else:
                malformed += 1
        if match_rows:
            grouped.append(match_rows)
    valid = [row for rows in grouped for row in rows]
    empirical = [row for row in valid if row["temporal_dependence_learned"]]
    empirical_matches = sum(
        any(row["temporal_dependence_learned"] for row in rows)
        for rows in grouped
    )

    def clustered(rows_filter, path: tuple[str, ...]) -> float:
        values = []
        for rows in grouped:
            selected = [row for row in rows if rows_filter(row)]
            if not selected:
                continue
            samples = []
            for row in selected:
                value: Any = row
                for key in path:
                    value = value[key]
                samples.append(float(value))
            values.append(float(np.mean(samples)))
        return float(np.mean(values)) if values else 0.0

    empirical_filter = lambda row: bool(row["temporal_dependence_learned"])
    primary_brier = clustered(
        empirical_filter, ("primary_score", "mean_event_brier_score"),
    )
    benchmark_brier = clustered(
        empirical_filter,
        ("comonotonic_benchmark_score", "mean_event_brier_score"),
    )
    primary_min_crps = clustered(
        empirical_filter, ("primary_score", "path_minimum_crps"),
    )
    benchmark_min_crps = clustered(
        empirical_filter,
        ("comonotonic_benchmark_score", "path_minimum_crps"),
    )
    primary_drawdown_crps = clustered(
        empirical_filter, ("primary_score", "maximum_drawdown_crps"),
    )
    benchmark_drawdown_crps = clustered(
        empirical_filter,
        ("comonotonic_benchmark_score", "maximum_drawdown_crps"),
    )
    event_gaps = []
    for key in TEMPORAL_PATH_EVENT_KEYS:
        if not empirical:
            break
        match_predictions = []
        match_observations = []
        for rows in grouped:
            selected = [row for row in rows if empirical_filter(row)]
            if not selected:
                continue
            match_predictions.append(float(np.mean([
                row["primary_score"]["event_scenario_rates"][key]
                for row in selected
            ])))
            match_observations.append(float(np.mean([
                row["primary_score"]["observed_events"][key]
                for row in selected
            ])))
        predicted = np.mean(match_predictions)
        observed = np.mean(match_observations)
        event_gaps.append(abs(float(predicted) - float(observed)))
    mean_gap = float(np.mean(event_gaps)) if event_gaps else 0.0
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "runtime_unspecified", "environment_unspecified",
    }
    provenance_compatible = bool(
        len(scopes) == 1
        and all(value not in unspecified for value in scopes[0])
        and all(
            row.get("issuance_evaluation_provenance_compatible")
            for row in valid
        )
    )
    if (
        len(empirical) < MIN_EMPIRICAL_VALIDATION_PATHS
        or empirical_matches < MIN_EMPIRICAL_VALIDATION_MATCHES
    ):
        status = "insufficient_evidence"
    elif (
        provenance_compatible
        and primary_brier <= benchmark_brier + MAX_SCORE_REGRET_VS_FALLBACK
        and primary_min_crps <= benchmark_min_crps + MAX_SCORE_REGRET_VS_FALLBACK
        and primary_drawdown_crps
        <= benchmark_drawdown_crps + MAX_SCORE_REGRET_VS_FALLBACK
        and mean_gap <= MAX_EVENT_CALIBRATION_GAP
    ):
        status = "validated"
    else:
        status = "degraded"
    return {
        "version": TEMPORAL_PATH_CALIBRATION_VERSION,
        "evaluation_kind": "realized_multi_horizon_path_calibration",
        "issued_forecasts": issued,
        "eligible_realized_paths": eligible,
        "scored_paths": len(valid),
        "matches": len(grouped),
        "missing_scores": missing,
        "malformed_scores": malformed,
        "empirical_temporal_paths": len(empirical),
        "empirical_temporal_matches": empirical_matches,
        "empirical_validation_status": status,
        "minimum_empirical_validation_paths": MIN_EMPIRICAL_VALIDATION_PATHS,
        "minimum_empirical_validation_matches": MIN_EMPIRICAL_VALIDATION_MATCHES,
        "match_clustered_empirical_mean_event_brier": primary_brier,
        "match_clustered_comonotonic_mean_event_brier": benchmark_brier,
        "match_clustered_empirical_path_minimum_crps": primary_min_crps,
        "match_clustered_comonotonic_path_minimum_crps": benchmark_min_crps,
        "match_clustered_empirical_maximum_drawdown_crps": (
            primary_drawdown_crps
        ),
        "match_clustered_comonotonic_maximum_drawdown_crps": (
            benchmark_drawdown_crps
        ),
        "empirical_mean_event_calibration_gap": mean_gap,
        "all_shadow_only": all(row.get("shadow_only") for row in valid),
        "all_non_controlling": all(
            not row.get("authority_active") and not row.get("policy_mutated")
            for row in valid
        ),
        "all_non_causal": all(
            not row.get("causal_interpretation") for row in valid
        ),
        "all_joint_probability_claims_disabled": all(
            not row.get("joint_temporal_probability_claimed") for row in valid
        ),
        "all_provenance_compatible": all(
            row.get("issuance_evaluation_provenance_compatible")
            for row in valid
        ),
        "provenance_compatible": provenance_compatible,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
