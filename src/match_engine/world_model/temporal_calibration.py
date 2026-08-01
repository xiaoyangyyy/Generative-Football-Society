"""Prospective scoring for multi-horizon utility-path dependence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np

from src.match_engine.world_model.temporal_utility import (
    build_temporal_utility_paths,
)
from src.match_engine.world_model.policy_experiment import (
    policy_horizon_seconds,
)


TEMPORAL_PATH_CALIBRATION_VERSION = 1
TEMPORAL_PATH_EVENT_KEYS = (
    "ever_downside",
    "terminal_upside",
    "downside_recovery",
    "positive_to_negative_reversal",
)


def _path_targets(paths: np.ndarray) -> dict[str, Any]:
    baseline = np.zeros((len(paths), 1), dtype=np.float64)
    with_baseline = np.concatenate([baseline, paths], axis=1)
    maximum_drawdown = np.max(
        np.maximum.accumulate(with_baseline, axis=1) - with_baseline,
        axis=1,
    )
    ever_downside = np.any(paths < 0.0, axis=1)
    terminal_upside = paths[:, -1] > 0.0
    recovery = ever_downside & (paths[:, -1] >= 0.0)
    reversal = np.zeros(len(paths), dtype=bool)
    for index in range(1, paths.shape[1]):
        reversal |= (
            np.any(paths[:, :index] > 0.0, axis=1)
            & (paths[:, index] < 0.0)
        )
    return {
        "path_minimum_values": np.min(paths, axis=1),
        "path_maximum_drawdown_values": maximum_drawdown,
        "event_scenario_rates": {
            "ever_downside": float(np.mean(ever_downside)),
            "terminal_upside": float(np.mean(terminal_upside)),
            "downside_recovery": float(np.mean(recovery)),
            "positive_to_negative_reversal": float(np.mean(reversal)),
        },
    }


def _snapshot(report: dict[str, Any]) -> dict[str, Any] | None:
    if not report.get("available") or report.get("temporal_joint_calibrated"):
        return None
    try:
        paths = np.asarray(report["scenario_utility_paths"], dtype=np.float64)
        horizons = [str(value) for value in report["horizon_keys"]]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2
        or paths.shape[1] != len(horizons)
        or not np.all(np.isfinite(paths))
    ):
        return None
    targets = _path_targets(paths)
    return {
        "distribution_scope": str(report.get("distribution_scope", "")),
        "horizon_keys": horizons,
        "scenario_utility_paths": paths.tolist(),
        "scenario_count": int(paths.shape[0]),
        "temporal_coupling_source": str(
            report.get("temporal_coupling_source", "")
        ),
        "temporal_dependence_learned": bool(
            report.get("temporal_dependence_learned")
        ),
        "temporal_dependence_validation": dict(
            report.get("temporal_dependence_validation") or {}
        ),
        "temporal_joint_calibrated": False,
        "event_scenario_rates": targets["event_scenario_rates"],
        "path_minimum_values": list(map(
            float, targets["path_minimum_values"],
        )),
        "path_maximum_drawdown_values": list(map(
            float, targets["path_maximum_drawdown_values"],
        )),
    }


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "temporal-path-forecast:" + hashlib.sha256(encoded).hexdigest()[:24]


def freeze_temporal_path_forecast(
    predictions: dict[str, dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any]:
    """Freeze a path forecast and an honest comonotonic benchmark."""
    report = build_temporal_utility_paths(predictions)
    primary = _snapshot(report)
    base = {
        "version": TEMPORAL_PATH_CALIBRATION_VERSION,
        "available": primary is not None,
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
        "temporal_joint_calibrated": False,
    }
    if primary is None:
        return {
            **base,
            "reason": str(report.get(
                "reason", "temporal_path_forecast_unavailable",
            )),
        }
    benchmark = None
    if primary["temporal_dependence_learned"]:
        fallback_predictions = copy.deepcopy(predictions)
        for prediction in fallback_predictions.values():
            distribution = prediction.get("distributional_policy_utility")
            if isinstance(distribution, dict):
                distribution.pop("temporal_residual_rank_coupling", None)
        benchmark = _snapshot(build_temporal_utility_paths(
            fallback_predictions,
        ))
        if benchmark is None:
            return {
                **base,
                "available": False,
                "reason": "comonotonic_benchmark_unavailable",
            }
    payload = {
        **base,
        "primary": primary,
        "comonotonic_benchmark": benchmark,
        "benchmark_required": bool(primary["temporal_dependence_learned"]),
    }
    return {**payload, "forecast_digest": _digest(payload)}


def _snapshot_is_valid(snapshot: Any) -> bool:
    if not isinstance(snapshot, dict):
        return False
    try:
        paths = np.asarray(
            snapshot["scenario_utility_paths"], dtype=np.float64,
        )
        minimum = np.asarray(snapshot["path_minimum_values"], dtype=np.float64)
        drawdown = np.asarray(
            snapshot["path_maximum_drawdown_values"], dtype=np.float64,
        )
        probabilities = snapshot["event_scenario_rates"]
        declared = np.asarray([
            probabilities[key] for key in TEMPORAL_PATH_EVENT_KEYS
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    try:
        horizons = [str(value) for value in snapshot["horizon_keys"]]
        ordered_horizons = sorted(
            set(horizons),
            key=lambda value: (policy_horizon_seconds(value), value),
        )
    except (KeyError, TypeError, ValueError):
        return False
    if (
        paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2
        or paths.shape[1] != len(horizons)
        or horizons != ordered_horizons
        or int(snapshot.get("scenario_count", 0)) != paths.shape[0]
        or not np.all(np.isfinite(paths))
        or snapshot.get("temporal_joint_calibrated") is not False
        or (snapshot.get("temporal_dependence_validation") or {}).get(
            "empirical_validation_status"
        ) not in {
            "not_applicable", "insufficient_evidence", "validated",
            "degraded",
        }
    ):
        return False
    expected = _path_targets(paths)
    expected_probabilities = np.asarray([
        expected["event_scenario_rates"][key]
        for key in TEMPORAL_PATH_EVENT_KEYS
    ])
    return bool(
        minimum.shape == (len(paths),)
        and drawdown.shape == (len(paths),)
        and declared.shape == (len(TEMPORAL_PATH_EVENT_KEYS),)
        and np.all((declared >= 0.0) & (declared <= 1.0))
        and np.allclose(
            minimum, expected["path_minimum_values"], atol=1e-12, rtol=0.0,
        )
        and np.allclose(
            drawdown, expected["path_maximum_drawdown_values"],
            atol=1e-12, rtol=0.0,
        )
        and np.allclose(
            declared, expected_probabilities, atol=1e-12, rtol=0.0,
        )
    )


def temporal_path_forecast_is_valid(forecast: Any) -> bool:
    if not isinstance(forecast, dict) or not forecast.get("available"):
        return False
    primary = forecast.get("primary")
    benchmark = forecast.get("comonotonic_benchmark")
    learned = bool(
        isinstance(primary, dict)
        and primary.get("temporal_dependence_learned")
    )
    primary_source = (
        str(primary.get("temporal_coupling_source", ""))
        if isinstance(primary, dict) else ""
    )
    primary_scope = (
        str(primary.get("distribution_scope", ""))
        if isinstance(primary, dict) else ""
    )
    validation_status = (
        (primary.get("temporal_dependence_validation") or {}).get(
            "empirical_validation_status"
        ) if isinstance(primary, dict) else None
    )
    payload = {key: value for key, value in forecast.items()
               if key != "forecast_digest"}
    try:
        digest_matches = forecast.get("forecast_digest") == _digest(payload)
    except (TypeError, ValueError):
        return False
    return bool(
        int(forecast.get("version", 0)) == TEMPORAL_PATH_CALIBRATION_VERSION
        and forecast.get("shadow_only")
        and not forecast.get("authority_active")
        and not forecast.get("policy_mutated")
        and not forecast.get("causal_interpretation")
        and forecast.get("temporal_joint_calibrated") is False
        and _snapshot_is_valid(primary)
        and primary_scope in {
            "epistemic_member_only", "calibrated_predictive",
        }
        and (not learned or primary_scope == "calibrated_predictive")
        and (
            not learned or validation_status in {
                "insufficient_evidence", "validated",
            }
        )
        and (
            primary_source
            == "member_identity_x_empirical_residual_rank_templates"
            if learned else primary_source in {
                "member_identity_x_comonotonic_residual_quantile_rank",
                "transition_member_identity",
            }
        )
        and bool(forecast.get("benchmark_required")) == learned
        and (
            _snapshot_is_valid(benchmark)
            and benchmark.get("temporal_dependence_learned") is False
            and benchmark.get("horizon_keys") == primary.get("horizon_keys")
            and benchmark.get("distribution_scope")
            == primary.get("distribution_scope")
            and benchmark.get("temporal_coupling_source")
            == "member_identity_x_comonotonic_residual_quantile_rank"
            if learned else benchmark is None
        )
        and digest_matches
    )


def _empirical_crps(values: np.ndarray, observed: float) -> float:
    return max(0.0, float(
        np.mean(np.abs(values - observed))
        - 0.5 * np.mean(np.abs(values[:, None] - values[None, :]))
    ))


def _score_snapshot(
    snapshot: dict[str, Any], realized: np.ndarray,
) -> dict[str, Any]:
    realized_targets = _path_targets(realized[None, :])
    observed_events = {
        key: bool(realized_targets["event_scenario_rates"][key])
        for key in TEMPORAL_PATH_EVENT_KEYS
    }
    probabilities = snapshot["event_scenario_rates"]
    brier = {
        key: (float(probabilities[key]) - float(observed_events[key])) ** 2
        for key in TEMPORAL_PATH_EVENT_KEYS
    }
    realized_minimum = float(realized_targets["path_minimum_values"][0])
    realized_drawdown = float(
        realized_targets["path_maximum_drawdown_values"][0]
    )
    return {
        "event_scenario_rates": dict(probabilities),
        "observed_events": observed_events,
        "event_brier_scores": brier,
        "mean_event_brier_score": float(np.mean(list(brier.values()))),
        "realized_path_minimum_utility": realized_minimum,
        "path_minimum_crps": _empirical_crps(
            np.asarray(snapshot["path_minimum_values"], dtype=np.float64),
            realized_minimum,
        ),
        "realized_maximum_drawdown": realized_drawdown,
        "maximum_drawdown_crps": _empirical_crps(
            np.asarray(
                snapshot["path_maximum_drawdown_values"], dtype=np.float64,
            ),
            realized_drawdown,
        ),
    }


def score_temporal_path_forecast(
    forecast: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any] | None:
    if not temporal_path_forecast_is_valid(forecast):
        return None
    horizons = forecast["primary"]["horizon_keys"]
    try:
        realized = np.asarray([
            float(outcomes[horizon]["policy_utility"])
            for horizon in horizons
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if realized.shape != (len(horizons),) or not np.all(np.isfinite(realized)):
        return None
    primary = _score_snapshot(forecast["primary"], realized)
    benchmark = (
        _score_snapshot(forecast["comonotonic_benchmark"], realized)
        if forecast["comonotonic_benchmark"] is not None else None
    )
    return {
        "version": TEMPORAL_PATH_CALIBRATION_VERSION,
        "forecast_digest": forecast["forecast_digest"],
        "horizon_keys": list(horizons),
        "realized_utility_path": list(map(float, realized)),
        "temporal_coupling_source": forecast["primary"][
            "temporal_coupling_source"
        ],
        "temporal_dependence_learned": bool(
            forecast["primary"]["temporal_dependence_learned"]
        ),
        "temporal_dependence_validation": dict(
            forecast["primary"]["temporal_dependence_validation"]
        ),
        "primary_score": primary,
        "comonotonic_benchmark_score": benchmark,
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "issuance_evaluation_provenance_compatible": bool(
            str(checkpoint_signature) == forecast["checkpoint_signature"]
            and str(environment_signature) == forecast["environment_signature"]
        ),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
        "joint_temporal_probability_claimed": False,
    }


def temporal_path_score_is_valid(
    score: Any, forecast: dict[str, Any],
) -> bool:
    if not isinstance(score, dict) or not temporal_path_forecast_is_valid(
        forecast
    ):
        return False
    outcomes = {
        horizon: {"policy_utility": value}
        for horizon, value in zip(
            score.get("horizon_keys") or [],
            score.get("realized_utility_path") or [],
        )
    }
    expected = score_temporal_path_forecast(
        forecast,
        outcomes,
        checkpoint_signature=str(score.get("checkpoint_signature", "")),
        environment_signature=str(score.get("environment_signature", "")),
    )
    return bool(expected is not None and score == expected)


def attach_temporal_path_evaluation(record: dict[str, Any]) -> None:
    if isinstance(record.get("world_model_temporal_path_evaluation"), dict):
        return
    forecast = record.get("world_model_temporal_path_forecast") or {}
    score = score_temporal_path_forecast(
        forecast,
        record.get("multi_horizon_regime_outcomes") or {},
        checkpoint_signature=str(record.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        environment_signature=str(record.get(
            "environment_signature", "environment_unspecified",
        )),
    )
    if score is not None:
        record["world_model_temporal_path_evaluation"] = score
