"""Calibrate multi-horizon policy-utility forecasts against realized outcomes."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.policy_experiment import policy_horizon_seconds


def _prediction_rows(
    records: Iterable[dict[str, Any]],
    *,
    outcome_family: str = "regime",
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        isolated = record.get("multi_horizon_outcomes") or {}
        regime = record.get("multi_horizon_regime_outcomes") or isolated
        outcomes = regime if outcome_family == "regime" else isolated
        for horizon_key, outcome in outcomes.items():
            prediction = outcome.get("world_model_prediction") or {}
            if "policy_utility" not in prediction:
                continue
            try:
                predicted = float(prediction["policy_utility"])
                actual = float(outcome["policy_utility"])
                uncertainty = float(prediction.get("uncertainty", 1.0))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (
                predicted, actual, uncertainty,
            )):
                continue
            rows.append({
                "action": str(record.get("intervention_actual_action", "other")),
                "horizon_key": str(horizon_key),
                "predicted": predicted,
                "actual": actual,
                "uncertainty": float(np.clip(uncertainty, 0.05, 1.0)),
            })
    return rows


def _group_metrics(
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
) -> dict[str, Any]:
    samples = len(rows)
    if not rows:
        return {
            "samples": 0,
            "active": False,
            "trust_factor": 1.0,
        }
    predicted = np.asarray([row["predicted"] for row in rows], dtype=float)
    actual = np.asarray([row["actual"] for row in rows], dtype=float)
    uncertainty = np.asarray([row["uncertainty"] for row in rows], dtype=float)
    residual = actual - predicted
    mse = float(np.mean(np.square(residual)))
    rmse = math.sqrt(mse)
    baseline_mse = float(np.mean(np.square(actual)))
    skill = 1.0 - mse / max(baseline_mse, 1e-6)
    calibration_ratio = rmse / max(0.05, float(np.mean(uncertainty)))
    active = samples >= max(2, int(min_samples))
    if active:
        error_factor = 1.0 / math.sqrt(max(1.0, calibration_ratio))
        bias_factor = 1.0 / math.sqrt(
            max(1.0, abs(float(np.mean(residual))) / 0.10)
        )
        skill_factor = 1.0 if skill >= 0.0 else math.exp(max(-4.0, skill))
        trust = float(np.clip(
            min(error_factor, bias_factor, skill_factor), 0.25, 1.0,
        ))
    else:
        trust = 1.0
    return {
        "samples": samples,
        "active": active,
        "mean_prediction": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
        "mean_residual": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "mean_uncertainty": float(np.mean(uncertainty)),
        "uncertainty_error_ratio": calibration_ratio,
        "coverage_90": float(np.mean(
            np.abs(residual) <= 1.645 * uncertainty
        )),
        "skill_vs_zero_prediction": skill,
        "trust_factor": trust,
    }


def policy_outcome_calibration(
    records: Iterable[dict[str, Any]],
    *,
    min_samples: int = 6,
    outcome_family: str = "regime",
) -> dict[str, Any]:
    rows = _prediction_rows(records, outcome_family=outcome_family)
    horizon_keys = sorted(
        {row["horizon_key"] for row in rows},
        key=policy_horizon_seconds,
    )
    actions = sorted({row["action"] for row in rows})
    by_horizon = {
        horizon: _group_metrics(
            [row for row in rows if row["horizon_key"] == horizon],
            min_samples=min_samples,
        )
        for horizon in horizon_keys
    }
    by_action_horizon = {
        action: {
            horizon: _group_metrics(
                [
                    row for row in rows
                    if row["action"] == action
                    and row["horizon_key"] == horizon
                ],
                min_samples=min_samples,
            )
            for horizon in horizon_keys
        }
        for action in actions
    }
    return {
        "version": 1,
        "calibration_target": "realized_multi_horizon_policy_utility",
        "outcome_family": outcome_family,
        "minimum_samples": max(2, int(min_samples)),
        "samples": len(rows),
        "overall": _group_metrics(rows, min_samples=min_samples),
        "by_horizon": by_horizon,
        "by_action_horizon": by_action_horizon,
        "policy": (
            "Residual calibration may only reduce forecast-derived trust; "
            "it cannot reopen a closed world-model quality gate."
        ),
    }


def outcome_prediction_trust_factor(
    state,
    *,
    action: str,
    horizon_keys: Iterable[str],
    min_samples: int = 6,
) -> tuple[float, dict[str, Any]]:
    records = list(getattr(state, "_wm_coach_decision_adoption", None) or [])
    calibration = policy_outcome_calibration(
        records, min_samples=min_samples, outcome_family="regime",
    )
    ordered = sorted(set(horizon_keys), key=policy_horizon_seconds, reverse=True)
    action_groups = calibration["by_action_horizon"].get(str(action), {})
    for horizon in ordered:
        group = action_groups.get(horizon)
        if group and group["active"]:
            return float(group["trust_factor"]), {
                "scope": "action_horizon",
                "action": str(action),
                "horizon_key": horizon,
                **group,
            }
    for horizon in ordered:
        group = calibration["by_horizon"].get(horizon)
        if group and group["active"]:
            return float(group["trust_factor"]), {
                "scope": "horizon",
                "action": str(action),
                "horizon_key": horizon,
                **group,
            }
    return 1.0, {
        "scope": "insufficient_history",
        "action": str(action),
        "samples": calibration["samples"],
        "trust_factor": 1.0,
    }
