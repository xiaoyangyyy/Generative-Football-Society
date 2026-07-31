"""Residual calibration turns realized policy outcomes into bounded trust."""

from types import SimpleNamespace

import pytest

from src.match_engine.world_model.outcome_calibration import (
    outcome_prediction_trust_factor,
    policy_outcome_calibration,
)


def _record(action, predictions, actuals, uncertainty=0.1):
    outcomes = {
        horizon: {
            "policy_utility": actual,
            "world_model_prediction": {
                "policy_utility": predictions[horizon],
                "uncertainty": uncertainty,
            },
        }
        for horizon, actual in actuals.items()
    }
    return {
        "intervention_actual_action": action,
        "multi_horizon_regime_outcomes": outcomes,
    }


def test_calibration_reports_action_horizon_residual_quality():
    records = [
        _record("pass", {"transition": 0.1}, {"transition": 0.1})
        for _ in range(4)
    ]
    report = policy_outcome_calibration(records, min_samples=4)
    group = report["by_action_horizon"]["pass"]["transition"]
    assert group["active"]
    assert group["rmse"] == pytest.approx(0.0)
    assert group["coverage_90"] == 1.0
    assert group["trust_factor"] == 1.0


def test_bad_long_horizon_predictions_only_reduce_trust():
    records = [
        _record(
            "shot",
            {"transition": 0.0, "60s": 1.0},
            {"transition": 0.0, "60s": 0.0},
            uncertainty=0.1,
        )
        for _ in range(4)
    ]
    state = SimpleNamespace(_wm_coach_decision_adoption=records)
    factor, audit = outcome_prediction_trust_factor(
        state,
        action="shot",
        horizon_keys=("transition", "60s"),
        min_samples=4,
    )
    assert factor == 0.25
    assert audit["scope"] == "action_horizon"
    assert audit["horizon_key"] == "60s"
    assert audit["skill_vs_zero_prediction"] < 0.0


def test_insufficient_residual_history_is_neutral():
    state = SimpleNamespace(_wm_coach_decision_adoption=[])
    factor, audit = outcome_prediction_trust_factor(
        state,
        action="pass",
        horizon_keys=("transition", "60s"),
        min_samples=4,
    )
    assert factor == 1.0
    assert audit["scope"] == "insufficient_history"
