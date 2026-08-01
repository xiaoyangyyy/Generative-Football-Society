"""Temporal utility paths preserve auditable scenario identities."""

import copy

import numpy as np
import pytest

from src.match_engine.world_model.temporal_utility import (
    build_temporal_utility_paths,
    couple_temporal_utility_scenarios,
)


def _predictive(members, residuals):
    members = np.asarray(members, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    scenarios = (members[:, None] + residuals[None, :]).reshape(-1)
    return {
        "available": True,
        "counts_trusted": True,
        "member_identity_preserved": True,
        "distribution_scope": "calibrated_predictive",
        "predictive_distribution_available": True,
        "member_values": members.tolist(),
        "epistemic_member_values": members.tolist(),
        "residual_scenario_offsets": residuals.tolist(),
        "residual_quantile_levels": [0.1, 0.5, 0.9],
        "decision_scenario_values": scenarios.tolist(),
        "mean_utility": float(scenarios.mean()),
        "residual_calibration_samples": 8,
        "residual_quantiles_split": "held_out_calibration",
        "uncertainty_decomposition": {
            "epistemic_member_variance": float(members.var()),
            "residual_outcome_variance": float(residuals.var()),
            "predictive_lattice_variance": float(scenarios.var()),
            "additive_identity_error": abs(
                float(scenarios.var() - members.var() - residuals.var())
            ),
            "axes_statistically_independent_claimed": False,
        },
    }


def _predictions():
    rows = {
        "180s": _predictive([-0.10, 0.30], [-0.20, 0.00, 0.20]),
        "transition": _predictive([0.05, 0.15], [-0.05, 0.00, 0.05]),
        "60s": _predictive([0.10, 0.20], [-0.10, 0.00, 0.10]),
    }
    return {
        horizon: {"distributional_policy_utility": distribution}
        for horizon, distribution in rows.items()
    }


def test_predictive_paths_preserve_member_and_residual_rank_across_time():
    report = build_temporal_utility_paths(_predictions())

    assert report["available"]
    assert report["horizon_keys"] == ["transition", "60s", "180s"]
    assert report["scenario_count"] == 6
    assert report["ensemble_members"] == 2
    assert report["residual_scenarios"] == 3
    assert report["scenario_utility_paths"][0] == pytest.approx(
        [0.0, 0.0, -0.30]
    )
    assert report["scenario_utility_paths"][5] == pytest.approx(
        [0.20, 0.30, 0.50]
    )
    assert report["member_identity_preserved_across_horizons"]
    assert report["residual_quantile_rank_preserved_across_horizons"]
    assert not report["temporal_dependence_learned"]
    assert not report["temporal_joint_calibrated"]
    assert not report["scenario_rates_are_temporal_probabilities"]
    assert report["positive_to_negative_reversal_scenario_rate"] > 0.0
    assert report["worst_maximum_drawdown"] > 0.0


def test_temporal_coupling_fails_closed_for_scope_and_quantile_mismatch():
    predictions = _predictions()
    distributions = {
        horizon: row["distributional_policy_utility"]
        for horizon, row in predictions.items()
    }
    mismatched = copy.deepcopy(distributions)
    mismatched["60s"]["residual_quantile_levels"] = [0.2, 0.5, 0.9]
    closed = couple_temporal_utility_scenarios(mismatched)
    assert not closed["available"]
    assert closed["reason"] == "temporal_residual_quantile_axis_mismatch"

    mixed_scope = copy.deepcopy(distributions)
    mixed_scope["180s"]["distribution_scope"] = "epistemic_member_only"
    assert couple_temporal_utility_scenarios(mixed_scope)["reason"] == (
        "temporal_scope_mismatch"
    )

    one_horizon = {"60s": distributions["60s"]}
    assert couple_temporal_utility_scenarios(one_horizon)["reason"] == (
        "at_least_two_strictly_ordered_horizons_required"
    )

    nonfinite_horizon = {
        "transition": distributions["transition"],
        "nans": distributions["60s"],
    }
    assert couple_temporal_utility_scenarios(nonfinite_horizon)["reason"] == (
        "at_least_two_strictly_ordered_horizons_required"
    )


def test_epistemic_paths_use_only_transition_member_identity():
    distributions = {
        "transition": {
            "distribution_scope": "epistemic_member_only",
            "member_identity_preserved": True,
            "member_values": [-0.1, 0.2],
        },
        "60s": {
            "distribution_scope": "epistemic_member_only",
            "member_identity_preserved": True,
            "member_values": [0.0, 0.3],
        },
    }
    coupled = couple_temporal_utility_scenarios(distributions)
    assert coupled["available"]
    assert coupled["scenario_utility_paths"] == [[-0.1, 0.0], [0.2, 0.3]]
    assert coupled["temporal_coupling_source"] == "transition_member_identity"
    assert not coupled["residual_quantile_rank_preserved_across_horizons"]
