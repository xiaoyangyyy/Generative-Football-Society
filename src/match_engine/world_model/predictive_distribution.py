"""Compose epistemic members with held-out residual outcome scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np


PREDICTIVE_DISTRIBUTION_VERSION = 1
MIN_RESIDUAL_SCENARIO_SAMPLES = 4
_QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
_QUANTILE_KEYS = ("q10", "q25", "q50", "q75", "q90")


def predictive_lattice_is_valid(distribution: dict[str, Any]) -> bool:
    """Rebuild and verify a persisted calibrated-predictive lattice."""
    if (
        str(distribution.get("distribution_scope", ""))
        != "calibrated_predictive"
    ):
        return False
    try:
        members = np.asarray(
            distribution["epistemic_member_values"], dtype=np.float64,
        )
        residuals = np.asarray(
            distribution["residual_scenario_offsets"], dtype=np.float64,
        )
        scenarios = np.asarray(
            distribution["decision_scenario_values"], dtype=np.float64,
        )
        samples = int(distribution["residual_calibration_samples"])
        decomposition = distribution["uncertainty_decomposition"]
        declared = tuple(float(decomposition[key]) for key in (
            "epistemic_member_variance",
            "residual_outcome_variance",
            "predictive_lattice_variance",
            "additive_identity_error",
        ))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not (
        distribution.get("predictive_distribution_available") is True
        and distribution.get("residual_quantiles_split")
        == "held_out_calibration"
        and samples >= MIN_RESIDUAL_SCENARIO_SAMPLES
        and members.ndim == residuals.ndim == scenarios.ndim == 1
        and len(members) >= 2 and len(residuals) >= 3
        and len(scenarios) == len(members) * len(residuals)
        and np.all(np.isfinite(members))
        and np.all(np.isfinite(residuals))
        and np.all(np.isfinite(scenarios))
        and all(np.isfinite(value) for value in declared)
        and decomposition.get("axes_statistically_independent_claimed") is False
    ):
        return False
    rebuilt = (members[:, None] + residuals[None, :]).reshape(-1)
    actual = (
        float(np.var(members)), float(np.var(residuals)),
        float(np.var(rebuilt)),
    )
    identity = abs(actual[2] - actual[0] - actual[1])
    return bool(
        np.allclose(scenarios, rebuilt, atol=1e-9, rtol=0.0)
        and all(abs(left - right) <= 1e-9 for left, right in zip(
            declared[:3], actual,
        ))
        and abs(declared[3] - identity) <= 1e-9
    )


def _summarize(values: np.ndarray) -> dict[str, Any]:
    ordered = np.sort(values)
    count = len(values)
    tail_count = max(1, int(np.ceil(0.25 * count)))
    quantiles = np.quantile(values, _QUANTILE_LEVELS, method="linear")
    downside = int(np.sum(values < 0.0))
    upside = int(np.sum(values > 0.0))
    return {
        "mean_utility": float(np.mean(values)),
        "utility_std": float(np.std(values)),
        "minimum_utility": float(ordered[0]),
        "maximum_utility": float(ordered[-1]),
        "lower_tail_cvar_25": float(np.mean(ordered[:tail_count])),
        "upper_tail_mean_25": float(np.mean(ordered[-tail_count:])),
        "quantiles": dict(zip(_QUANTILE_KEYS, map(float, quantiles))),
        "downside_probability": float((downside + 0.5) / (count + 1.0)),
        "upside_probability": float((upside + 0.5) / (count + 1.0)),
        "downside_scenario_count": downside,
        "upside_scenario_count": upside,
        "tail_scenario_count": tail_count,
    }


def build_predictive_scenario_lattice(
    distribution: dict[str, Any],
    residual_memory: dict[str, Any],
) -> dict[str, Any]:
    """Cross centered members with split-calibration residual quantiles.

    This is an auditable predictive construction, not a claim that the two axes
    are causally independent. Equal cross-product weights make the reported
    variance decomposition an exact identity for this construction.
    """
    output = dict(distribution)
    output.update({
        "predictive_distribution_version": PREDICTIVE_DISTRIBUTION_VERSION,
        "predictive_distribution_available": False,
        "distribution_scope": "epistemic_member_only",
        "decision_scenario_values": list(output.get("member_values") or []),
        "residual_calibration_samples": int(
            residual_memory.get("calibration_samples", 0) or 0
        ),
    })
    if not (
        output.get("available")
        and output.get("counts_trusted")
        and output.get("member_identity_preserved")
    ):
        output["predictive_distribution_reason"] = (
            "trusted_member_distribution_required"
        )
        return output
    try:
        members = np.asarray(output["member_values"], dtype=np.float64)
        residuals = np.asarray(
            residual_memory["residual_quantiles"], dtype=np.float64,
        )
        levels = np.asarray(
            residual_memory["residual_quantile_levels"], dtype=np.float64,
        )
        calibration_samples = int(residual_memory["calibration_samples"])
    except (KeyError, TypeError, ValueError, OverflowError):
        output["predictive_distribution_reason"] = (
            "held_out_residual_quantiles_unavailable"
        )
        return output
    valid = bool(
        members.ndim == residuals.ndim == levels.ndim == 1
        and len(members) >= 2
        and len(residuals) == len(levels) >= 3
        and calibration_samples >= MIN_RESIDUAL_SCENARIO_SAMPLES
        and np.all(np.isfinite(members))
        and np.all(np.isfinite(residuals))
        and np.all(np.isfinite(levels))
        and np.all(np.diff(levels) > 0.0)
        and np.all(np.diff(residuals) >= -1e-12)
        and residual_memory.get("residual_quantiles_split")
        == "held_out_calibration"
        and residual_memory.get("residual_quantiles_observed") is True
        and (residual_memory.get("drift") or {}).get("status")
        in {"stable", "watch", "insufficient_history"}
    )
    if not valid:
        output["predictive_distribution_reason"] = (
            "residual_scenario_calibration_gate_closed"
        )
        return output

    member_mean = float(np.mean(members))
    epistemic_deviations = members - member_mean
    epistemic_summary = {
        key: output[key] for key in (
            "mean_utility", "utility_std", "minimum_utility",
            "maximum_utility", "lower_tail_cvar_25",
            "upper_tail_mean_25", "quantiles", "downside_probability",
            "upside_probability", "downside_member_count",
            "upside_member_count", "tail_member_count",
        ) if key in output
    }
    lattice = (
        member_mean + epistemic_deviations[:, None] + residuals[None, :]
    ).reshape(-1)
    epistemic_variance = float(np.var(epistemic_deviations))
    residual_variance = float(np.var(residuals))
    total_variance = float(np.var(lattice))
    identity_error = abs(
        total_variance - epistemic_variance - residual_variance
    )
    output.update(_summarize(lattice))
    output.update({
        "predictive_distribution_available": True,
        "predictive_distribution_reason": (
            "member_x_split_calibration_residual_quantile_lattice"
        ),
        "distribution_scope": "calibrated_predictive",
        "epistemic_member_values": list(map(float, members)),
        "epistemic_member_mean": member_mean,
        "epistemic_distribution_summary": epistemic_summary,
        "residual_scenario_offsets": list(map(float, residuals)),
        "residual_quantile_levels": list(map(float, levels)),
        "predictive_scenario_values": list(map(float, lattice)),
        "decision_scenario_values": list(map(float, lattice)),
        "predictive_scenario_count": len(lattice),
        "residual_calibration_samples": calibration_samples,
        "uncertainty_decomposition": {
            "epistemic_member_variance": epistemic_variance,
            "residual_outcome_variance": residual_variance,
            "predictive_lattice_variance": total_variance,
            "additive_identity_error": identity_error,
            "axes_statistically_independent_claimed": False,
        },
        "probability_source": (
            "equal_weight_member_x_calibration_quantile_scenarios"
        ),
        "residual_quantiles_split": "held_out_calibration",
        "observed": False,
        "causal_interpretation": False,
    })
    return output
