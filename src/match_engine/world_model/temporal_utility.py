"""Couple marginal utility distributions into auditable temporal paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.policy_experiment import (
    policy_horizon_seconds,
)
from src.match_engine.world_model.predictive_distribution import (
    predictive_lattice_is_valid,
)


TEMPORAL_UTILITY_VERSION = 1


def _ordered_horizons(distributions: dict[str, Any]) -> list[str] | None:
    try:
        ordered = sorted(
            distributions,
            key=lambda horizon: (policy_horizon_seconds(horizon), horizon),
        )
    except (TypeError, ValueError):
        return None
    if len(ordered) < 2:
        return None
    seconds = [policy_horizon_seconds(horizon) for horizon in ordered]
    if (
        not all(np.isfinite(value) and value >= 0.0 for value in seconds)
        or any(right <= left for left, right in zip(seconds, seconds[1:]))
    ):
        return None
    return ordered


def couple_temporal_utility_scenarios(
    distributions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Preserve member and residual-rank identities across horizon marginals."""
    horizons = _ordered_horizons(distributions)
    if horizons is None:
        return {
            "available": False,
            "reason": "at_least_two_strictly_ordered_horizons_required",
        }
    scopes = {
        str(distributions[horizon].get("distribution_scope", ""))
        for horizon in horizons
    }
    if len(scopes) != 1:
        return {"available": False, "reason": "temporal_scope_mismatch"}
    scope = next(iter(scopes))
    if scope not in {"epistemic_member_only", "calibrated_predictive"}:
        return {"available": False, "reason": "temporal_scope_invalid"}
    try:
        if not all(
            distribution.get("member_identity_preserved") is True
            for distribution in distributions.values()
        ):
            return {
                "available": False,
                "reason": "temporal_member_identity_not_preserved",
            }
        member_axes = [np.asarray(
            distributions[horizon][
                "epistemic_member_values"
                if scope == "calibrated_predictive" else "member_values"
            ], dtype=np.float64,
        ) for horizon in horizons]
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"available": False, "reason": "temporal_member_axis_invalid"}
    member_count = len(member_axes[0])
    if (
        member_count < 2
        or any(axis.shape != (member_count,) for axis in member_axes)
        or any(not np.all(np.isfinite(axis)) for axis in member_axes)
    ):
        return {"available": False, "reason": "temporal_member_axis_mismatch"}

    residual_count = 0
    marginal_residual_quantiles = 0
    empirical_coupling = None
    residual_rank_coupling_audit = None
    if scope == "calibrated_predictive":
        if not all(predictive_lattice_is_valid(distributions[horizon])
                   for horizon in horizons):
            return {
                "available": False,
                "reason": "temporal_predictive_lattice_invalid",
            }
        try:
            residual_axes = [np.asarray(
                distributions[horizon]["residual_scenario_offsets"],
                dtype=np.float64,
            ) for horizon in horizons]
            level_axes = [np.asarray(
                distributions[horizon]["residual_quantile_levels"],
                dtype=np.float64,
            ) for horizon in horizons]
        except (KeyError, TypeError, ValueError, OverflowError):
            return {
                "available": False,
                "reason": "temporal_residual_axis_invalid",
            }
        residual_count = len(residual_axes[0])
        marginal_residual_quantiles = residual_count
        reference_levels = level_axes[0]
        if (
            residual_count < 3
            or reference_levels.shape != (residual_count,)
            or not np.all(np.diff(reference_levels) > 0.0)
            or any(axis.shape != (residual_count,) for axis in residual_axes)
            or any(not np.all(np.isfinite(axis)) for axis in residual_axes)
            or any(
                levels.shape != reference_levels.shape
                or not np.allclose(
                    levels, reference_levels, atol=1e-12, rtol=0.0,
                )
                for levels in level_axes[1:]
            )
        ):
            return {
                "available": False,
                "reason": "temporal_residual_quantile_axis_mismatch",
            }
        coupling_rows = [
            distributions[horizon].get("temporal_residual_rank_coupling")
            for horizon in horizons
        ]
        if any(row is not None for row in coupling_rows):
            if (
                not all(isinstance(row, dict) for row in coupling_rows)
                or any(row != coupling_rows[0] for row in coupling_rows[1:])
            ):
                return {
                    "available": False,
                    "reason": "temporal_empirical_rank_coupling_mismatch",
                }
            candidate = coupling_rows[0]
            if candidate.get("available"):
                try:
                    template_values = np.asarray(
                        candidate["rank_templates"], dtype=np.float64,
                    )
                    coupling_levels = np.asarray(
                        candidate["rank_levels"], dtype=np.float64,
                    )
                    calibration_samples = int(
                        candidate["calibration_samples"]
                    )
                    rank_correlation = float(
                        candidate["mean_absolute_rank_correlation"]
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    return {
                        "available": False,
                        "reason": "temporal_empirical_rank_contract_invalid",
                    }
                if (
                    int(candidate.get("version", 0)) != 2
                    or list(candidate.get("horizon_keys") or []) != horizons
                    or candidate.get("shared_across_candidate_actions") is not True
                    or candidate.get("temporal_joint_calibrated") is not False
                    or candidate.get("temporal_dependence_empirical") is not True
                    or candidate.get("dependence_source")
                    != "observed_multi_horizon_residual_ranks"
                    or candidate.get("split")
                    != "reference_then_held_out_calibration"
                    or not np.isfinite(rank_correlation)
                    or not 0.0 <= rank_correlation <= 1.0
                    or coupling_levels.shape != reference_levels.shape
                    or not np.allclose(
                        coupling_levels, reference_levels,
                        atol=1e-12, rtol=0.0,
                    )
                    or template_values.ndim != 2
                    or template_values.shape[0] < 4
                    or template_values.shape[1] != len(horizons)
                    or calibration_samples != template_values.shape[0]
                    or not np.all(np.isfinite(template_values))
                    or not np.all(template_values == np.floor(template_values))
                    or np.any(template_values < 0)
                    or np.any(template_values >= residual_count)
                ):
                    return {
                        "available": False,
                        "reason": "temporal_empirical_rank_contract_invalid",
                    }
                templates = template_values.astype(np.int64)
                columns = [np.asarray([
                    member + residual_axes[horizon_index][template[horizon_index]]
                    for member in member_axes[horizon_index]
                    for template in templates
                ], dtype=np.float64) for horizon_index in range(len(horizons))]
                residual_count = templates.shape[0]
                coupling = (
                    "member_identity_x_empirical_residual_rank_templates"
                )
                empirical_coupling = dict(candidate)
            else:
                residual_rank_coupling_audit = {
                    **candidate,
                    "fallback_used": True,
                }
        if empirical_coupling is None:
            columns = [
                (members[:, None] + residuals[None, :]).reshape(-1)
                for members, residuals in zip(member_axes, residual_axes)
            ]
            coupling = "member_identity_x_comonotonic_residual_quantile_rank"
    else:
        columns = member_axes
        coupling = "transition_member_identity"
    paths = np.stack(columns, axis=1)
    if not np.all(np.isfinite(paths)):
        return {"available": False, "reason": "temporal_paths_non_finite"}
    coupling_audit = empirical_coupling or residual_rank_coupling_audit or {
        "available": False,
        "reason": "transparent_comonotonic_rank_fallback",
    }
    dependence_validation = coupling_audit.get("validation") or {
        "empirical_validation_status": (
            "insufficient_evidence"
            if empirical_coupling is not None else "not_applicable"
        ),
    }
    return {
        "available": True,
        "distribution_scope": scope,
        "horizon_keys": horizons,
        "horizon_seconds": [
            float(policy_horizon_seconds(horizon)) for horizon in horizons
        ],
        "scenario_utility_paths": paths.tolist(),
        "scenario_count": len(paths),
        "ensemble_members": member_count,
        "residual_scenarios": residual_count,
        "marginal_residual_quantiles": marginal_residual_quantiles,
        "member_identity_preserved_across_horizons": True,
        "residual_quantile_rank_preserved_across_horizons": bool(
            scope == "calibrated_predictive" and empirical_coupling is None
        ),
        "empirical_residual_rank_template_preserved_across_horizons": bool(
            empirical_coupling is not None
        ),
        "temporal_coupling_source": coupling,
        "temporal_dependence_learned": bool(empirical_coupling is not None),
        "temporal_residual_rank_coupling": coupling_audit,
        "temporal_dependence_validation": dict(dependence_validation),
        "temporal_joint_calibrated": False,
        "marginals_split_calibrated": bool(
            scope == "calibrated_predictive"
        ),
    }


def _path_metrics(paths: np.ndarray) -> dict[str, Any]:
    baseline = np.zeros((len(paths), 1), dtype=np.float64)
    with_baseline = np.concatenate([baseline, paths], axis=1)
    running_max = np.maximum.accumulate(with_baseline, axis=1)
    drawdowns = running_max - with_baseline
    max_drawdown = np.max(drawdowns, axis=1)
    minimum = np.min(paths, axis=1)
    ever_downside = np.any(paths < 0.0, axis=1)
    terminal_upside = paths[:, -1] > 0.0
    recovery = ever_downside & (paths[:, -1] >= 0.0)
    reversal = np.zeros(len(paths), dtype=bool)
    for index in range(1, paths.shape[1]):
        reversal |= (
            np.any(paths[:, :index] > 0.0, axis=1)
            & (paths[:, index] < 0.0)
        )
    tail_count = max(1, int(np.ceil(0.25 * len(paths))))
    return {
        "mean_path_minimum_utility": float(np.mean(minimum)),
        "lower_tail_path_minimum_cvar_25": float(
            np.mean(np.sort(minimum)[:tail_count])
        ),
        "mean_maximum_drawdown": float(np.mean(max_drawdown)),
        "worst_maximum_drawdown": float(np.max(max_drawdown)),
        "ever_downside_scenario_count": int(np.sum(ever_downside)),
        "ever_downside_scenario_rate": float(np.mean(ever_downside)),
        "terminal_upside_scenario_rate": float(np.mean(terminal_upside)),
        "downside_recovery_scenario_rate": float(np.mean(recovery)),
        "positive_to_negative_reversal_scenario_rate": float(
            np.mean(reversal)
        ),
        "path_minimum_values": list(map(float, minimum)),
        "path_maximum_drawdown_values": list(map(float, max_drawdown)),
    }


def build_temporal_utility_paths(
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    distributions = {
        str(horizon): prediction.get("distributional_policy_utility") or {}
        for horizon, prediction in predictions.items()
    }
    coupled = couple_temporal_utility_scenarios(distributions)
    if not coupled.get("available"):
        return {"version": TEMPORAL_UTILITY_VERSION, **coupled}
    paths = np.asarray(coupled["scenario_utility_paths"], dtype=np.float64)
    return {
        "version": TEMPORAL_UTILITY_VERSION,
        **coupled,
        **_path_metrics(paths),
        "scenario_rates_are_temporal_probabilities": False,
        "utility_path_semantics": (
            "same-origin cumulative utility forecasts at ordered horizons; "
            "differences are not independent incremental rewards"
        ),
        "causal_interpretation": False,
    }
