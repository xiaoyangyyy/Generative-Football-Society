"""Pathwise multi-action regret under an LLM's frozen risk preference."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.temporal_utility import (
    couple_temporal_utility_scenarios,
)


TEMPORAL_PREFERENCE_VERSION = 1
MIN_PATHWISE_ROBUSTNESS_RATE = 0.75
_ACTIONS = ("hold", "pass", "cross", "shot")


def _prospect_values(
    values: np.ndarray,
    preference: dict[str, Any],
) -> np.ndarray:
    sensitivity = float(preference["diminishing_sensitivity"])
    loss_aversion = float(preference["loss_aversion"])
    gains = np.power(np.maximum(values, 0.0), sensitivity)
    losses = -loss_aversion * np.power(
        np.maximum(-values, 0.0), sensitivity,
    )
    return np.where(values >= 0.0, gains, losses)


def build_temporal_preference_report(
    preference: dict[str, Any],
    evidence: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    """Compare actions inside each coupled path before aggregating scenarios."""
    try:
        horizons = list(preference["horizon_weights"])
        if set(evidence) != set(horizons):
            return None
        action_paths = {}
        coupling_reports = {}
        for action in _ACTIONS:
            report = couple_temporal_utility_scenarios({
                horizon: evidence[horizon][action] for horizon in horizons
            })
            if not report.get("available"):
                return None
            coupling_reports[action] = report
            action_paths[action] = np.asarray(
                report["scenario_utility_paths"], dtype=np.float64,
            )
        if preference["distribution_scope"] == "calibrated_predictive":
            levels = [np.asarray(
                evidence[horizon][action]["residual_quantile_levels"],
                dtype=np.float64,
            ) for horizon in horizons for action in _ACTIONS]
            reference_levels = levels[0]
            if any(
                level.shape != reference_levels.shape
                or not np.allclose(
                    level, reference_levels, atol=1e-12, rtol=0.0,
                )
                for level in levels[1:]
            ):
                return None
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    reference = coupling_reports[_ACTIONS[0]]
    path_shape = action_paths[_ACTIONS[0]].shape
    if (
        path_shape[1] != len(horizons)
        or reference["distribution_scope"] != preference["distribution_scope"]
        or any(paths.shape != path_shape for paths in action_paths.values())
        or any(
            report["horizon_keys"] != reference["horizon_keys"]
            or report["distribution_scope"] != reference["distribution_scope"]
            or report["temporal_coupling_source"]
            != reference["temporal_coupling_source"]
            or report["temporal_residual_rank_coupling"]
            != reference["temporal_residual_rank_coupling"]
            for report in coupling_reports.values()
        )
    ):
        return None
    # Preference weights are keyed independently of lexical input order; align
    # them to the temporal coupling's strictly increasing horizon order.
    ordered_horizons = reference["horizon_keys"]
    weights = np.asarray([
        preference["horizon_weights"][horizon]
        for horizon in ordered_horizons
    ], dtype=np.float64)
    transformed = np.stack([
        _prospect_values(action_paths[action], preference)
        for action in _ACTIONS
    ], axis=1)  # [scenario, action, horizon]
    path_scores = np.sum(transformed * weights[None, None, :], axis=2)
    selected_index = _ACTIONS.index(preference["selected_action"])
    best_path_scores = np.max(path_scores, axis=1)
    selected_path_scores = path_scores[:, selected_index]
    aggregate_regrets = np.maximum(
        0.0, best_path_scores - selected_path_scores,
    )
    best_horizon_scores = np.max(transformed, axis=1)
    selected_horizon_scores = transformed[:, selected_index, :]
    horizon_regrets = np.maximum(
        0.0, best_horizon_scores - selected_horizon_scores,
    )
    worst_horizon_regrets = np.max(horizon_regrets, axis=1)
    threshold = float(preference["max_acceptable_regret"])
    within = (
        (aggregate_regrets <= threshold + 1e-12)
        & (worst_horizon_regrets <= threshold + 1e-12)
    )
    horizon_best = np.max(transformed, axis=1, keepdims=True)
    horizon_leaders = transformed >= horizon_best - 1e-12
    action_switch = np.any(
        horizon_leaders != horizon_leaders[:, :, :1], axis=(1, 2),
    )
    selected_pathwise_preferred = (
        selected_path_scores >= best_path_scores - 1e-12
    )
    rate = float(np.mean(within))
    worst_index = int(np.lexsort((
        worst_horizon_regrets, aggregate_regrets,
    ))[-1])

    def scenario_row(index: int) -> dict[str, Any]:
        return {
            "scenario_index": index,
            "aggregate_preferred_actions": [
                action for action, value in zip(_ACTIONS, path_scores[index])
                if value >= best_path_scores[index] - 1e-12
            ],
            "selected_aggregate_preference_regret": float(
                aggregate_regrets[index]
            ),
            "selected_worst_horizon_preference_regret": float(
                worst_horizon_regrets[index]
            ),
            "within_declared_regret": bool(within[index]),
        }

    return {
        "version": TEMPORAL_PREFERENCE_VERSION,
        "available": True,
        "horizon_keys": ordered_horizons,
        "scenario_count": path_shape[0],
        "distribution_scope": reference["distribution_scope"],
        "temporal_coupling_source": reference["temporal_coupling_source"],
        "temporal_dependence_learned": bool(
            reference["temporal_dependence_learned"]
        ),
        "temporal_joint_calibrated": False,
        "member_axis_jointly_aligned_across_actions": True,
        "residual_quantile_axis_jointly_aligned_across_actions": bool(
            preference["distribution_scope"] == "calibrated_predictive"
            and not reference["temporal_dependence_learned"]
        ),
        "empirical_residual_rank_templates_jointly_aligned_across_actions": bool(
            preference["distribution_scope"] == "calibrated_predictive"
            and reference["temporal_dependence_learned"]
        ),
        "pathwise_comparison_order": (
            "compare_actions_within_scenario_then_aggregate_scenarios"
        ),
        "pathwise_regret_within_declared_rate": rate,
        "minimum_pathwise_robustness_rate": MIN_PATHWISE_ROBUSTNESS_RATE,
        "mean_selected_pathwise_preference_regret": float(
            np.mean(aggregate_regrets)
        ),
        "q90_selected_pathwise_preference_regret": float(np.quantile(
            aggregate_regrets, 0.90, method="linear",
        )),
        "worst_selected_pathwise_preference_regret": float(
            np.max(aggregate_regrets)
        ),
        "worst_selected_horizon_preference_regret": float(
            np.max(worst_horizon_regrets)
        ),
        "selected_pathwise_preferred_scenario_rate": float(np.mean(
            selected_pathwise_preferred
        )),
        "best_action_switches_across_horizons_scenario_rate": float(
            np.mean(action_switch)
        ),
        "preference_temporally_robust": bool(
            rate + 1e-12 >= MIN_PATHWISE_ROBUSTNESS_RATE
        ),
        "worst_scenario": scenario_row(worst_index),
        "failed_scenarios": [
            scenario_row(index) for index in range(path_shape[0])
            if not within[index]
        ],
        "scenario_rates_are_temporal_probabilities": False,
        "causal_interpretation": False,
    }
