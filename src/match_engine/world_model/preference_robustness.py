"""Model-owned stress tests against post-hoc LLM preference rationalization."""

from __future__ import annotations

from typing import Any

import numpy as np


PREFERENCE_ROBUSTNESS_VERSION = 1
LOSS_AVERSION_RADIUS = 0.50
SENSITIVITY_RADIUS = 0.10
_ACTIONS = ("hold", "pass", "cross", "shot")


def _parameter_grid(preference: dict[str, Any]) -> list[tuple[float, float]]:
    loss = float(preference["loss_aversion"])
    sensitivity = float(preference["diminishing_sensitivity"])
    losses = sorted({float(np.clip(
        loss + delta, 1.0, 4.0,
    )) for delta in (-LOSS_AVERSION_RADIUS, 0.0, LOSS_AVERSION_RADIUS)})
    sensitivities = sorted({float(np.clip(
        sensitivity + delta, 0.50, 1.0,
    )) for delta in (-SENSITIVITY_RADIUS, 0.0, SENSITIVITY_RADIUS)})
    return [(loss_value, sensitivity_value)
            for loss_value in losses
            for sensitivity_value in sensitivities]


def _stress_grid(
    scope: str,
    evidence: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    items = [item for actions in evidence.values() for item in actions.values()]
    try:
        if scope == "calibrated_predictive":
            member_count = min(len(item["epistemic_member_values"])
                               for item in items)
            residual_count = min(len(item["residual_scenario_offsets"])
                                 for item in items)
            levels = [np.asarray(
                item["residual_quantile_levels"], dtype=np.float64,
            ) for item in items]
            reference_levels = levels[0]
            if (
                len(reference_levels) != residual_count
                or not np.all(np.diff(reference_levels) > 0.0)
                or any(
                    level.shape != reference_levels.shape
                    or not np.allclose(
                        level, reference_levels, atol=1e-12, rtol=0.0,
                    )
                    for level in levels[1:]
                )
            ):
                return None
        else:
            member_count = min(len(item["member_values"]) for item in items)
            residual_count = 0
    except (KeyError, TypeError, ValueError):
        return None
    if member_count < 2 or (
        scope == "calibrated_predictive" and residual_count < 3
    ):
        return None
    cases = [{"kind": "nominal", "index": None}]
    cases.extend(
        {"kind": "drop_epistemic_member", "index": index}
        for index in range(member_count)
    )
    cases.extend(
        {"kind": "drop_residual_scenario", "index": index}
        for index in range(residual_count)
    )
    return cases


def _stressed_values(
    item: dict[str, Any],
    scope: str,
    stress: dict[str, Any],
) -> np.ndarray | None:
    try:
        if scope == "calibrated_predictive":
            members = np.asarray(
                item["epistemic_member_values"], dtype=np.float64,
            )
            residuals = np.asarray(
                item["residual_scenario_offsets"], dtype=np.float64,
            )
            if stress["kind"] == "drop_epistemic_member":
                members = np.delete(members, int(stress["index"]))
            elif stress["kind"] == "drop_residual_scenario":
                residuals = np.delete(residuals, int(stress["index"]))
            values = (members[:, None] + residuals[None, :]).reshape(-1)
        else:
            values = np.asarray(item["member_values"], dtype=np.float64)
            if stress["kind"] == "drop_epistemic_member":
                values = np.delete(values, int(stress["index"]))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        return None
    return values


def _prospect_mean(
    values: np.ndarray,
    *,
    loss_aversion: float,
    sensitivity: float,
) -> float:
    gains = np.power(np.maximum(values, 0.0), sensitivity)
    losses = -loss_aversion * np.power(
        np.maximum(-values, 0.0), sensitivity,
    )
    return float(np.mean(np.where(values >= 0.0, gains, losses)))


def _case_report(
    preference: dict[str, Any],
    evidence: dict[str, dict[str, dict[str, Any]]],
    *,
    loss_aversion: float,
    sensitivity: float,
    stress: dict[str, Any],
) -> dict[str, Any] | None:
    selected = preference["selected_action"]
    aggregate = {action: 0.0 for action in _ACTIONS}
    horizon_regrets: dict[str, float] = {}
    for horizon, weight in preference["horizon_weights"].items():
        action_values: dict[str, float] = {}
        for action in _ACTIONS:
            item = evidence[horizon][action]
            values = _stressed_values(
                item, preference["distribution_scope"], stress,
            )
            if values is None:
                return None
            expected = _prospect_mean(
                values, loss_aversion=loss_aversion,
                sensitivity=sensitivity,
            )
            if not np.isfinite(expected):
                return None
            action_values[action] = expected
            aggregate[action] += float(weight) * expected
        horizon_regrets[horizon] = max(
            0.0, max(action_values.values()) - action_values[selected],
        )
    best = max(aggregate.values())
    aggregate_regret = max(0.0, best - aggregate[selected])
    worst_horizon_regret = max(horizon_regrets.values(), default=0.0)
    threshold = float(preference["max_acceptable_regret"])
    return {
        "loss_aversion": loss_aversion,
        "diminishing_sensitivity": sensitivity,
        "stress_kind": stress["kind"],
        "stress_index": stress["index"],
        "aggregate_preferred_actions": sorted(
            action for action, value in aggregate.items()
            if abs(value - best) <= 1e-12
        ),
        "selected_aggregate_preference_regret": aggregate_regret,
        "selected_worst_horizon_preference_regret": worst_horizon_regret,
        "within_declared_regret": bool(
            aggregate_regret <= threshold + 1e-12
            and worst_horizon_regret <= threshold + 1e-12
        ),
    }


def build_preference_robustness_report(
    preference: dict[str, Any],
    evidence: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    """Cross fixed preference perturbations with leave-one-axis-out evidence."""
    try:
        scope = str(preference["distribution_scope"])
        parameter_grid = _parameter_grid(preference)
        stress_grid = _stress_grid(scope, evidence)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if stress_grid is None:
        return None
    cases = []
    for loss_aversion, sensitivity in parameter_grid:
        for stress in stress_grid:
            report = _case_report(
                preference, evidence,
                loss_aversion=loss_aversion,
                sensitivity=sensitivity,
                stress=stress,
            )
            if report is None:
                return None
            cases.append(report)
    if not cases:
        return None
    within = sum(case["within_declared_regret"] for case in cases)
    worst_aggregate = max(
        case["selected_aggregate_preference_regret"] for case in cases
    )
    worst_horizon = max(
        case["selected_worst_horizon_preference_regret"] for case in cases
    )
    worst_case = max(
        cases,
        key=lambda case: (
            case["selected_aggregate_preference_regret"],
            case["selected_worst_horizon_preference_regret"],
        ),
    )
    return {
        "version": PREFERENCE_ROBUSTNESS_VERSION,
        "available": True,
        "parameter_grid_owned_by": "world_model",
        "loss_aversion_radius": LOSS_AVERSION_RADIUS,
        "diminishing_sensitivity_radius": SENSITIVITY_RADIUS,
        "parameter_cases": len(parameter_grid),
        "evidence_stress_cases": len(stress_grid),
        "joint_stress_cases": len(cases),
        "cases_within_declared_regret": within,
        "robustness_rate": float(within / len(cases)),
        "rationalization_fragility": float(1.0 - within / len(cases)),
        "worst_case_aggregate_preference_regret": worst_aggregate,
        "worst_case_horizon_preference_regret": worst_horizon,
        "preference_robust": within == len(cases),
        "member_axis_jointly_aligned": True,
        "residual_quantile_axis_jointly_aligned": (
            scope == "calibrated_predictive"
        ),
        "worst_case": worst_case,
        "failed_cases": [
            case for case in cases if not case["within_declared_regret"]
        ],
        "llm_selected_stress_range": False,
        "causal_interpretation": False,
    }
