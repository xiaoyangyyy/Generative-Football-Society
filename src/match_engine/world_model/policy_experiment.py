"""Randomized policy-bridge assignment, causal estimates, and trust feedback."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np


def randomized_experiment_arm(
    decision_id: str,
    *,
    seed: int,
    control_rate: float,
) -> str:
    """Assign an opportunity without consuming the match action RNG."""
    rate = min(0.5, max(0.0, float(control_rate)))
    if rate <= 0.0:
        return "treatment"
    digest = hashlib.sha256(
        f"policy-bridge-v1:{int(seed)}:{decision_id}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    return "control" if draw < rate else "treatment"


def randomized_adoption_effect_from_counts(
    *,
    treatment: int,
    treatment_adopted: int,
    control: int,
    control_adopted: int,
    min_per_arm: int = 2,
) -> dict[str, Any]:
    """Build an add-two-corrected difference-in-proportions report."""
    treatment = max(0, int(treatment))
    control = max(0, int(control))
    treatment_adopted = min(treatment, max(0, int(treatment_adopted)))
    control_adopted = min(control, max(0, int(control_adopted)))
    treatment_rate = treatment_adopted / max(1, treatment)
    control_rate = control_adopted / max(1, control)
    effect = treatment_rate - control_rate
    treatment_adjusted = (treatment_adopted + 1.0) / (treatment + 2.0)
    control_adjusted = (control_adopted + 1.0) / (control + 2.0)
    standard_error = math.sqrt(
        treatment_adjusted * (1.0 - treatment_adjusted) / (treatment + 2.0)
        + control_adjusted * (1.0 - control_adjusted) / (control + 2.0)
    )
    minimum = max(2, int(min_per_arm))
    ready = treatment >= minimum and control >= minimum
    return {
        "evaluation_kind": "randomized_zero_bias_policy_bridge_adoption",
        "opportunities": treatment + control,
        "treatment": treatment,
        "treatment_adopted": treatment_adopted,
        "treatment_adoption_rate": treatment_rate,
        "control": control,
        "control_adopted": control_adopted,
        "control_adoption_rate": control_rate,
        "average_treatment_effect": effect,
        "standard_error": standard_error,
        "confidence_interval": [
            max(-1.0, effect - 1.96 * standard_error),
            min(1.0, effect + 1.96 * standard_error),
        ],
        "minimum_per_arm": minimum,
        "ready": ready,
        "causal_interpretation": (
            "within_simulator_action_selection"
            if ready else "insufficient_randomized_samples"
        ),
    }


def randomized_adoption_effect(
    records: Iterable[dict[str, Any]],
    *,
    min_per_arm: int = 2,
) -> dict[str, Any]:
    opportunities = [
        record for record in records
        if record.get("policy_opportunity_observed")
        and record.get("experiment_arm") in {"treatment", "control"}
        and record.get("adopted") is not None
    ]
    treatment = [r for r in opportunities if r["experiment_arm"] == "treatment"]
    control = [r for r in opportunities if r["experiment_arm"] == "control"]
    return randomized_adoption_effect_from_counts(
        treatment=len(treatment),
        treatment_adopted=sum(bool(record["adopted"]) for record in treatment),
        control=len(control),
        control_adopted=sum(bool(record["adopted"]) for record in control),
        min_per_arm=min_per_arm,
    )


def _outcome_rows(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> list[tuple[int, dict[str, Any], float]]:
    rows = []
    for cluster_index, records in enumerate(record_clusters):
        for record in records:
            outcome = record.get("short_horizon_outcome") or {}
            arm = record.get("experiment_arm")
            value = outcome.get("policy_utility")
            if arm not in {"treatment", "control"} or value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                rows.append((cluster_index, record, value))
    return rows


def randomized_outcome_effect(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
    *,
    min_per_arm: int = 2,
) -> dict[str, Any]:
    """Estimate short-horizon utility with stabilized IPW and cluster SEs."""
    clusters = list(record_clusters)
    rows = _outcome_rows(clusters)
    treatment = [row for row in rows if row[1]["experiment_arm"] == "treatment"]
    control = [row for row in rows if row[1]["experiment_arm"] == "control"]
    minimum = max(2, int(min_per_arm))
    ready = len(treatment) >= minimum and len(control) >= minimum

    def propensity(record: dict[str, Any]) -> float:
        value = float(record.get("experiment_treatment_propensity", 0.8))
        return float(np.clip(value, 0.01, 0.99))

    treated_weights = np.asarray(
        [1.0 / propensity(record) for _, record, _ in treatment], dtype=float,
    )
    control_weights = np.asarray(
        [1.0 / (1.0 - propensity(record)) for _, record, _ in control],
        dtype=float,
    )
    treated_values = np.asarray([value for _, _, value in treatment], dtype=float)
    control_values = np.asarray([value for _, _, value in control], dtype=float)
    treated_total = float(treated_weights.sum())
    control_total = float(control_weights.sum())
    treated_mean = float(
        np.dot(treated_weights, treated_values) / treated_total
    ) if treated_total else 0.0
    control_mean = float(
        np.dot(control_weights, control_values) / control_total
    ) if control_total else 0.0
    effect = treated_mean - control_mean

    contributions: dict[int, float] = {
        cluster_index: 0.0 for cluster_index in range(len(clusters))
    }
    individual = []
    for cluster_index, record, value in rows:
        p = propensity(record)
        if record["experiment_arm"] == "treatment" and treated_total:
            contribution = (value - treated_mean) / p / treated_total
        elif control_total:
            contribution = -(value - control_mean) / (1.0 - p) / control_total
        else:
            contribution = 0.0
        contributions[cluster_index] += contribution
        individual.append(contribution)
    populated_clusters = len({cluster for cluster, _, _ in rows})
    treatment_clusters = len({cluster for cluster, _, _ in treatment})
    control_clusters = len({cluster for cluster, _, _ in control})
    if (
        populated_clusters >= 2
        and treatment_clusters >= 2
        and control_clusters >= 2
    ):
        correction = populated_clusters / (populated_clusters - 1.0)
        standard_error = math.sqrt(
            correction * sum(value * value for value in contributions.values())
        )
        cluster_robust = True
    else:
        standard_error = math.sqrt(sum(value * value for value in individual))
        cluster_robust = False
    return {
        "evaluation_kind": "randomized_policy_bridge_short_horizon_outcome",
        "outcome": "goal_diff + 0.35*xg_net + 0.15*progress + 0.05*retention_edge",
        "estimator": "stabilized_inverse_propensity_weighted",
        "opportunities": len(rows),
        "treatment": len(treatment),
        "control": len(control),
        "treatment_mean_utility": treated_mean,
        "control_mean_utility": control_mean,
        "average_treatment_effect": effect,
        "standard_error": standard_error,
        "confidence_interval": [
            effect - 1.96 * standard_error,
            effect + 1.96 * standard_error,
        ],
        "clusters": populated_clusters,
        "treatment_clusters": treatment_clusters,
        "control_clusters": control_clusters,
        "cluster_robust": cluster_robust,
        "minimum_per_arm": minimum,
        "ready": ready,
        "causal_interpretation": (
            "within_simulator_short_horizon_outcome"
            if ready and cluster_robust
            else "within_simulator_single_match_exploratory"
            if ready else "insufficient_randomized_samples"
        ),
    }


def policy_bridge_reliability_factor(
    state,
    *,
    min_per_arm: int = 2,
) -> float:
    """Use outcome evidence first; never raise the confidence-derived bound."""
    records = list(getattr(state, "_wm_coach_decision_adoption", None) or [])
    outcome = randomized_outcome_effect([records], min_per_arm=min_per_arm)
    effect = outcome if outcome["ready"] else randomized_adoption_effect(
        records, min_per_arm=min_per_arm,
    )
    if not effect["ready"]:
        return 1.0
    low, high = effect["confidence_interval"]
    if high <= 0.0:
        return 0.25
    if effect["average_treatment_effect"] <= 0.0:
        return 0.60
    if low > 0.0:
        return 1.0
    return 0.85
