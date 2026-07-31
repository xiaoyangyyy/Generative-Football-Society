"""Aggregate persisted online world-model calibration diagnostics."""

from __future__ import annotations

import math
from typing import Any, Iterable


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def aggregate_online_calibration(
    match_logs: Iterable[dict[str, Any]],
    *,
    min_transitions: int = 50,
) -> dict[str, Any]:
    logs = list(match_logs)
    branch_rows: dict[str, list[dict[str, Any]]] = {
        "general": [], "pass": [], "shot": [],
    }
    target_mismatches = 0
    adoption_registered = adoption_resolved = adoption_count = 0
    interventions_applied = intervention_adopted = 0
    for payload in logs:
        calibration = payload.get("world_model_online_calibration") or {}
        if calibration.get("calibration_target") not in (
            None, "same_tick_next_observation",
        ):
            target_mismatches += 1
        branches = calibration.get("branches") or {}
        for branch in branch_rows:
            row = branches.get(branch)
            if isinstance(row, dict) and int(row.get("samples", 0)) > 0:
                branch_rows[branch].append(row)
        adoption = payload.get("world_model_decision_adoption") or {}
        adoption_registered += int(adoption.get("registered", 0))
        adoption_resolved += int(adoption.get("resolved", 0))
        adoption_count += int(adoption.get("adopted", 0))
        interventions_applied += int(adoption.get("interventions_applied", 0))
        intervention_adopted += int(adoption.get("intervention_adopted", 0))

    summaries = {}
    all_finite = True
    for branch, rows in branch_rows.items():
        samples = sum(int(row.get("samples", 0)) for row in rows)

        def weighted(key: str) -> float:
            if not samples:
                return 0.0
            values = [
                _finite(row.get(key)) * int(row.get("samples", 0))
                for row in rows
            ]
            return sum(values) / samples

        summary = {
            "samples": samples,
            "matches": len(rows),
            "mean_weighted_mse": weighted("mean_weighted_mse"),
            "mean_expected_weighted_mse": weighted("expected_weighted_mse"),
            "mean_skill_vs_persistence": weighted("skill_vs_persistence"),
            "mean_trust_factor": weighted("trust_factor"),
            "mean_uncertainty_error_correlation": weighted(
                "uncertainty_error_correlation"
            ),
        }
        all_finite = all_finite and all(
            math.isfinite(float(value))
            for value in summary.values()
            if isinstance(value, (int, float))
        )
        summaries[branch] = summary

    transitions = summaries["general"]["samples"]
    gates = {
        "minimum_transitions": transitions >= int(min_transitions),
        "consistent_same_target": target_mismatches == 0,
        "finite_metrics": all_finite,
    }
    return {
        "version": 1,
        "evaluation_kind": "online_same_target_world_model_calibration",
        "match_logs": len(logs),
        "branches": summaries,
        "decision_adoption": {
            "registered": adoption_registered,
            "resolved": adoption_resolved,
            "adopted": adoption_count,
            "adoption_rate": adoption_count / max(1, adoption_resolved),
            "interventions_applied": interventions_applied,
            "intervention_adopted": intervention_adopted,
            "intervention_adoption_rate": (
                intervention_adopted / max(1, interventions_applied)
            ),
            "causal_interpretation": False,
        },
        "gates": gates,
        "ready": all(gates.values()),
        "limitations": [
            "Trust factors are valid only for the configured checkpoint and simulator.",
            "Action adoption is temporal association, not causal attribution.",
            "Intervention adoption rate needs a randomized no-bias control for causal attribution.",
        ],
    }
