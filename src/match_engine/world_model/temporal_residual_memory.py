"""Contextual empirical rank templates for multi-horizon residual dependence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.policy_experiment import (
    policy_horizon_seconds,
)


TEMPORAL_RESIDUAL_MEMORY_VERSION = 1
RANK_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class TemporalRankCorrection:
    scope: str
    horizons: tuple[str, ...]
    samples: int
    reference_samples: int
    calibration_samples: int
    rank_templates: tuple[tuple[int, ...], ...]
    mean_absolute_rank_correlation: float


@dataclass
class TemporalResidualRankMemory:
    checkpoint_signature: str
    environment_signature: str
    groups: dict[tuple[str, ...], TemporalRankCorrection]
    path_rows: int
    min_samples: int
    drift: dict[str, Any]

    def _keys(
        self,
        context: dict[str, Any],
        horizons: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        team = str(context.get("team_id", "unknown"))
        opponent = str(context.get("opponent_team_id", "unknown"))
        zone = str(context.get("zone", "unknown"))
        score = str(context.get("score_state", "unknown"))
        phase = str(context.get("match_phase", "unknown"))
        suffix = ("horizons", *horizons)
        return (
            ("team_opponent_context", team, opponent, zone, score, phase,
             *suffix),
            ("team_context", team, zone, score, phase, *suffix),
            ("context", zone, score, phase, *suffix),
            ("team", team, *suffix),
            ("global", *suffix),
        )

    def lookup(
        self,
        *,
        context: dict[str, Any],
        horizon_keys: Iterable[str],
    ) -> dict[str, Any]:
        try:
            horizons = tuple(sorted(
                {str(horizon) for horizon in horizon_keys},
                key=lambda horizon: (
                    policy_horizon_seconds(horizon), horizon,
                ),
            ))
        except (TypeError, ValueError):
            horizons = ()
        base = {
            "version": TEMPORAL_RESIDUAL_MEMORY_VERSION,
            "available": False,
            "horizon_keys": list(horizons),
            "rank_levels": list(RANK_LEVELS),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "shared_across_candidate_actions": True,
            "temporal_joint_calibrated": False,
            "causal_interpretation": False,
        }
        if len(horizons) < 2:
            return {**base, "reason": "at_least_two_horizons_required"}
        if self.drift.get("status") in {"watch", "quarantined"}:
            return {
                **base,
                "reason": "temporal_rank_memory_drift_gate_closed",
                "drift": self.drift,
            }
        for key in self._keys(context, horizons):
            correction = self.groups.get(key)
            if correction is None:
                continue
            return {
                **base,
                "available": True,
                "reason": "held_out_empirical_residual_rank_templates",
                "scope": correction.scope,
                "samples": correction.samples,
                "reference_samples": correction.reference_samples,
                "calibration_samples": correction.calibration_samples,
                "rank_templates": [
                    list(template) for template in correction.rank_templates
                ],
                "mean_absolute_rank_correlation": (
                    correction.mean_absolute_rank_correlation
                ),
                "split": "reference_then_held_out_calibration",
                "dependence_source": "observed_multi_horizon_residual_ranks",
                "temporal_dependence_empirical": True,
                "drift": self.drift,
            }
        return {
            **base,
            "reason": "insufficient_contextual_temporal_residual_history",
            "samples": self.path_rows,
            "drift": self.drift,
        }

    def summary(self) -> dict[str, Any]:
        scopes: dict[str, int] = {}
        for correction in self.groups.values():
            scopes[correction.scope] = scopes.get(correction.scope, 0) + 1
        return {
            "version": TEMPORAL_RESIDUAL_MEMORY_VERSION,
            "path_rows": self.path_rows,
            "active_groups": len(self.groups),
            "groups_by_scope": scopes,
            "minimum_samples": self.min_samples,
            "rank_levels": list(RANK_LEVELS),
            "shared_across_candidate_actions": True,
            "drift": self.drift,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "groups": [
                {"key": list(key), **asdict(correction)}
                for key, correction in sorted(self.groups.items())
            ],
        }


def _ordered_residuals(record: dict[str, Any]) -> dict[str, float]:
    outcomes = (
        record.get("multi_horizon_regime_outcomes")
        or record.get("multi_horizon_outcomes")
        or {}
    )
    residuals = {}
    for horizon, outcome in outcomes.items():
        prediction = (outcome or {}).get("world_model_prediction") or {}
        try:
            predicted = float(prediction.get(
                "raw_policy_utility", prediction.get("policy_utility"),
            ))
            actual = float(outcome["policy_utility"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(predicted) and math.isfinite(actual):
            residuals[str(horizon)] = actual - predicted
    try:
        return dict(sorted(
            residuals.items(),
            key=lambda item: (policy_horizon_seconds(item[0]), item[0]),
        ))
    except ValueError:
        return {}


def _path_rows(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> list[dict[str, Any]]:
    rows = []
    for payload in logs:
        adoption = payload.get("world_model_decision_adoption") or {}
        for record in adoption.get("records") or []:
            if (
                str(record.get("checkpoint_signature"))
                != checkpoint_signature
                or int(record.get("policy_utility_version", 0)) != 1
                or str(record.get(
                    "environment_signature", "environment_unspecified",
                )) != environment_signature
            ):
                continue
            residuals = _ordered_residuals(record)
            if len(residuals) < 2:
                continue
            baseline = record.get("outcome_baseline") or {}
            team = str(record.get("team_id", "unknown"))
            opponent = baseline.get("opponent_team_id")
            if not opponent:
                opponent = (
                    payload.get("away")
                    if team == payload.get("home") else payload.get("home")
                )
            rows.append({
                "context": {
                    "team_id": team,
                    "opponent_team_id": str(opponent or "unknown"),
                    "zone": str(baseline.get("zone", "unknown")),
                    "score_state": str(baseline.get("score_state", "unknown")),
                    "match_phase": str(baseline.get("match_phase", "unknown")),
                },
                "horizons": tuple(residuals),
                "residuals": residuals,
            })
    return rows


def _keys(row: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    context = row["context"]
    horizons = row["horizons"]
    suffix = ("horizons", *horizons)
    return (
        ("team_opponent_context", context["team_id"],
         context["opponent_team_id"], context["zone"],
         context["score_state"], context["match_phase"], *suffix),
        ("team_context", context["team_id"], context["zone"],
         context["score_state"], context["match_phase"], *suffix),
        ("context", context["zone"], context["score_state"],
         context["match_phase"], *suffix),
        ("team", context["team_id"], *suffix),
        ("global", *suffix),
    )


def _rank_correlation(matrix: np.ndarray) -> float:
    if matrix.shape[1] < 2:
        return 0.0
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    ranks = np.stack([
        average_ranks(matrix[:, index])
        for index in range(matrix.shape[1])
    ], axis=1)
    correlation = np.corrcoef(ranks, rowvar=False)
    if np.ndim(correlation) != 2 or not np.all(np.isfinite(correlation)):
        return 0.0
    upper = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
    return float(np.mean(upper)) if len(upper) else 0.0


def _fit(
    scope: str,
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
) -> TemporalRankCorrection | None:
    if len(rows) < min_samples:
        return None
    horizons = rows[0]["horizons"]
    if any(row["horizons"] != horizons for row in rows):
        return None
    split = len(rows) // 2
    reference = rows[:split]
    calibration = rows[split:]
    if len(reference) < 4 or len(calibration) < 4:
        return None
    reference_matrix = np.asarray([
        [row["residuals"][horizon] for horizon in horizons]
        for row in reference
    ], dtype=np.float64)
    calibration_matrix = np.asarray([
        [row["residuals"][horizon] for horizon in horizons]
        for row in calibration
    ], dtype=np.float64)
    templates = []
    levels = np.asarray(RANK_LEVELS, dtype=np.float64)
    for row in calibration_matrix:
        template = []
        for index, value in enumerate(row):
            reference_values = np.sort(reference_matrix[:, index])
            percentile = (
                np.searchsorted(reference_values, value, side="right") + 0.5
            ) / (len(reference_values) + 1.0)
            template.append(int(np.argmin(np.abs(levels - percentile))))
        templates.append(tuple(template))
    return TemporalRankCorrection(
        scope=scope,
        horizons=horizons,
        samples=len(rows),
        reference_samples=len(reference),
        calibration_samples=len(calibration),
        rank_templates=tuple(templates),
        mean_absolute_rank_correlation=_rank_correlation(reference_matrix),
    )


def compile_temporal_residual_rank_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
    min_samples: int,
    drift: dict[str, Any],
) -> TemporalResidualRankMemory:
    rows = _path_rows(
        logs,
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
    )
    minimum = max(8, int(min_samples))
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        for key in _keys(row):
            grouped.setdefault(key, []).append(row)
    groups = {}
    for key, group_rows in grouped.items():
        correction = _fit(key[0], group_rows, min_samples=minimum)
        if correction is not None:
            groups[key] = correction
    return TemporalResidualRankMemory(
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
        groups=groups,
        path_rows=len(rows),
        min_samples=minimum,
        drift=dict(drift),
    )
