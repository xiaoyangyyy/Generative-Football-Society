"""Held-out calibration memory for opponent tactical information queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    TACTICAL_FEATURES,
)


@dataclass(frozen=True)
class OpponentInformationCalibration:
    scope: str
    horizon: str
    feature: str
    samples: int
    reference_samples: int
    validation_samples: int
    logit_offset: float
    raw_validation_brier: float
    calibrated_validation_brier: float
    validation_skill_vs_raw: float


@dataclass
class OpponentInformationCalibrationMemory:
    checkpoint_signature: str
    environment_signature: str
    groups: dict[tuple[str, ...], OpponentInformationCalibration]
    rows: int
    source_logs: int
    min_partition_samples: int

    def lookup(self, *, horizon: str, feature: str) -> dict[str, Any]:
        for key in (("horizon_feature", str(horizon), str(feature)),
                    ("feature", str(feature))):
            correction = self.groups.get(key)
            if correction is not None:
                return {
                    "available": True,
                    **asdict(correction),
                    "checkpoint_signature": self.checkpoint_signature,
                    "environment_signature": self.environment_signature,
                    "split": "chronological_reference_then_validation",
                    "causal_interpretation": False,
                }
        return {
            "available": False,
            "reason": "insufficient_or_unvalidated_history",
            "horizon": str(horizon),
            "feature": str(feature),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "causal_interpretation": False,
        }

    def contract(self, *, horizon: str) -> dict[str, Any]:
        return {
            "version": 1,
            "horizon": str(horizon),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "features": {
                feature: self.lookup(horizon=horizon, feature=feature)
                for feature in TACTICAL_FEATURES
            },
            "policy": (
                "Apply only offsets that improved chronological held-out "
                "Brier score; retain raw probabilities as the benchmark."
            ),
        }

    def summary(self) -> dict[str, Any]:
        scopes: dict[str, int] = {}
        for correction in self.groups.values():
            scopes[correction.scope] = scopes.get(correction.scope, 0) + 1
        return {
            "version": 1,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "rows": self.rows,
            "active_groups": len(self.groups),
            "groups_by_scope": scopes,
            "minimum_partition_samples": self.min_partition_samples,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "groups": [
                {"key": list(key), **asdict(value)}
                for key, value in sorted(self.groups.items())
            ],
        }


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def apply_logit_offset(probability: float, offset: float) -> float:
    value = _logit(probability) + float(np.clip(offset, -2.0, 2.0))
    return float(1.0 / (1.0 + math.exp(-value)))


def _calibrated_probability(row: dict[str, Any], offset: float) -> float:
    return float(np.dot(
        row["posterior"],
        [apply_logit_offset(value, offset) for value in row["likelihoods"]],
    ))


def _fit_offset(rows: list[dict[str, Any]]) -> float:
    target = float(np.mean([row["event"] for row in rows]))
    low, high = -2.0, 2.0
    for _ in range(60):
        middle = (low + high) / 2.0
        mean = float(np.mean([
            _calibrated_probability(row, middle)
            for row in rows
        ]))
        if mean < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _fit_group(
    scope: str,
    horizon: str,
    feature: str,
    rows: list[dict[str, Any]],
    *,
    min_partition_samples: int,
) -> OpponentInformationCalibration | None:
    if len(rows) < 2 * min_partition_samples:
        return None
    split = len(rows) // 2
    reference, validation = rows[:split], rows[split:]
    if len(reference) < min_partition_samples or len(validation) < min_partition_samples:
        return None
    offset = _fit_offset(reference)
    raw = float(np.mean([
        (row["raw_probability"] - row["event"]) ** 2
        for row in validation
    ]))
    calibrated = float(np.mean([
        (_calibrated_probability(row, offset) - row["event"]) ** 2
        for row in validation
    ]))
    skill = raw - calibrated
    if not math.isfinite(skill) or skill <= 1e-6:
        return None
    return OpponentInformationCalibration(
        scope=scope,
        horizon=horizon,
        feature=feature,
        samples=len(rows),
        reference_samples=len(reference),
        validation_samples=len(validation),
        logit_offset=float(offset),
        raw_validation_brier=raw,
        calibrated_validation_brier=calibrated,
        validation_skill_vs_raw=skill,
    )


def _rows(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> list[dict[str, Any]]:
    # Lazy imports avoid a module cycle: query evaluation consumes this memory,
    # while memory compilation must still reject altered audits and scores.
    from src.match_engine.world_model.opponent_information_query import (
        opponent_information_query_audit_is_valid,
    )
    from src.match_engine.world_model.opponent_information_query_outcomes import (
        opponent_information_query_score_is_valid,
    )

    output = []
    seen_match_groups: set[tuple[int, str, str]] = set()
    for match_index, payload in enumerate(logs):
        records = ((payload.get("world_model_decision_adoption") or {}).get(
            "records"
        ) or [])
        for record_index, record in enumerate(records):
            audit = record.get("llm_opponent_information_query_context") or {}
            if (
                int(audit.get("version", 0)) < 3
                or not opponent_information_query_audit_is_valid(audit)
            ):
                continue
            query = audit.get("query") or {}
            horizon = str(query.get("horizon", ""))
            score = ((record.get("multi_horizon_regime_outcomes") or {}).get(
                horizon
            ) or {}).get("llm_opponent_information_query_evaluation") or {}
            if (
                int(score.get("version", 0)) < 3
                or str(score.get("checkpoint_signature"))
                != str(checkpoint_signature)
                or str(score.get("environment_signature"))
                != str(environment_signature)
                or not score.get("issuance_evaluation_provenance_compatible")
                or not opponent_information_query_score_is_valid(score, audit)
            ):
                continue
            reports = {
                str(row.get("feature")): row
                for row in audit.get("query_leaderboard") or []
                if isinstance(row, dict)
            }
            events = score.get("observed_high_events") or {}
            posterior_payload = (audit.get("model_evidence") or {}).get(
                "posterior"
            ) or {}
            try:
                posterior = np.asarray([
                    float(posterior_payload[name])
                    for name in OPPONENT_HYPOTHESES
                ], dtype=np.float64)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (
                not np.all(np.isfinite(posterior))
                or np.any(posterior < 0.0)
                or abs(float(np.sum(posterior)) - 1.0) > 1e-6
            ):
                continue
            for feature in TACTICAL_FEATURES:
                report = reports.get(feature) or {}
                try:
                    probability = float(report["raw_forecast_high_rate"])
                    event = float(bool(events[feature]))
                    likelihoods = np.asarray([
                        float(report["raw_hypothesis_high_likelihoods"][name])
                        for name in OPPONENT_HYPOTHESES
                    ], dtype=np.float64)
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if (
                    math.isfinite(probability) and 0.0 <= probability <= 1.0
                    and np.all(np.isfinite(likelihoods))
                    and np.all((likelihoods >= 0.0) & (likelihoods <= 1.0))
                    and abs(float(np.dot(posterior, likelihoods)) - probability)
                    <= 1e-9
                ):
                    match_group = (match_index, horizon, feature)
                    if match_group in seen_match_groups:
                        continue
                    seen_match_groups.add(match_group)
                    output.append({
                        "order": (match_index, record_index),
                        "horizon": horizon,
                        "feature": feature,
                        "raw_probability": probability,
                        "posterior": posterior,
                        "likelihoods": likelihoods,
                        "event": event,
                    })
    return output


def compile_opponent_information_calibration_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
    min_samples: int = 8,
) -> OpponentInformationCalibrationMemory:
    payloads = list(logs)
    minimum = max(8, int(min_samples))
    rows = _rows(
        payloads,
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
    )
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    seen_feature_matches: set[tuple[int, str]] = set()
    for row in rows:
        grouped.setdefault(
            ("horizon_feature", row["horizon"], row["feature"]), []
        ).append(row)
        feature_match = (int(row["order"][0]), row["feature"])
        if feature_match not in seen_feature_matches:
            grouped.setdefault(("feature", row["feature"]), []).append(row)
            seen_feature_matches.add(feature_match)
    groups = {}
    for key, values in grouped.items():
        values.sort(key=lambda row: row["order"])
        correction = _fit_group(
            key[0], key[1] if key[0] == "horizon_feature" else "all",
            key[-1], values, min_partition_samples=minimum,
        )
        if correction is not None:
            groups[key] = correction
    return OpponentInformationCalibrationMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        groups=groups,
        rows=len(rows),
        source_logs=len(payloads),
        min_partition_samples=minimum,
    )
