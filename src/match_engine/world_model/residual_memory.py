"""Checkpoint-scoped contextual residual memory with split conformal intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.temporal_residual_memory import (
    TemporalResidualRankMemory,
    compile_temporal_residual_rank_memory,
)


@dataclass(frozen=True)
class ResidualCorrection:
    scope: str
    samples: int
    validation_samples: int
    calibration_samples: int
    bias: float
    interval_radius_90: float
    validation_skill_vs_raw: float
    trust_factor: float
    residual_quantile_levels: tuple[float, ...] = ()
    residual_quantiles: tuple[float, ...] = ()


@dataclass
class ContextualResidualMemory:
    checkpoint_signature: str
    environment_signature: str
    groups: dict[tuple[str, ...], ResidualCorrection]
    rows: int
    source_logs: int
    min_samples: int
    drift: dict[str, Any]
    temporal_rank_memory: TemporalResidualRankMemory

    def _keys(self, context: dict[str, Any], action: str, horizon: str):
        team = str(context.get("team_id", "unknown"))
        opponent = str(context.get("opponent_team_id", "unknown"))
        zone = str(context.get("zone", "unknown"))
        score = str(context.get("score_state", "unknown"))
        phase = str(context.get("match_phase", "unknown"))
        return (
            ("team_opponent_context", team, opponent, action, horizon, zone, score, phase),
            ("team_action", team, action, horizon),
            ("action_context", action, horizon, zone, score),
            ("action_horizon", action, horizon),
            ("horizon", horizon),
        )

    def calibrate(
        self,
        prediction: dict[str, Any],
        *,
        context: dict[str, Any],
        action: str,
        horizon_key: str,
    ) -> dict[str, Any]:
        output = dict(prediction)
        raw = float(output.get("raw_policy_utility", output["policy_utility"]))
        output["raw_policy_utility"] = raw
        if self.drift["status"] == "quarantined":
            output["policy_utility"] = raw
            output["residual_memory"] = {
                "scope": "quarantined_distribution_drift",
                "samples": self.rows,
                "trust_factor": 0.5,
                "checkpoint_signature": self.checkpoint_signature,
                "environment_signature": self.environment_signature,
                "drift": self.drift,
            }
            output["residual_memory_trust_factor"] = 0.5
            return output
        for key in self._keys(context, str(action), str(horizon_key)):
            correction = self.groups.get(key)
            if correction is None:
                continue
            watch = self.drift["status"] == "watch"
            bias_scale = 0.5 if watch else 1.0
            interval_scale = 1.5 if watch else 1.0
            corrected = raw + bias_scale * correction.bias
            trust = min(
                correction.trust_factor,
                0.75 if watch else 1.0,
            )
            output["policy_utility"] = corrected
            output["residual_memory"] = {
                "scope": correction.scope,
                "samples": correction.samples,
                "calibration_samples": correction.calibration_samples,
                "bias_correction": bias_scale * correction.bias,
                "interval_90": [
                    corrected - interval_scale * correction.interval_radius_90,
                    corrected + interval_scale * correction.interval_radius_90,
                ],
                "interval_radius_90": (
                    interval_scale * correction.interval_radius_90
                ),
                "validation_skill_vs_raw": correction.validation_skill_vs_raw,
                "residual_quantile_levels": list(
                    correction.residual_quantile_levels
                ),
                "residual_quantiles": _effective_residual_quantiles(
                    correction,
                    bias_scale=bias_scale,
                    spread_scale=interval_scale,
                ),
                "residual_quantiles_split": "held_out_calibration",
                "residual_quantiles_observed": True,
                "residual_quantiles_causal": False,
                "trust_factor": trust,
                "checkpoint_signature": self.checkpoint_signature,
                "environment_signature": self.environment_signature,
                "drift": self.drift,
            }
            output["residual_memory_trust_factor"] = trust
            return output
        output["residual_memory"] = {
            "scope": "insufficient_contextual_history",
            "samples": 0,
            "trust_factor": 1.0,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "drift": self.drift,
        }
        output["residual_memory_trust_factor"] = 1.0
        return output

    def summary(self) -> dict[str, Any]:
        scopes: dict[str, int] = {}
        for correction in self.groups.values():
            scopes[correction.scope] = scopes.get(correction.scope, 0) + 1
        return {
            "version": 4,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "residual_rows": self.rows,
            "active_groups": len(self.groups),
            "groups_by_scope": scopes,
            "minimum_samples": self.min_samples,
            "drift": self.drift,
            "temporal_residual_rank_memory": (
                self.temporal_rank_memory.summary()
            ),
            "policy": (
                "Use only same-checkpoint and same-policy-environment "
                "historical residuals; corrections and trust remain advisory "
                "and bounded."
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        report = self.summary()
        report["groups"] = [
            {"key": list(key), **asdict(correction)}
            for key, correction in sorted(self.groups.items())
        ]
        report["temporal_residual_rank_memory"] = (
            self.temporal_rank_memory.diagnostics()
        )
        return report

    def temporal_rank_coupling(
        self,
        *,
        context: dict[str, Any],
        horizon_keys: Iterable[str],
    ) -> dict[str, Any]:
        return self.temporal_rank_memory.lookup(
            context=context,
            horizon_keys=horizon_keys,
        )


def _contextual_keys(row: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    context = row["context"]
    action = row["action"]
    horizon = row["horizon_key"]
    team = context["team_id"]
    opponent = context["opponent_team_id"]
    zone = context["zone"]
    score = context["score_state"]
    phase = context["match_phase"]
    return (
        ("team_opponent_context", team, opponent, action, horizon, zone, score, phase),
        ("team_action", team, action, horizon),
        ("action_context", action, horizon, zone, score),
        ("action_horizon", action, horizon),
        ("horizon", horizon),
    )


def _residual_rows(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> list[dict[str, Any]]:
    rows = []
    for payload in logs:
        adoption = payload.get("world_model_decision_adoption") or {}
        for record in adoption.get("records") or []:
            if str(record.get("checkpoint_signature")) != checkpoint_signature:
                continue
            if int(record.get("policy_utility_version", 0)) != 1:
                continue
            if str(record.get(
                "environment_signature", "environment_unspecified",
            )) != environment_signature:
                continue
            baseline = record.get("outcome_baseline") or {}
            team = str(record.get("team_id", "unknown"))
            opponent = baseline.get("opponent_team_id")
            if not opponent:
                opponent = (
                    payload.get("away")
                    if team == payload.get("home") else payload.get("home")
                )
            context = {
                "team_id": team,
                "opponent_team_id": str(opponent or "unknown"),
                "zone": str(baseline.get("zone", "unknown")),
                "score_state": str(baseline.get("score_state", "unknown")),
                "match_phase": str(baseline.get("match_phase", "unknown")),
            }
            action = str(record.get("intervention_actual_action", "other"))
            outcomes = (
                record.get("multi_horizon_regime_outcomes")
                or record.get("multi_horizon_outcomes")
                or {}
            )
            for horizon_key, outcome in outcomes.items():
                prediction = outcome.get("world_model_prediction") or {}
                try:
                    predicted = float(prediction.get(
                        "raw_policy_utility", prediction["policy_utility"]
                    ))
                    actual = float(outcome["policy_utility"])
                    uncertainty = float(prediction.get("uncertainty", 1.0))
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in (
                    predicted, actual, uncertainty,
                )):
                    continue
                rows.append({
                    "context": context,
                    "action": action,
                    "horizon_key": str(horizon_key),
                    "predicted": predicted,
                    "actual": actual,
                    "uncertainty": float(np.clip(uncertainty, 0.05, 1.0)),
                })
    return rows


def policy_environment_signature(micro_cfg, cognitive_cfg) -> str:
    """Fingerprint decision-relevant simulator and bridge settings."""
    payload = {
        "policy_utility_version": 1,
        # Conservative isolation is intentional: physics, scheduling, affective
        # coupling, and cognitive cadence can all alter downstream policy value.
        "micro": asdict(micro_cfg),
        "cognitive": asdict(cognitive_cfg),
    }
    encoded = json.dumps(
        payload, sort_keys=True, default=str, separators=(",", ":"),
    ).encode("utf-8")
    return "policy-env:" + hashlib.sha256(encoded).hexdigest()[:24]


def _total_variation(
    reference: list[dict[str, Any]],
    recent: list[dict[str, Any]],
    key: str,
) -> float:
    values = sorted({
        str(row["context"].get(key, "unknown"))
        for row in [*reference, *recent]
    })
    if not values:
        return 0.0
    return 0.5 * sum(abs(
        sum(str(row["context"].get(key, "unknown")) == value for row in reference)
        / max(1, len(reference))
        - sum(str(row["context"].get(key, "unknown")) == value for row in recent)
        / max(1, len(recent))
    ) for value in values)


def detect_residual_drift(
    rows: list[dict[str, Any]],
    *,
    minimum_rows: int = 16,
) -> dict[str, Any]:
    if len(rows) < max(8, int(minimum_rows)):
        return {
            "status": "insufficient_history",
            "samples": len(rows),
            "minimum_samples": max(8, int(minimum_rows)),
            "memory_trust_ceiling": 1.0,
        }
    window = min(16, len(rows) // 2)
    reference = rows[-2 * window:-window]
    recent = rows[-window:]
    reference_residual = np.asarray([
        row["actual"] - row["predicted"] for row in reference
    ], dtype=float)
    recent_residual = np.asarray([
        row["actual"] - row["predicted"] for row in recent
    ], dtype=float)
    reference_scale = max(0.05, float(np.std(reference_residual)))
    recent_scale = max(0.05, float(np.std(recent_residual)))
    mean_shift = abs(float(
        np.mean(recent_residual) - np.mean(reference_residual)
    )) / reference_scale
    scale_ratio = max(
        recent_scale / reference_scale,
        reference_scale / recent_scale,
    )
    reference_rmse = math.sqrt(float(np.mean(np.square(reference_residual))))
    recent_rmse = math.sqrt(float(np.mean(np.square(recent_residual))))
    rmse_ratio = recent_rmse / max(0.05, reference_rmse)
    combined = np.sort(np.concatenate([reference_residual, recent_residual]))
    ks_distance = max((
        abs(
            float(np.mean(reference_residual <= threshold))
            - float(np.mean(recent_residual <= threshold))
        )
        for threshold in combined
    ), default=0.0)
    context_keys = ("zone", "score_state", "match_phase")
    context_tv = float(np.mean([
        _total_variation(reference, recent, key) for key in context_keys
    ]))
    quarantined = (
        mean_shift > 3.0
        or rmse_ratio > 3.0
        or (ks_distance > 0.55 and rmse_ratio > 1.5)
    )
    watch = (
        mean_shift > 1.5
        or scale_ratio > 2.0
        or rmse_ratio > 1.75
        or ks_distance > 0.40
        or context_tv > 0.45
    )
    status = "quarantined" if quarantined else "watch" if watch else "stable"
    return {
        "status": status,
        "samples": len(rows),
        "window_samples": window,
        "standardized_mean_shift": mean_shift,
        "scale_ratio": scale_ratio,
        "rmse_ratio": rmse_ratio,
        "ks_distance": ks_distance,
        "context_total_variation": context_tv,
        "memory_trust_ceiling": (
            0.5 if status == "quarantined"
            else 0.75 if status == "watch"
            else 1.0
        ),
        "recovery_policy": (
            "Re-evaluate adjacent rolling windows as fresh matches arrive."
        ),
    }


def _fit_group(
    scope: str,
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
) -> ResidualCorrection | None:
    if len(rows) < min_samples:
        return None
    training_end = max(1, len(rows) // 2)
    remaining = rows[training_end:]
    validation_end = max(1, len(remaining) // 2)
    training = rows[:training_end]
    validation = remaining[:validation_end]
    calibration = remaining[validation_end:]
    if len(validation) < 2 or len(calibration) < 2:
        return None
    training_residuals = np.asarray([
        row["actual"] - row["predicted"] for row in training
    ], dtype=float)
    proposed_bias = float(np.median(training_residuals))
    validation_actual = np.asarray([
        row["actual"] for row in validation
    ], dtype=float)
    validation_raw = np.asarray([
        row["predicted"] for row in validation
    ], dtype=float)
    raw_mse = float(np.mean(np.square(validation_actual - validation_raw)))
    corrected_mse = float(np.mean(np.square(
        validation_actual - (validation_raw + proposed_bias)
    )))
    bias = proposed_bias if corrected_mse <= raw_mse else 0.0
    actual = np.asarray([row["actual"] for row in calibration], dtype=float)
    raw = np.asarray([row["predicted"] for row in calibration], dtype=float)
    corrected_error = actual - (raw + bias)
    quantile_levels = (0.10, 0.25, 0.50, 0.75, 0.90)
    residual_quantiles = tuple(map(float, np.quantile(
        corrected_error, quantile_levels, method="linear",
    )))
    conformity = np.sort(np.abs(corrected_error))
    rank = int(math.ceil((len(conformity) + 1) * 0.90))
    radius = float(conformity[min(len(conformity) - 1, rank - 1)])
    final_mse = float(np.mean(np.square(corrected_error)))
    validation_final_mse = float(np.mean(np.square(
        validation_actual - (validation_raw + bias)
    )))
    skill_vs_raw = 1.0 - validation_final_mse / max(raw_mse, 1e-8)
    mean_uncertainty = float(np.mean([
        row["uncertainty"] for row in calibration
    ]))
    error_ratio = math.sqrt(final_mse) / max(0.05, mean_uncertainty)
    baseline_mse = float(np.mean(np.square(actual)))
    skill_vs_zero = 1.0 - final_mse / max(baseline_mse, 1e-6)
    error_factor = 1.0 / math.sqrt(max(1.0, error_ratio))
    skill_factor = 1.0 if skill_vs_zero >= 0.0 else math.exp(
        max(-4.0, skill_vs_zero)
    )
    trust = float(np.clip(min(error_factor, skill_factor), 0.25, 1.0))
    return ResidualCorrection(
        scope=scope,
        samples=len(rows),
        validation_samples=len(validation),
        calibration_samples=len(calibration),
        bias=bias,
        interval_radius_90=radius,
        validation_skill_vs_raw=skill_vs_raw,
        trust_factor=trust,
        residual_quantile_levels=quantile_levels,
        residual_quantiles=residual_quantiles,
    )


def _effective_residual_quantiles(
    correction: ResidualCorrection,
    *,
    bias_scale: float,
    spread_scale: float,
) -> list[float]:
    """Express held-out residual quantiles around the correction used now."""
    if not correction.residual_quantiles:
        return []
    values = np.asarray(correction.residual_quantiles, dtype=float)
    median_index = min(
        range(len(correction.residual_quantile_levels)),
        key=lambda index: abs(
            correction.residual_quantile_levels[index] - 0.50
        ),
    )
    median = float(values[median_index])
    # Watch mode applies only half of the learned bias and deliberately widens
    # the residual spread.  The omitted bias therefore remains in the signed
    # prediction error instead of silently disappearing.
    center = median + (1.0 - bias_scale) * correction.bias
    effective = center + spread_scale * (values - median)
    return list(map(float, effective))


def compile_contextual_residual_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
    min_samples: int = 8,
) -> ContextualResidualMemory:
    payloads = list(logs)
    minimum = max(8, int(min_samples))
    rows = _residual_rows(
        payloads,
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
    )
    drift = detect_residual_drift(rows)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        for key in _contextual_keys(row):
            grouped.setdefault(key, []).append(row)
    groups = {}
    for key, group_rows in grouped.items():
        correction = _fit_group(
            key[0], group_rows, min_samples=minimum,
        )
        if correction is not None:
            groups[key] = correction
    temporal_rank_memory = compile_temporal_residual_rank_memory(
        payloads,
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
        min_samples=minimum,
        drift=drift,
    )
    return ContextualResidualMemory(
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
        groups=groups,
        rows=len(rows),
        source_logs=len(payloads),
        min_samples=minimum,
        drift=drift,
        temporal_rank_memory=temporal_rank_memory,
    )


def load_contextual_residual_memory(
    base_dir: str | Path,
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
    min_samples: int = 8,
) -> ContextualResidualMemory:
    log_dir = Path(base_dir) / "data" / "persistence" / "cognitive_log"
    payloads = []
    if log_dir.is_dir():
        for path in sorted(log_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return compile_contextual_residual_memory(
        payloads,
        checkpoint_signature=checkpoint_signature,
        environment_signature=environment_signature,
        min_samples=min_samples,
    )
