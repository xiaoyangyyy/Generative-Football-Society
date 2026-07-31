"""Checkpoint-scoped contextual residual memory with split conformal intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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


@dataclass
class ContextualResidualMemory:
    checkpoint_signature: str
    groups: dict[tuple[str, ...], ResidualCorrection]
    rows: int
    source_logs: int
    min_samples: int

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
        for key in self._keys(context, str(action), str(horizon_key)):
            correction = self.groups.get(key)
            if correction is None:
                continue
            corrected = raw + correction.bias
            output["policy_utility"] = corrected
            output["residual_memory"] = {
                "scope": correction.scope,
                "samples": correction.samples,
                "calibration_samples": correction.calibration_samples,
                "bias_correction": correction.bias,
                "interval_90": [
                    corrected - correction.interval_radius_90,
                    corrected + correction.interval_radius_90,
                ],
                "interval_radius_90": correction.interval_radius_90,
                "validation_skill_vs_raw": correction.validation_skill_vs_raw,
                "trust_factor": correction.trust_factor,
                "checkpoint_signature": self.checkpoint_signature,
            }
            output["residual_memory_trust_factor"] = correction.trust_factor
            return output
        output["residual_memory"] = {
            "scope": "insufficient_contextual_history",
            "samples": 0,
            "trust_factor": 1.0,
            "checkpoint_signature": self.checkpoint_signature,
        }
        output["residual_memory_trust_factor"] = 1.0
        return output

    def summary(self) -> dict[str, Any]:
        scopes: dict[str, int] = {}
        for correction in self.groups.values():
            scopes[correction.scope] = scopes.get(correction.scope, 0) + 1
        return {
            "version": 1,
            "checkpoint_signature": self.checkpoint_signature,
            "source_logs": self.source_logs,
            "residual_rows": self.rows,
            "active_groups": len(self.groups),
            "groups_by_scope": scopes,
            "minimum_samples": self.min_samples,
            "policy": (
                "Use only same-checkpoint historical residuals; corrections "
                "and trust remain advisory and bounded."
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        report = self.summary()
        report["groups"] = [
            {"key": list(key), **asdict(correction)}
            for key, correction in sorted(self.groups.items())
        ]
        return report


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
) -> list[dict[str, Any]]:
    rows = []
    for payload in logs:
        adoption = payload.get("world_model_decision_adoption") or {}
        for record in adoption.get("records") or []:
            if str(record.get("checkpoint_signature")) != checkpoint_signature:
                continue
            if int(record.get("policy_utility_version", 0)) != 1:
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
    )


def compile_contextual_residual_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    min_samples: int = 8,
) -> ContextualResidualMemory:
    payloads = list(logs)
    minimum = max(8, int(min_samples))
    rows = _residual_rows(
        payloads, checkpoint_signature=checkpoint_signature,
    )
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
    return ContextualResidualMemory(
        checkpoint_signature=checkpoint_signature,
        groups=groups,
        rows=len(rows),
        source_logs=len(payloads),
        min_samples=minimum,
    )


def load_contextual_residual_memory(
    base_dir: str | Path,
    *,
    checkpoint_signature: str,
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
        min_samples=min_samples,
    )
