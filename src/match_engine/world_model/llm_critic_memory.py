"""Cross-match reliability memory and diagnostics for semantic LLM critiques."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.llm_critic import LLM_CRITIC_VERSION
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS


@dataclass(frozen=True)
class LlmCriticProfile:
    action: str
    horizon: str
    matches: int
    rows: int
    training_matches: int
    validation_matches: int
    validation_rows: int
    fitted_scale: float
    baseline_mse: float
    corrected_mse: float
    validation_skill: float
    active: bool
    trust: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "horizon": self.horizon,
            "matches": self.matches,
            "rows": self.rows,
            "training_matches": self.training_matches,
            "validation_matches": self.validation_matches,
            "validation_rows": self.validation_rows,
            "fitted_scale": self.fitted_scale,
            "baseline_mse": self.baseline_mse,
            "corrected_mse": self.corrected_mse,
            "validation_skill": self.validation_skill,
            "active": self.active,
            "trust": self.trust,
        }


@dataclass
class LlmCriticMemory:
    checkpoint_signature: str
    environment_signature: str
    critic_signature: str
    profiles: dict[tuple[str, str], LlmCriticProfile]
    source_logs: int
    compatible_matches: int

    def profile(self, action: str, horizon: str) -> LlmCriticProfile | None:
        return self.profiles.get((str(action), str(horizon)))

    def authority(self, action: str, horizon: str) -> dict[str, Any]:
        profile = self.profile(action, horizon)
        active = bool(profile is not None and profile.active)
        return {
            "version": LLM_CRITIC_VERSION,
            "active": active,
            "scale": float(profile.fitted_scale) if active else 0.0,
            "trust": float(profile.trust) if active else 0.0,
            "reason": (
                "match_held_out_critic_gain"
                if active else "no_validated_critic_history"
            ),
            "profile": (
                profile.to_dict() if profile is not None else {
                    "action": str(action),
                    "horizon": str(horizon),
                    "active": False,
                }
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": LLM_CRITIC_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "critic_signature": self.critic_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "active_profiles": sum(
                profile.active for profile in self.profiles.values()
            ),
            "profiles": {
                f"{action}|{horizon}": profile.to_dict()
                for (action, horizon), profile in sorted(self.profiles.items())
            },
        }


def _match_rows(
    payload: dict[str, Any],
    *,
    checkpoint_signature: str,
    environment_signature: str,
    critic_signature: str,
) -> dict[tuple[str, str], list[dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    adoption = payload.get("world_model_decision_adoption") or {}
    for record in adoption.get("records") or []:
        if str(record.get("checkpoint_signature")) != checkpoint_signature:
            continue
        if str(record.get(
            "environment_signature", "environment_unspecified",
        )) != environment_signature:
            continue
        context = record.get("llm_world_model_critique_context") or {}
        critique = context.get("critique") or {}
        action = str(critique.get("action", ""))
        horizon = str(critique.get("horizon", ""))
        if (
            int(context.get("version", 0)) != LLM_CRITIC_VERSION
            or str(context.get(
                "critic_signature", "critic-unspecified",
            )) != critic_signature
            or action not in RESPONSE_ACTIONS
            or str(record.get("intervention_actual_action")) != action
        ):
            continue
        outcomes = record.get("multi_horizon_regime_outcomes") or {}
        outcome = outcomes.get(horizon) or {}
        prediction = outcome.get("world_model_prediction") or {}
        try:
            predicted = float(prediction.get(
                "raw_policy_utility", prediction.get("policy_utility"),
            ))
            actual = float(outcome["policy_utility"])
            proposal = float(context["proposed_policy_utility_correction"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (
            predicted, actual, proposal,
        )):
            continue
        grouped.setdefault((action, horizon), []).append({
            "residual": actual - predicted,
            "proposal": float(np.clip(proposal, -0.08, 0.08)),
        })
    return grouped


def _fit_scale(matches: list[list[dict[str, float]]]) -> float:
    numerator = denominator = 0.0
    for rows in matches:
        if not rows:
            continue
        numerator += float(np.mean([
            row["proposal"] * row["residual"] for row in rows
        ]))
        denominator += float(np.mean([
            row["proposal"] ** 2 for row in rows
        ]))
    return float(np.clip(
        numerator / max(1e-12, denominator), 0.0, 1.0,
    ))


def _errors(
    matches: list[list[dict[str, float]]], scale: float,
) -> tuple[float, float, int]:
    baseline, corrected = [], []
    rows_total = 0
    for rows in matches:
        if not rows:
            continue
        rows_total += len(rows)
        baseline.append(float(np.mean([
            row["residual"] ** 2 for row in rows
        ])))
        corrected.append(float(np.mean([
            (row["residual"] - scale * row["proposal"]) ** 2
            for row in rows
        ])))
    return (
        float(np.mean(baseline)) if baseline else 0.0,
        float(np.mean(corrected)) if corrected else 0.0,
        rows_total,
    )


def compile_llm_critic_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
    critic_signature: str = "critic-unspecified",
) -> LlmCriticMemory:
    payloads = list(logs)
    by_profile: dict[tuple[str, str], list[list[dict[str, float]]]] = {}
    compatible_matches = 0
    for payload in payloads:
        rows = _match_rows(
            payload,
            checkpoint_signature=str(checkpoint_signature),
            environment_signature=str(environment_signature),
            critic_signature=str(critic_signature),
        )
        if rows:
            compatible_matches += 1
        for key, match_rows in rows.items():
            by_profile.setdefault(key, []).append(match_rows)
    profiles = {}
    for (action, horizon), matches in by_profile.items():
        split = max(1, len(matches) // 2)
        training, validation = matches[:split], matches[split:]
        training_scale = _fit_scale(training)
        baseline_mse, corrected_mse, validation_rows = _errors(
            validation, training_scale,
        )
        skill = (
            1.0 - corrected_mse / max(1e-12, baseline_mse)
            if validation_rows and baseline_mse > 1e-12 else 0.0
        )
        active = bool(
            len(training) >= 2
            and len(validation) >= 2
            and validation_rows >= 2
            and training_scale > 0.0
            and skill >= 0.02
        )
        fitted_scale = _fit_scale(matches) if active else training_scale
        sample_factor = len(matches) / (len(matches) + 8.0)
        skill_factor = float(np.clip(skill / 0.25, 0.0, 1.0))
        trust = float(np.clip(
            0.35 * sample_factor * skill_factor if active else 0.0,
            0.0,
            0.35,
        ))
        profiles[(action, horizon)] = LlmCriticProfile(
            action=action,
            horizon=horizon,
            matches=len(matches),
            rows=sum(len(rows) for rows in matches),
            training_matches=len(training),
            validation_matches=len(validation),
            validation_rows=validation_rows,
            fitted_scale=fitted_scale,
            baseline_mse=baseline_mse,
            corrected_mse=corrected_mse,
            validation_skill=skill,
            active=active,
            trust=trust,
        )
    return LlmCriticMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        critic_signature=str(critic_signature),
        profiles=profiles,
        source_logs=len(payloads),
        compatible_matches=compatible_matches,
    )


def load_llm_critic_memory(
    base_dir: str | Path,
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
    critic_signature: str = "critic-unspecified",
) -> LlmCriticMemory:
    directory = Path(base_dir) / "data" / "persistence" / "cognitive_log"
    payloads = []
    if directory.is_dir():
        def chronology(path: Path) -> tuple[int, str]:
            try:
                return path.stat().st_mtime_ns, path.name
            except OSError:
                return 0, path.name

        for path in sorted(directory.glob("*.json"), key=chronology):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return compile_llm_critic_memory(
        payloads,
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        critic_signature=str(critic_signature),
    )


def llm_critic_diagnostics(logs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    audits = []
    match_baseline_errors = []
    match_corrected_errors = []
    realized = active_realized = 0
    for payload in logs:
        baseline_errors = []
        corrected_errors = []
        adoption = payload.get("world_model_decision_adoption") or {}
        for record in adoption.get("records") or []:
            context = record.get("llm_world_model_critique_context") or {}
            if not context:
                continue
            audits.append(context)
            critique = context.get("critique") or {}
            action = str(critique.get("action", ""))
            horizon = str(critique.get("horizon", ""))
            if str(record.get("intervention_actual_action")) != action:
                continue
            outcome = (record.get("multi_horizon_regime_outcomes") or {}).get(
                horizon
            ) or {}
            prediction = outcome.get("world_model_prediction") or {}
            try:
                predicted = float(prediction.get(
                    "raw_policy_utility", prediction.get("policy_utility"),
                ))
                actual = float(outcome["policy_utility"])
                correction = float(context.get(
                    "validated_policy_utility_correction",
                    context.get("applied_ranking_correction", 0.0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
            baseline_errors.append((actual - predicted) ** 2)
            corrected_errors.append((actual - predicted - correction) ** 2)
            realized += 1
            active_realized += bool(context.get("correction_applied"))
        if baseline_errors:
            match_baseline_errors.append(float(np.mean(baseline_errors)))
            match_corrected_errors.append(float(np.mean(corrected_errors)))
    baseline_mse = (
        float(np.mean(match_baseline_errors)) if match_baseline_errors else 0.0
    )
    corrected_mse = (
        float(np.mean(match_corrected_errors)) if match_corrected_errors else 0.0
    )
    return {
        "version": LLM_CRITIC_VERSION,
        "available": bool(audits),
        "critiques_registered": len(audits),
        "authority_active_critiques": sum(
            bool(audit.get("authority_active")) for audit in audits
        ),
        "realized_critiques": realized,
        "realized_authority_active_critiques": active_realized,
        "realized_validated_corrections": active_realized,
        "scored_matches": len(match_baseline_errors),
        "match_clustered_baseline_mse": baseline_mse,
        "match_clustered_corrected_mse": corrected_mse,
        "realized_skill": (
            1.0 - corrected_mse / max(1e-12, baseline_mse)
            if realized and baseline_mse > 1e-12 else 0.0
        ),
        "all_non_persistent": all(
            audit.get("can_update_world_model") is False
            and audit.get("can_update_critic_memory") is False
            for audit in audits
        ),
        "all_predictions_immutable": all(
            audit.get("world_model_prediction_mutated") is False
            for audit in audits
        ),
        "all_bridge_authority_non_increasing": all(
            0.50 <= float(audit.get("bridge_safety_factor", 1.0)) <= 1.0
            for audit in audits
        ),
        "causal_interpretation": False,
    }
