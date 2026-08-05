"""Held-out reliability memory for LLM/world-model chain forecast fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.predictive_mechanism_chain import (
    predictive_mechanism_chain_audit_is_valid,
)
from src.match_engine.world_model.predictive_mechanism_chain_evaluation import (
    predictive_mechanism_chain_evaluation_is_valid,
)


PREDICTIVE_MECHANISM_CHAIN_FUSION_MEMORY_VERSION = 1
MAXIMUM_LLM_WEIGHT = 0.35
MINIMUM_VALIDATION_SKILL = 0.02


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True)
class PredictiveChainFusionProfile:
    action: str
    horizon: str
    first_event: str
    mediator_event: str
    outcome_event: str
    matches: int
    rows: int
    training_matches: int
    validation_matches: int
    validation_rows: int
    fitted_llm_weight: float
    world_model_validation_mse: float
    llm_validation_mse: float
    fused_validation_mse: float
    validation_skill: float
    active: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictiveMechanismChainFusionMemory:
    checkpoint_signature: str
    environment_signature: str
    llm_signature: str
    profiles: dict[
        tuple[str, str, str, str, str], PredictiveChainFusionProfile
    ]
    source_logs: int
    compatible_matches: int

    def authority(
        self, *, action: str, horizon: str, first_event: str,
        mediator_event: str, outcome_event: str, checkpoint_signature: str,
        environment_signature: str, llm_signature: str,
    ) -> dict[str, Any]:
        key = tuple(map(str, (
            action, horizon, first_event, mediator_event, outcome_event,
        )))
        scope_ok = bool(
            str(checkpoint_signature) == self.checkpoint_signature
            and str(environment_signature) == self.environment_signature
            and str(llm_signature) == self.llm_signature
        )
        profile = self.profiles.get(key) if scope_ok else None
        active = bool(profile is not None and profile.active)
        payload = {
            "version": PREDICTIVE_MECHANISM_CHAIN_FUSION_MEMORY_VERSION,
            "active": active,
            "llm_weight": (
                float(profile.fitted_llm_weight) if active else 0.0
            ),
            "reason": (
                "match_held_out_chain_forecast_gain" if active
                else "incompatible_checkpoint_environment_or_llm"
                if not scope_ok else "no_match_held_out_chain_forecast_gain"
            ),
            "profile_matches": int(profile.matches) if profile else 0,
            "training_matches": (
                int(profile.training_matches) if profile else 0
            ),
            "validation_matches": (
                int(profile.validation_matches) if profile else 0
            ),
            "validation_skill": (
                float(profile.validation_skill) if profile else 0.0
            ),
            "authority_cap": MAXIMUM_LLM_WEIGHT,
            "profile_key": "|".join(key),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "llm_signature": self.llm_signature,
            "can_change_current_action": False,
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "contract_digest": _digest(
                payload, "predictive-chain-fusion-contract:",
            ),
        }

    def summary(self) -> dict[str, Any]:
        payload = {
            "version": PREDICTIVE_MECHANISM_CHAIN_FUSION_MEMORY_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "llm_signature": self.llm_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "active_profiles": sum(row.active for row in self.profiles.values()),
            "profiles": {
                "|".join(key): row.to_dict()
                for key, row in sorted(self.profiles.items())
            },
            "maximum_llm_weight": MAXIMUM_LLM_WEIGHT,
            "can_change_current_action": False,
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "memory_digest": _digest(
                payload, "predictive-chain-fusion-memory:",
            ),
        }


def _match_rows(
    payload: dict[str, Any], *, checkpoint_signature: str,
    environment_signature: str, llm_signature: str,
) -> dict[tuple[str, str, str, str, str], list[dict[str, float]]]:
    grouped: dict[
        tuple[str, str, str, str, str], list[dict[str, float]]
    ] = {}
    records = (
        (payload.get("world_model_decision_adoption") or {}).get("records")
        or []
    )
    for record in records:
        if (
            str(record.get("checkpoint_signature")) != checkpoint_signature
            or str(record.get(
                "environment_signature", "environment_unspecified",
            )) != environment_signature
        ):
            continue
        audit = record.get("llm_predictive_mechanism_chain_context") or {}
        if (
            not predictive_mechanism_chain_audit_is_valid(audit)
            or str(audit.get("llm_signature")) != llm_signature
        ):
            continue
        chain = audit["chain"]
        evaluation = (
            (record.get("multi_horizon_regime_outcomes") or {}).get(
                str(chain["horizon"]),
            ) or {}
        ).get("llm_predictive_mechanism_chain_evaluation") or {}
        if not predictive_mechanism_chain_evaluation_is_valid(
            evaluation, audit,
        ):
            continue
        try:
            probabilities = evaluation["chain_completion_probabilities"]
            row = {
                "target": float(evaluation["chain_completion_observed"]),
                "world_model": float(probabilities["world_model"]),
                "llm": float(probabilities["llm"]),
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in row.values()):
            continue
        key = tuple(str(chain[name]) for name in (
            "action", "horizon", "first_event", "mediator_event",
            "outcome_event",
        ))
        grouped.setdefault(key, []).append(row)
    return grouped


def _fit_weight(matches: list[list[dict[str, float]]]) -> float:
    numerator = denominator = 0.0
    for rows in matches:
        if not rows:
            continue
        numerator += float(np.mean([
            (row["llm"] - row["world_model"])
            * (row["target"] - row["world_model"])
            for row in rows
        ]))
        denominator += float(np.mean([
            (row["llm"] - row["world_model"]) ** 2 for row in rows
        ]))
    return float(np.clip(
        numerator / max(1e-12, denominator), 0.0, MAXIMUM_LLM_WEIGHT,
    ))


def _errors(
    matches: list[list[dict[str, float]]], weight: float,
) -> tuple[float, float, float, int]:
    world_errors = []
    llm_errors = []
    fused_errors = []
    rows_total = 0
    for rows in matches:
        if not rows:
            continue
        rows_total += len(rows)
        world_errors.append(float(np.mean([
            (row["target"] - row["world_model"]) ** 2 for row in rows
        ])))
        llm_errors.append(float(np.mean([
            (row["target"] - row["llm"]) ** 2 for row in rows
        ])))
        fused_errors.append(float(np.mean([
            (
                row["target"] - row["world_model"]
                - weight * (row["llm"] - row["world_model"])
            ) ** 2 for row in rows
        ])))
    return tuple(float(np.mean(values)) if values else 0.0 for values in (
        world_errors, llm_errors, fused_errors,
    )) + (rows_total,)


def compile_predictive_mechanism_chain_fusion_memory(
    logs: Iterable[dict[str, Any]], *, checkpoint_signature: str,
    environment_signature: str, llm_signature: str,
) -> PredictiveMechanismChainFusionMemory:
    payloads = list(logs)
    by_profile: dict[
        tuple[str, str, str, str, str], list[list[dict[str, float]]]
    ] = {}
    compatible_matches = 0
    for payload in payloads:
        rows = _match_rows(
            payload, checkpoint_signature=str(checkpoint_signature),
            environment_signature=str(environment_signature),
            llm_signature=str(llm_signature),
        )
        compatible_matches += int(bool(rows))
        for key, match_rows in rows.items():
            by_profile.setdefault(key, []).append(match_rows)
    profiles = {}
    for key, matches in by_profile.items():
        split = max(1, len(matches) // 2)
        training, validation = matches[:split], matches[split:]
        weight = _fit_weight(training)
        world_mse, llm_mse, fused_mse, validation_rows = _errors(
            validation, weight,
        )
        skill = (
            1.0 - fused_mse / max(1e-12, world_mse)
            if validation_rows and world_mse > 1e-12 else 0.0
        )
        active = bool(
            len(training) >= 2 and len(validation) >= 2
            and validation_rows >= 2 and weight > 0.0
            and skill >= MINIMUM_VALIDATION_SKILL
        )
        fitted_weight = _fit_weight(matches) if active else weight
        profiles[key] = PredictiveChainFusionProfile(
            action=key[0], horizon=key[1], first_event=key[2],
            mediator_event=key[3], outcome_event=key[4],
            matches=len(matches), rows=sum(map(len, matches)),
            training_matches=len(training), validation_matches=len(validation),
            validation_rows=validation_rows,
            fitted_llm_weight=fitted_weight,
            world_model_validation_mse=world_mse,
            llm_validation_mse=llm_mse,
            fused_validation_mse=fused_mse,
            validation_skill=skill, active=active,
        )
    return PredictiveMechanismChainFusionMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        llm_signature=str(llm_signature), profiles=profiles,
        source_logs=len(payloads), compatible_matches=compatible_matches,
    )


def load_predictive_mechanism_chain_fusion_memory(
    base_dir: str | Path, *, checkpoint_signature: str,
    environment_signature: str, llm_signature: str,
) -> PredictiveMechanismChainFusionMemory:
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
    return compile_predictive_mechanism_chain_fusion_memory(
        payloads, checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        llm_signature=str(llm_signature),
    )


def predictive_mechanism_chain_fusion_diagnostics(
    logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    payloads = list(logs)
    scopes = sorted({
        (
            str(record.get("checkpoint_signature", "")),
            str(record.get(
                "environment_signature", "environment_unspecified",
            )),
            str((record.get(
                "llm_predictive_mechanism_chain_context"
            ) or {}).get("llm_signature", "")),
        )
        for payload in payloads
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        if record.get("llm_predictive_mechanism_chain_context")
    })
    memories = [
        compile_predictive_mechanism_chain_fusion_memory(
            payloads, checkpoint_signature=checkpoint,
            environment_signature=environment, llm_signature=signature,
        )
        for checkpoint, environment, signature in scopes
        if checkpoint and environment and signature
    ]
    profiles = [row for memory in memories for row in memory.profiles.values()]
    active = [row for row in profiles if row.active]
    return {
        "version": PREDICTIVE_MECHANISM_CHAIN_FUSION_MEMORY_VERSION,
        "evaluation_kind": "held_out_llm_world_model_chain_forecast_fusion",
        "scopes": len(memories), "profiles": len(profiles),
        "active_profiles": len(active),
        "compatible_matches": sum(
            memory.compatible_matches for memory in memories
        ),
        "active_profile_matches": sum(row.matches for row in active),
        "active_validation_matches": sum(
            row.validation_matches for row in active
        ),
        "mean_active_validation_skill": (
            float(np.mean([row.validation_skill for row in active]))
            if active else 0.0
        ),
        "all_chronological_match_held_out": all(
            row.training_matches >= 2 and row.validation_matches >= 2
            for row in active
        ),
        "all_active_improve_world_model": all(
            row.validation_skill >= MINIMUM_VALIDATION_SKILL for row in active
        ),
        "llm_comparator_reported": all(
            math.isfinite(row.llm_validation_mse) for row in profiles
        ),
        "all_authority_bounded": all(
            0.0 < row.fitted_llm_weight <= MAXIMUM_LLM_WEIGHT for row in active
        ),
        "all_checkpoint_environment_llm_scoped": all(
            memory.checkpoint_signature and memory.environment_signature
            and memory.llm_signature for memory in memories
        ),
        "can_change_current_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
