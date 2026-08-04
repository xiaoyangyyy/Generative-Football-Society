"""Machine-owned cross-match memory for mechanism context stress tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.mechanism_stress_evaluation import (
    mechanism_stress_evaluation_is_valid,
)
from src.match_engine.world_model.mechanism_stress_test import (
    mechanism_stress_test_audit_is_valid,
)


MECHANISM_STRESS_MEMORY_VERSION = 1
STRESS_LOG_EVIDENCE_BOUNDARY = math.log(20.0)
STRESS_MAXIMUM_MATCHES = 20


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _status(matches: int, cumulative: float) -> str:
    if cumulative >= STRESS_LOG_EVIDENCE_BOUNDARY:
        return "validated_context_dependence"
    if cumulative <= -STRESS_LOG_EVIDENCE_BOUNDARY:
        return "invalidated_context_dependence"
    if matches >= STRESS_MAXIMUM_MATCHES:
        return "retired_inconclusive"
    return "start" if matches == 0 else "continue"


@dataclass(frozen=True)
class MechanismStressProfile:
    action: str
    horizon: str
    driver_event: str
    outcome_event: str
    relationship: str
    context_factor: str
    matches: int
    rows: int
    cumulative_log_likelihood_ratio: float
    mean_log_likelihood_ratio: float
    mean_brier_skill: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismStressMemory:
    checkpoint_signature: str
    environment_signature: str
    profiles: dict[tuple[str, ...], MechanismStressProfile]
    source_logs: int
    compatible_matches: int

    def evidence(
        self, *, action: str, horizon: str, driver_event: str,
        outcome_event: str, relationship: str, context_factor: str,
        checkpoint_signature: str, environment_signature: str,
    ) -> dict[str, Any]:
        scope_ok = bool(
            str(checkpoint_signature) == self.checkpoint_signature
            and str(environment_signature) == self.environment_signature
        )
        key = tuple(map(str, (
            action, horizon, driver_event, outcome_event, relationship,
            context_factor,
        )))
        profile = self.profiles.get(key) if scope_ok else None
        matches = int(profile.matches) if profile else 0
        cumulative = (
            float(profile.cumulative_log_likelihood_ratio) if profile else 0.0
        )
        payload = {
            "version": MECHANISM_STRESS_MEMORY_VERSION,
            "status": (
                profile.status if profile else "start" if scope_ok
                else "incompatible_checkpoint_or_environment"
            ),
            "profile_matches": matches,
            "cumulative_log_likelihood_ratio": cumulative,
            "mean_log_likelihood_ratio": (
                float(profile.mean_log_likelihood_ratio) if profile else 0.0
            ),
            "mean_brier_skill": (
                float(profile.mean_brier_skill) if profile else 0.0
            ),
            "positive_log_evidence_boundary": STRESS_LOG_EVIDENCE_BOUNDARY,
            "negative_log_evidence_boundary": -STRESS_LOG_EVIDENCE_BOUNDARY,
            "maximum_matches": STRESS_MAXIMUM_MATCHES,
            "profile_key": "|".join(key),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "can_change_current_action": False,
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "contract_digest": _digest(
                payload, "mechanism-stress-memory-contract:",
            ),
        }

    def summary(self) -> dict[str, Any]:
        payload = {
            "version": MECHANISM_STRESS_MEMORY_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "continuing_profiles": sum(
                row.status in {"start", "continue"}
                for row in self.profiles.values()
            ),
            "validated_profiles": sum(
                row.status == "validated_context_dependence"
                for row in self.profiles.values()
            ),
            "invalidated_profiles": sum(
                row.status == "invalidated_context_dependence"
                for row in self.profiles.values()
            ),
            "retired_inconclusive_profiles": sum(
                row.status == "retired_inconclusive"
                for row in self.profiles.values()
            ),
            "profiles": {
                "|".join(key): profile.to_dict()
                for key, profile in sorted(self.profiles.items())
            },
            "can_change_current_action": False,
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "memory_digest": _digest(payload, "mechanism-stress-memory:"),
        }


def compile_mechanism_stress_memory(
    logs: Iterable[dict[str, Any]], *, checkpoint_signature: str,
    environment_signature: str,
) -> MechanismStressMemory:
    payloads = list(logs)
    grouped: dict[tuple[str, ...], list[list[tuple[float, float]]]] = {}
    compatible_matches = 0
    for payload in payloads:
        per_match: dict[tuple[str, ...], list[tuple[float, float]]] = {}
        records = (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        for record in records:
            if (
                str(record.get("checkpoint_signature"))
                != str(checkpoint_signature)
                or str(record.get(
                    "environment_signature", "environment_unspecified",
                )) != str(environment_signature)
            ):
                continue
            audit = record.get("llm_mechanism_stress_test_context") or {}
            if not mechanism_stress_test_audit_is_valid(audit):
                continue
            test = audit["stress_test"]
            evaluation = (
                (record.get("multi_horizon_regime_outcomes") or {}).get(
                    str(test["horizon"]),
                ) or {}
            ).get("llm_mechanism_stress_test_evaluation") or {}
            if not mechanism_stress_evaluation_is_valid(evaluation, audit):
                continue
            try:
                log_ratio = float(evaluation[
                    "log_likelihood_ratio_observed_vs_neutralized"
                ])
                brier_skill = float(evaluation[
                    "brier_skill_observed_vs_neutralized"
                ])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(log_ratio) or not math.isfinite(brier_skill):
                continue
            key = tuple(map(str, (
                test["action"], test["horizon"], test["driver_event"],
                test["outcome_event"], test["observed_relationship"],
                test["context_factor"],
            )))
            per_match.setdefault(key, []).append((log_ratio, brier_skill))
        compatible_matches += int(bool(per_match))
        for key, rows in per_match.items():
            grouped.setdefault(key, []).append(rows)
    profiles = {}
    for key, match_rows in grouped.items():
        log_contributions = [
            float(np.mean([row[0] for row in rows]))
            for rows in match_rows if rows
        ]
        brier_contributions = [
            float(np.mean([row[1] for row in rows]))
            for rows in match_rows if rows
        ]
        cumulative = float(sum(log_contributions))
        matches = len(log_contributions)
        profiles[key] = MechanismStressProfile(
            action=key[0], horizon=key[1], driver_event=key[2],
            outcome_event=key[3], relationship=key[4],
            context_factor=key[5], matches=matches,
            rows=sum(len(rows) for rows in match_rows),
            cumulative_log_likelihood_ratio=cumulative,
            mean_log_likelihood_ratio=(
                cumulative / matches if matches else 0.0
            ),
            mean_brier_skill=(
                float(np.mean(brier_contributions))
                if brier_contributions else 0.0
            ),
            status=_status(matches, cumulative),
        )
    return MechanismStressMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature), profiles=profiles,
        source_logs=len(payloads), compatible_matches=compatible_matches,
    )


def load_mechanism_stress_memory(
    base_dir: str | Path, *, checkpoint_signature: str,
    environment_signature: str,
) -> MechanismStressMemory:
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
    return compile_mechanism_stress_memory(
        payloads, checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
    )


def mechanism_stress_memory_diagnostics(
    logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    payloads = list(logs)
    scopes = sorted({
        (
            str(record.get("checkpoint_signature", "")),
            str(record.get(
                "environment_signature", "environment_unspecified",
            )),
        )
        for payload in payloads
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        if record.get("llm_mechanism_stress_test_context")
    })
    memories = [
        compile_mechanism_stress_memory(
            payloads, checkpoint_signature=checkpoint,
            environment_signature=environment,
        )
        for checkpoint, environment in scopes if checkpoint and environment
    ]
    profiles = [
        profile for memory in memories for profile in memory.profiles.values()
    ]
    resolved = [
        row for row in profiles if row.status not in {"start", "continue"}
    ]
    return {
        "version": MECHANISM_STRESS_MEMORY_VERSION,
        "evaluation_kind": "match_clustered_mechanism_stress_memory",
        "scopes": len(memories), "profiles": len(profiles),
        "resolved_profiles": len(resolved),
        "validated_profiles": sum(
            row.status == "validated_context_dependence" for row in profiles
        ),
        "invalidated_profiles": sum(
            row.status == "invalidated_context_dependence" for row in profiles
        ),
        "retired_inconclusive_profiles": sum(
            row.status == "retired_inconclusive" for row in profiles
        ),
        "resolved_profile_matches": sum(row.matches for row in resolved),
        "all_checkpoint_environment_scoped": all(
            memory.checkpoint_signature and memory.environment_signature
            for memory in memories
        ),
        "all_match_clustered": all(row.matches <= row.rows for row in profiles),
        "all_stopping_rules_machine_owned": all(
            row.status == _status(
                row.matches, row.cumulative_log_likelihood_ratio,
            ) for row in profiles
        ),
        "can_change_current_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
