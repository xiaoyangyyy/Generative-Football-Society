"""Match-clustered evidence memory for predictive mechanism hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.predictive_mechanism import (
    predictive_mechanism_audit_is_valid,
    predictive_mechanism_evaluation_is_valid,
)


PREDICTIVE_MECHANISM_MEMORY_VERSION = 1
MECHANISM_LOG_EVIDENCE_BOUNDARY = math.log(20.0)
MECHANISM_MAXIMUM_MATCHES = 20


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _status(matches: int, cumulative: float) -> str:
    if cumulative >= MECHANISM_LOG_EVIDENCE_BOUNDARY:
        return "retained_supported"
    if cumulative <= -MECHANISM_LOG_EVIDENCE_BOUNDARY:
        return "eliminated_falsified"
    if matches >= MECHANISM_MAXIMUM_MATCHES:
        return "retired_inconclusive"
    return "start" if matches == 0 else "continue"


@dataclass(frozen=True)
class PredictiveMechanismProfile:
    action: str
    horizon: str
    driver_event: str
    outcome_event: str
    relationship: str
    matches: int
    rows: int
    cumulative_log_likelihood_ratio: float
    mean_log_likelihood_ratio: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictiveMechanismMemory:
    checkpoint_signature: str
    environment_signature: str
    profiles: dict[tuple[str, str, str, str, str], PredictiveMechanismProfile]
    source_logs: int
    compatible_matches: int

    def evidence(
        self, *, action: str, horizon: str, driver_event: str,
        outcome_event: str, relationship: str, checkpoint_signature: str,
        environment_signature: str,
    ) -> dict[str, Any]:
        scope_ok = bool(
            str(checkpoint_signature) == self.checkpoint_signature
            and str(environment_signature) == self.environment_signature
        )
        key = (
            str(action), str(horizon), str(driver_event), str(outcome_event),
            str(relationship),
        )
        profile = self.profiles.get(key) if scope_ok else None
        matches = int(profile.matches) if profile else 0
        cumulative = (
            float(profile.cumulative_log_likelihood_ratio) if profile else 0.0
        )
        status = (
            profile.status if profile else "start" if scope_ok
            else "incompatible_checkpoint_or_environment"
        )
        payload = {
            "version": PREDICTIVE_MECHANISM_MEMORY_VERSION,
            "status": status,
            "profile_matches": matches,
            "cumulative_log_likelihood_ratio": cumulative,
            "mean_log_likelihood_ratio": (
                float(profile.mean_log_likelihood_ratio) if profile else 0.0
            ),
            "positive_log_evidence_boundary": MECHANISM_LOG_EVIDENCE_BOUNDARY,
            "negative_log_evidence_boundary": -MECHANISM_LOG_EVIDENCE_BOUNDARY,
            "maximum_matches": MECHANISM_MAXIMUM_MATCHES,
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
                payload, "predictive-mechanism-memory-contract:",
            ),
        }

    def summary(self) -> dict[str, Any]:
        payload = {
            "version": PREDICTIVE_MECHANISM_MEMORY_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "continuing_profiles": sum(
                profile.status in {"start", "continue"}
                for profile in self.profiles.values()
            ),
            "retained_profiles": sum(
                profile.status == "retained_supported"
                for profile in self.profiles.values()
            ),
            "eliminated_profiles": sum(
                profile.status == "eliminated_falsified"
                for profile in self.profiles.values()
            ),
            "retired_inconclusive_profiles": sum(
                profile.status == "retired_inconclusive"
                for profile in self.profiles.values()
            ),
            "profiles": {
                "|".join(key): profile.to_dict()
                for key, profile in sorted(self.profiles.items())
            },
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "memory_digest": _digest(payload, "predictive-mechanism-memory:"),
        }


def compile_predictive_mechanism_memory(
    logs: Iterable[dict[str, Any]], *, checkpoint_signature: str,
    environment_signature: str,
) -> PredictiveMechanismMemory:
    payloads = list(logs)
    grouped: dict[tuple[str, str, str, str, str], list[list[float]]] = {}
    compatible_matches = 0
    for payload in payloads:
        per_match: dict[tuple[str, str, str, str, str], list[float]] = {}
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
            audit = record.get("llm_predictive_mechanism_context") or {}
            if not predictive_mechanism_audit_is_valid(audit):
                continue
            hypothesis = audit["hypothesis"]
            evaluation = (
                (record.get("multi_horizon_regime_outcomes") or {}).get(
                    str(hypothesis["horizon"]),
                ) or {}
            ).get("llm_predictive_mechanism_evaluation") or {}
            if not predictive_mechanism_evaluation_is_valid(evaluation, audit):
                continue
            try:
                log_ratio = float(evaluation[
                    "categorical_log_likelihood_ratio_vs_independence"
                ])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(log_ratio):
                continue
            key = (
                str(hypothesis["action"]), str(hypothesis["horizon"]),
                str(hypothesis["driver_event"]),
                str(hypothesis["outcome_event"]),
                str(hypothesis["relationship"]),
            )
            per_match.setdefault(key, []).append(log_ratio)
        compatible_matches += int(bool(per_match))
        for key, rows in per_match.items():
            grouped.setdefault(key, []).append(rows)
    profiles = {}
    for key, match_rows in grouped.items():
        contributions = [float(np.mean(rows)) for rows in match_rows if rows]
        cumulative = float(sum(contributions))
        profiles[key] = PredictiveMechanismProfile(
            action=key[0], horizon=key[1], driver_event=key[2],
            outcome_event=key[3], matches=len(contributions),
            relationship=key[4],
            rows=sum(len(rows) for rows in match_rows),
            cumulative_log_likelihood_ratio=cumulative,
            mean_log_likelihood_ratio=(
                cumulative / len(contributions) if contributions else 0.0
            ),
            status=_status(len(contributions), cumulative),
        )
    return PredictiveMechanismMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature), profiles=profiles,
        source_logs=len(payloads), compatible_matches=compatible_matches,
    )


def load_predictive_mechanism_memory(
    base_dir: str | Path, *, checkpoint_signature: str,
    environment_signature: str,
) -> PredictiveMechanismMemory:
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
    return compile_predictive_mechanism_memory(
        payloads, checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
    )


def predictive_mechanism_memory_diagnostics(
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
        if record.get("llm_predictive_mechanism_context")
    })
    memories = [
        compile_predictive_mechanism_memory(
            payloads, checkpoint_signature=checkpoint,
            environment_signature=environment,
        )
        for checkpoint, environment in scopes if checkpoint and environment
    ]
    profiles = [
        profile for memory in memories for profile in memory.profiles.values()
    ]
    resolved = [
        profile for profile in profiles
        if profile.status in {
            "retained_supported", "eliminated_falsified",
            "retired_inconclusive",
        }
    ]
    return {
        "version": PREDICTIVE_MECHANISM_MEMORY_VERSION,
        "evaluation_kind": "match_clustered_predictive_mechanism_memory",
        "scopes": len(memories),
        "profiles": len(profiles),
        "resolved_profiles": len(resolved),
        "retained_profiles": sum(
            row.status == "retained_supported" for row in profiles
        ),
        "eliminated_profiles": sum(
            row.status == "eliminated_falsified" for row in profiles
        ),
        "retired_inconclusive_profiles": sum(
            row.status == "retired_inconclusive" for row in profiles
        ),
        "resolved_profile_matches": sum(row.matches for row in resolved),
        "all_checkpoint_environment_scoped": all(
            memory.checkpoint_signature and memory.environment_signature
            for memory in memories
        ),
        "all_match_clustered": all(
            row.matches <= row.rows for row in profiles
        ),
        "all_stopping_rules_machine_owned": all(
            row.status == _status(
                row.matches, row.cumulative_log_likelihood_ratio,
            ) for row in profiles
        ),
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
