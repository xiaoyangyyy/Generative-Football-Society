"""Match-clustered sequential evidence for higher-order predictive chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.predictive_mechanism_chain import (
    CHAIN_LOG_EVIDENCE_BOUNDARY,
    CHAIN_MAXIMUM_MATCHES,
    predictive_mechanism_chain_audit_is_valid,
)
from src.match_engine.world_model.predictive_mechanism_chain_evaluation import (
    predictive_mechanism_chain_evaluation_is_valid,
)


PREDICTIVE_MECHANISM_CHAIN_MEMORY_VERSION = 1


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _status(matches: int, cumulative: float) -> str:
    if cumulative >= CHAIN_LOG_EVIDENCE_BOUNDARY:
        return "retained_higher_order_dependence"
    if cumulative <= -CHAIN_LOG_EVIDENCE_BOUNDARY:
        return "eliminated_markov_sufficient"
    if matches >= CHAIN_MAXIMUM_MATCHES:
        return "retired_inconclusive"
    return "start" if matches == 0 else "continue"


@dataclass(frozen=True)
class PredictiveMechanismChainProfile:
    action: str
    horizon: str
    first_event: str
    mediator_event: str
    outcome_event: str
    matches: int
    rows: int
    cumulative_log_likelihood_ratio: float
    mean_log_likelihood_ratio: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictiveMechanismChainMemory:
    checkpoint_signature: str
    environment_signature: str
    profiles: dict[
        tuple[str, str, str, str, str], PredictiveMechanismChainProfile
    ]
    source_logs: int
    compatible_matches: int

    def evidence(
        self, *, action: str, horizon: str, first_event: str,
        mediator_event: str, outcome_event: str, checkpoint_signature: str,
        environment_signature: str,
    ) -> dict[str, Any]:
        scope_ok = bool(
            str(checkpoint_signature) == self.checkpoint_signature
            and str(environment_signature) == self.environment_signature
        )
        key = tuple(map(str, (
            action, horizon, first_event, mediator_event, outcome_event,
        )))
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
            "version": PREDICTIVE_MECHANISM_CHAIN_MEMORY_VERSION,
            "status": status,
            "profile_matches": matches,
            "cumulative_log_likelihood_ratio": cumulative,
            "mean_log_likelihood_ratio": (
                float(profile.mean_log_likelihood_ratio) if profile else 0.0
            ),
            "positive_log_evidence_boundary": CHAIN_LOG_EVIDENCE_BOUNDARY,
            "negative_log_evidence_boundary": -CHAIN_LOG_EVIDENCE_BOUNDARY,
            "maximum_matches": CHAIN_MAXIMUM_MATCHES,
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
                payload, "predictive-chain-memory-contract:",
            ),
        }

    def summary(self) -> dict[str, Any]:
        payload = {
            "version": PREDICTIVE_MECHANISM_CHAIN_MEMORY_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "continuing_profiles": sum(
                row.status in {"start", "continue"}
                for row in self.profiles.values()
            ),
            "retained_profiles": sum(
                row.status == "retained_higher_order_dependence"
                for row in self.profiles.values()
            ),
            "eliminated_profiles": sum(
                row.status == "eliminated_markov_sufficient"
                for row in self.profiles.values()
            ),
            "retired_inconclusive_profiles": sum(
                row.status == "retired_inconclusive"
                for row in self.profiles.values()
            ),
            "profiles": {
                "|".join(key): row.to_dict()
                for key, row in sorted(self.profiles.items())
            },
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "memory_digest": _digest(
                payload, "predictive-mechanism-chain-memory:",
            ),
        }


def compile_predictive_mechanism_chain_memory(
    logs: Iterable[dict[str, Any]], *, checkpoint_signature: str,
    environment_signature: str,
) -> PredictiveMechanismChainMemory:
    payloads = list(logs)
    grouped: dict[
        tuple[str, str, str, str, str], list[list[float]]
    ] = {}
    compatible_matches = 0
    for payload in payloads:
        per_match: dict[
            tuple[str, str, str, str, str], list[float]
        ] = {}
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
            audit = record.get("llm_predictive_mechanism_chain_context") or {}
            if not predictive_mechanism_chain_audit_is_valid(audit):
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
                log_ratio = float(evaluation[
                    "categorical_log_likelihood_ratio_vs_pairwise_markov_null"
                ])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(log_ratio):
                continue
            key = tuple(str(chain[name]) for name in (
                "action", "horizon", "first_event", "mediator_event",
                "outcome_event",
            ))
            per_match.setdefault(key, []).append(log_ratio)
        compatible_matches += int(bool(per_match))
        for key, rows in per_match.items():
            grouped.setdefault(key, []).append(rows)
    profiles = {}
    for key, match_rows in grouped.items():
        contributions = [float(np.mean(rows)) for rows in match_rows if rows]
        cumulative = float(sum(contributions))
        profiles[key] = PredictiveMechanismChainProfile(
            action=key[0], horizon=key[1], first_event=key[2],
            mediator_event=key[3], outcome_event=key[4],
            matches=len(contributions), rows=sum(map(len, match_rows)),
            cumulative_log_likelihood_ratio=cumulative,
            mean_log_likelihood_ratio=(
                cumulative / len(contributions) if contributions else 0.0
            ), status=_status(len(contributions), cumulative),
        )
    return PredictiveMechanismChainMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature), profiles=profiles,
        source_logs=len(payloads), compatible_matches=compatible_matches,
    )


def load_predictive_mechanism_chain_memory(
    base_dir: str | Path, *, checkpoint_signature: str,
    environment_signature: str,
) -> PredictiveMechanismChainMemory:
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
    return compile_predictive_mechanism_chain_memory(
        payloads, checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
    )


def predictive_mechanism_chain_memory_diagnostics(
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
        if record.get("llm_predictive_mechanism_chain_context")
    })
    memories = [
        compile_predictive_mechanism_chain_memory(
            payloads, checkpoint_signature=checkpoint,
            environment_signature=environment,
        )
        for checkpoint, environment in scopes if checkpoint and environment
    ]
    profiles = [
        row for memory in memories for row in memory.profiles.values()
    ]
    resolved = [row for row in profiles if row.status in {
        "retained_higher_order_dependence",
        "eliminated_markov_sufficient", "retired_inconclusive",
    }]
    return {
        "version": PREDICTIVE_MECHANISM_CHAIN_MEMORY_VERSION,
        "evaluation_kind": "match_clustered_predictive_chain_memory",
        "scopes": len(memories), "profiles": len(profiles),
        "resolved_profiles": len(resolved),
        "retained_profiles": sum(
            row.status == "retained_higher_order_dependence"
            for row in profiles
        ),
        "eliminated_profiles": sum(
            row.status == "eliminated_markov_sufficient" for row in profiles
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
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
