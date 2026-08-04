"""Chronologically validated cross-match memory for active probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.active_probe import (
    active_probe_audit_is_valid,
    active_probe_evaluation_is_valid,
)
from src.match_engine.world_model.active_probe_portfolio import (
    active_probe_portfolio_audit_is_valid,
)


ACTIVE_PROBE_MEMORY_VERSION = 1
SEQUENTIAL_LOG_EVIDENCE_BOUNDARY = math.log(20.0)
SEQUENTIAL_MAXIMUM_MATCHES = 20


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True)
class ActiveProbeDiscoveryProfile:
    action: str
    null_action: str
    horizon: str
    endpoint: str
    matches: int
    rows: int
    training_matches: int
    validation_matches: int
    calibration_offset: float
    raw_validation_brier: float
    corrected_validation_brier: float
    validation_skill: float
    train_observed_rate: float
    validation_observed_rate: float
    observed_rate_shift: float
    status: str
    authority: float
    cumulative_log_likelihood_ratio: float
    mean_log_likelihood_ratio: float
    sequential_status: str

    @property
    def active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "active": self.active}


@dataclass
class ActiveProbeDiscoveryMemory:
    checkpoint_signature: str
    environment_signature: str
    profiles: dict[tuple[str, str, str, str], ActiveProbeDiscoveryProfile]
    source_logs: int
    compatible_matches: int

    def profile(
        self,
        *,
        action: str,
        null_action: str,
        horizon: str,
        endpoint: str,
    ) -> ActiveProbeDiscoveryProfile | None:
        return self.profiles.get((
            str(action), str(null_action), str(horizon), str(endpoint),
        ))

    def calibration(
        self,
        *,
        action: str,
        null_action: str,
        horizon: str,
        endpoint: str,
        raw_probability: float,
        checkpoint_signature: str,
        environment_signature: str,
    ) -> dict[str, Any]:
        raw = min(1.0, max(0.0, float(raw_probability)))
        scope_ok = bool(
            str(checkpoint_signature) == self.checkpoint_signature
            and str(environment_signature) == self.environment_signature
        )
        profile = self.profile(
            action=action,
            null_action=null_action,
            horizon=horizon,
            endpoint=endpoint,
        ) if scope_ok else None
        active = bool(profile is not None and profile.active)
        adjustment = (
            float(profile.calibration_offset * profile.authority)
            if active else 0.0
        )
        calibrated = min(1.0 - 1e-6, max(1e-6, raw + adjustment))
        status = (
            profile.status if profile is not None
            else "insufficient_history" if scope_ok
            else "incompatible_checkpoint_or_environment"
        )
        payload = {
            "version": ACTIVE_PROBE_MEMORY_VERSION,
            "active": active,
            "status": status,
            "raw_probability": raw,
            "calibrated_probability": calibrated,
            "calibration_adjustment": calibrated - raw,
            "authority": float(profile.authority) if active else 0.0,
            "validation_skill": (
                float(profile.validation_skill) if profile is not None else 0.0
            ),
            "profile_matches": int(profile.matches) if profile else 0,
            "cumulative_log_likelihood_ratio": (
                float(profile.cumulative_log_likelihood_ratio)
                if profile is not None else 0.0
            ),
            "mean_log_likelihood_ratio": (
                float(profile.mean_log_likelihood_ratio)
                if profile is not None else 0.0
            ),
            "sequential_status": (
                profile.sequential_status if profile is not None else "start"
            ),
            "positive_log_evidence_boundary": (
                SEQUENTIAL_LOG_EVIDENCE_BOUNDARY
            ),
            "negative_log_evidence_boundary": (
                -SEQUENTIAL_LOG_EVIDENCE_BOUNDARY
            ),
            "maximum_sequential_matches": SEQUENTIAL_MAXIMUM_MATCHES,
            "profile_key": "|".join((
                str(action), str(null_action), str(horizon), str(endpoint),
            )),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "can_update_world_model": False,
            "can_change_current_action": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "contract_digest": _digest(
                payload, "active-probe-memory-contract:"
            ),
        }

    def summary(self) -> dict[str, Any]:
        payload = {
            "version": ACTIVE_PROBE_MEMORY_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "active_profiles": sum(
                profile.active for profile in self.profiles.values()
            ),
            "quarantined_profiles": sum(
                profile.status == "quarantined_drift"
                for profile in self.profiles.values()
            ),
            "maximum_authority": max((
                profile.authority for profile in self.profiles.values()
            ), default=0.0),
            "profiles": {
                "|".join(key): profile.to_dict()
                for key, profile in sorted(self.profiles.items())
            },
            "can_update_world_model": False,
            "causal_interpretation": False,
        }
        return {
            **payload,
            "memory_digest": _digest(payload, "active-probe-memory:"),
        }


def _match_rows(
    payload: dict[str, Any],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[tuple[str, str, str, str], list[dict[str, float]]]:
    grouped: dict[
        tuple[str, str, str, str], list[dict[str, float]]
    ] = {}
    adoption = payload.get("world_model_decision_adoption") or {}
    for record in adoption.get("records") or []:
        if (
            str(record.get("checkpoint_signature")) != checkpoint_signature
            or str(record.get(
                "environment_signature", "environment_unspecified",
            )) != environment_signature
        ):
            continue
        audit_rows = []
        single_audit = record.get("llm_active_probe_context") or {}
        if single_audit:
            audit_rows.append((single_audit, "llm_active_probe_evaluation"))
        portfolio_audit = record.get(
            "llm_active_probe_portfolio_context"
        ) or {}
        if active_probe_portfolio_audit_is_valid(portfolio_audit):
            audit_rows.extend((
                audit, "llm_active_probe_portfolio_evaluation"
            ) for audit in portfolio_audit["probe_audits"])
        for audit, evaluation_key in audit_rows:
            if not active_probe_audit_is_valid(audit):
                continue
            probe = audit["probe"]
            if (
                str(record.get("intervention_actual_action"))
                != str(probe["action"])
            ):
                continue
            outcome = (
                record.get("multi_horizon_regime_outcomes") or {}
            ).get(str(probe["horizon"])) or {}
            evaluation = outcome.get(evaluation_key) or {}
            if not active_probe_evaluation_is_valid(evaluation, audit):
                continue
            try:
                raw_probability = float(evaluation.get(
                    "raw_alternative_probability",
                    evaluation["alternative_probability"],
                ))
                observed = (
                    1.0 if bool(evaluation["observed_value"]) else 0.0
                )
                null_probability = float(evaluation.get(
                    "null_probability", probe["null_probability"],
                ))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if (
                not math.isfinite(raw_probability)
                or not math.isfinite(null_probability)
                or not 0.0 <= raw_probability <= 1.0
                or not 0.0 <= null_probability <= 1.0
            ):
                continue
            key = (
                str(probe["action"]), str(probe["null_action"]),
                str(probe["horizon"]), str(probe["endpoint"]),
            )
            grouped.setdefault(key, []).append({
                "raw_probability": raw_probability,
                "null_probability": null_probability,
                "observed": observed,
            })
    return grouped


def _match_means(
    matches: list[list[dict[str, float]]],
) -> list[dict[str, float]]:
    return [
        {
            "raw_probability": float(np.mean([
                row["raw_probability"] for row in rows
            ])),
            "null_probability": float(np.mean([
                row["null_probability"] for row in rows
            ])),
            "observed": float(np.mean([row["observed"] for row in rows])),
        }
        for rows in matches if rows
    ]


def _brier(rows: list[dict[str, float]], offset: float) -> float:
    if not rows:
        return 0.0
    return float(np.mean([
        (
            min(1.0 - 1e-6, max(
                1e-6, row["raw_probability"] + offset,
            )) - row["observed"]
        ) ** 2
        for row in rows
    ]))


def _bernoulli_log_likelihood_ratio(row: dict[str, float]) -> float:
    epsilon = 1e-6
    alternative = min(
        1.0 - epsilon, max(epsilon, row["raw_probability"]),
    )
    null = min(1.0 - epsilon, max(epsilon, row["null_probability"]))
    observed = min(1.0, max(0.0, row["observed"]))
    return float(
        observed * math.log(alternative / null)
        + (1.0 - observed) * math.log(
            (1.0 - alternative) / (1.0 - null)
        )
    )


def _sequential_status(matches: int, cumulative: float) -> str:
    if cumulative >= SEQUENTIAL_LOG_EVIDENCE_BOUNDARY:
        return "stop_supported"
    if cumulative <= -SEQUENTIAL_LOG_EVIDENCE_BOUNDARY:
        return "stop_falsified"
    if matches >= SEQUENTIAL_MAXIMUM_MATCHES:
        return "stop_inconclusive_maximum_matches"
    return "start" if matches == 0 else "continue"


def compile_active_probe_discovery_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
) -> ActiveProbeDiscoveryMemory:
    payloads = list(logs)
    by_profile: dict[
        tuple[str, str, str, str], list[list[dict[str, float]]]
    ] = {}
    compatible_matches = 0
    for payload in payloads:
        rows = _match_rows(
            payload,
            checkpoint_signature=str(checkpoint_signature),
            environment_signature=str(environment_signature),
        )
        compatible_matches += int(bool(rows))
        for key, values in rows.items():
            by_profile.setdefault(key, []).append(values)
    profiles = {}
    for key, matches in by_profile.items():
        split = max(1, len(matches) // 2)
        training = _match_means(matches[:split])
        validation = _match_means(matches[split:])
        offset = float(np.clip(np.mean([
            row["observed"] - row["raw_probability"] for row in training
        ]) if training else 0.0, -0.15, 0.15))
        raw_brier = _brier(validation, 0.0)
        corrected_brier = _brier(validation, offset)
        skill = (
            1.0 - corrected_brier / max(1e-12, raw_brier)
            if validation and raw_brier > 1e-12 else 0.0
        )
        train_rate = float(np.mean([
            row["observed"] for row in training
        ])) if training else 0.0
        validation_rate = float(np.mean([
            row["observed"] for row in validation
        ])) if validation else 0.0
        shift = abs(train_rate - validation_rate)
        enough = len(training) >= 4 and len(validation) >= 4
        if enough and shift > 0.30:
            status = "quarantined_drift"
        elif enough and abs(offset) > 1e-6 and skill >= 0.02:
            status = "active"
        elif enough:
            status = "no_held_out_gain"
        else:
            status = "insufficient_history"
        sample_factor = len(matches) / (len(matches) + 8.0)
        skill_factor = float(np.clip(skill / 0.25, 0.0, 1.0))
        authority = float(np.clip(
            0.35 * sample_factor * skill_factor
            if status == "active" else 0.0,
            0.0, 0.35,
        ))
        evidence_rows = _match_means(matches)
        cumulative_log_likelihood_ratio = float(sum(
            _bernoulli_log_likelihood_ratio(row) for row in evidence_rows
        ))
        mean_log_likelihood_ratio = (
            cumulative_log_likelihood_ratio / len(evidence_rows)
            if evidence_rows else 0.0
        )
        profiles[key] = ActiveProbeDiscoveryProfile(
            action=key[0],
            null_action=key[1],
            horizon=key[2],
            endpoint=key[3],
            matches=len(matches),
            rows=sum(len(rows) for rows in matches),
            training_matches=len(training),
            validation_matches=len(validation),
            calibration_offset=offset,
            raw_validation_brier=raw_brier,
            corrected_validation_brier=corrected_brier,
            validation_skill=skill,
            train_observed_rate=train_rate,
            validation_observed_rate=validation_rate,
            observed_rate_shift=shift,
            status=status,
            authority=authority,
            cumulative_log_likelihood_ratio=(
                cumulative_log_likelihood_ratio
            ),
            mean_log_likelihood_ratio=mean_log_likelihood_ratio,
            sequential_status=_sequential_status(
                len(matches), cumulative_log_likelihood_ratio,
            ),
        )
    return ActiveProbeDiscoveryMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        profiles=profiles,
        source_logs=len(payloads),
        compatible_matches=compatible_matches,
    )


def load_active_probe_discovery_memory(
    base_dir: str | Path,
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
) -> ActiveProbeDiscoveryMemory:
    directory = Path(base_dir) / "data" / "persistence" / "cognitive_log"
    payloads = []
    if directory.is_dir():
        def chronology(path: Path) -> tuple[int, str]:
            try:
                return path.stat().st_mtime_ns, path.name
            except OSError:
                return 0, path.name

        for path in sorted(
            directory.glob("*.json"),
            key=chronology,
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return compile_active_probe_discovery_memory(
        payloads,
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
    )


def active_probe_discovery_memory_diagnostics(
    logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    payloads = list(logs)
    source_audits = malformed_source_rows = 0
    for payload in payloads:
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get(
                "records"
            ) or []
        ):
            audit_rows = []
            single_audit = record.get("llm_active_probe_context") or {}
            if single_audit:
                audit_rows.append((
                    single_audit, "llm_active_probe_evaluation",
                ))
            portfolio_audit = record.get(
                "llm_active_probe_portfolio_context"
            ) or {}
            if portfolio_audit:
                if not active_probe_portfolio_audit_is_valid(portfolio_audit):
                    malformed_source_rows += 1
                else:
                    audit_rows.extend((
                        audit, "llm_active_probe_portfolio_evaluation"
                    ) for audit in portfolio_audit["probe_audits"])
            for audit, evaluation_key in audit_rows:
                source_audits += 1
                if not active_probe_audit_is_valid(audit):
                    malformed_source_rows += 1
                    continue
                probe = audit["probe"]
                outcome = (
                    record.get("multi_horizon_regime_outcomes") or {}
                ).get(str(probe["horizon"])) or {}
                evaluation = outcome.get(evaluation_key) or {}
                if evaluation and not active_probe_evaluation_is_valid(
                    evaluation, audit,
                ):
                    malformed_source_rows += 1
    scopes = sorted({
        (
            str(record.get("checkpoint_signature", "")),
            str(record.get(
                "environment_signature", "environment_unspecified",
            )),
        )
        for payload in payloads
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get(
                "records"
            ) or []
        )
        if (
            record.get("llm_active_probe_context")
            or record.get("llm_active_probe_portfolio_context")
        )
    })
    memories = [
        compile_active_probe_discovery_memory(
            payloads,
            checkpoint_signature=checkpoint,
            environment_signature=environment,
        )
        for checkpoint, environment in scopes
    ]
    profiles = [
        profile for memory in memories for profile in memory.profiles.values()
    ]
    active = [profile for profile in profiles if profile.active]
    return {
        "version": ACTIVE_PROBE_MEMORY_VERSION,
        "evaluation_kind": "chronological_active_probe_discovery_memory",
        "scopes": len(scopes),
        "profiles": len(profiles),
        "source_probe_audits": source_audits,
        "malformed_source_rows": malformed_source_rows,
        "active_profiles": len(active),
        "quarantined_profiles": sum(
            profile.status == "quarantined_drift" for profile in profiles
        ),
        "active_profile_matches": sum(profile.matches for profile in active),
        "minimum_active_validation_skill": min((
            profile.validation_skill for profile in active
        ), default=0.0),
        "maximum_authority": max((
            profile.authority for profile in profiles
        ), default=0.0),
        "all_checkpoint_environment_scoped": all(
            memory.checkpoint_signature and memory.environment_signature
            for memory in memories
        ),
        "all_authority_bounded": all(
            0.0 <= profile.authority <= 0.35 for profile in profiles
        ),
        "all_raw_forecasts_immutable": malformed_source_rows == 0,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
