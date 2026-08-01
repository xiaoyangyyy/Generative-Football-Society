"""Cross-match opponent-style priors isolated by model and policy regime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    observation_only_tactic_posterior,
)


OPPONENT_META_MEMORY_VERSION = 1


def _normalise(values) -> np.ndarray:
    clean = np.clip(np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
    ), 0.0, None)
    total = float(clean.sum())
    if total <= 1e-12:
        return np.full(len(OPPONENT_HYPOTHESES), 1.0 / len(OPPONENT_HYPOTHESES))
    return clean / total


def _posterior_vector(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, dict):
        return None
    try:
        values = np.asarray([
            float(raw.get(name, 0.0)) for name in OPPONENT_HYPOTHESES
        ], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(values).all() or float(values.sum()) <= 1e-12:
        return None
    return _normalise(values)


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * float(np.abs(left - right).sum())


@dataclass(frozen=True)
class OpponentMetaProfile:
    opponent_team_id: str
    matches: int
    decision_snapshots: int
    observation_grounded_matches: int
    posterior_fallback_matches: int
    empirical_posterior: dict[str, float]
    prior: dict[str, float]
    trust: float
    between_match_instability: float
    drift_status: str
    last_match_posterior: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_team_id": self.opponent_team_id,
            "matches": self.matches,
            "decision_snapshots": self.decision_snapshots,
            "observation_grounded_matches": self.observation_grounded_matches,
            "posterior_fallback_matches": self.posterior_fallback_matches,
            "empirical_posterior": dict(self.empirical_posterior),
            "prior": dict(self.prior),
            "trust": self.trust,
            "between_match_instability": self.between_match_instability,
            "drift_status": self.drift_status,
            "last_match_posterior": dict(self.last_match_posterior),
        }


@dataclass
class OpponentMetaBeliefMemory:
    checkpoint_signature: str
    environment_signature: str
    profiles: dict[str, OpponentMetaProfile]
    source_logs: int
    compatible_matches: int

    def prior_for(self, opponent_team_id: str) -> dict[str, Any]:
        profile = self.profiles.get(str(opponent_team_id))
        if profile is None:
            uniform = 1.0 / len(OPPONENT_HYPOTHESES)
            return {
                "available": False,
                "reason": "no_compatible_opponent_history",
                "opponent_team_id": str(opponent_team_id),
                "prior": {name: uniform for name in OPPONENT_HYPOTHESES},
                "trust": 0.0,
                "matches": 0,
                "checkpoint_signature": self.checkpoint_signature,
                "environment_signature": self.environment_signature,
            }
        return {
            "version": OPPONENT_META_MEMORY_VERSION,
            "available": True,
            **profile.to_dict(),
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "policy": (
                "Historical matches seed a weak prior only; live evidence and "
                "confirmed tactical changes can override it."
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": OPPONENT_META_MEMORY_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "opponents": len(self.profiles),
            "profiles": {
                key: value.to_dict() for key, value in sorted(self.profiles.items())
            },
        }


def _match_distributions(
    payload: dict[str, Any],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, tuple[np.ndarray, int, bool]]:
    """Collapse correlated within-match decisions to one row per opponent."""
    adoption = payload.get("world_model_decision_adoption") or {}
    by_opponent: dict[str, list[tuple[np.ndarray, bool]]] = {}
    for record in adoption.get("records") or []:
        if str(record.get("checkpoint_signature")) != checkpoint_signature:
            continue
        if str(record.get(
            "environment_signature", "environment_unspecified",
        )) != environment_signature:
            continue
        context = record.get("opponent_belief_context") or {}
        opponent = str(context.get("opponent_team_id", "")).strip()
        grounded = False
        try:
            posterior = _posterior_vector(observation_only_tactic_posterior(
                context.get("observed_feature_vector")
            ))
            grounded = posterior is not None
        except (TypeError, ValueError):
            posterior = _posterior_vector(context.get("posterior"))
        if not opponent or posterior is None:
            continue
        by_opponent.setdefault(opponent, []).append((posterior, grounded))
    return {
        opponent: (
            _normalise(np.mean([row[0] for row in rows], axis=0)),
            len(rows),
            sum(row[1] for row in rows) >= max(1, len(rows) / 2),
        )
        for opponent, rows in by_opponent.items()
    }


def compile_opponent_meta_belief_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
) -> OpponentMetaBeliefMemory:
    payloads = list(logs)
    matches_by_opponent: dict[str, list[tuple[np.ndarray, int, bool]]] = {}
    compatible_matches = 0
    for payload in payloads:
        rows = _match_distributions(
            payload,
            checkpoint_signature=str(checkpoint_signature),
            environment_signature=str(environment_signature),
        )
        if rows:
            compatible_matches += 1
        for opponent, row in rows.items():
            matches_by_opponent.setdefault(opponent, []).append(row)

    uniform = np.full(len(OPPONENT_HYPOTHESES), 1.0 / len(OPPONENT_HYPOTHESES))
    profiles: dict[str, OpponentMetaProfile] = {}
    for opponent, match_rows in matches_by_opponent.items():
        distributions = [row[0] for row in match_rows]
        empirical = _normalise(np.mean(distributions, axis=0))
        instability = float(np.mean([
            _total_variation(row, empirical) for row in distributions
        ]))
        recent_shift = (
            _total_variation(distributions[-1], _normalise(np.mean(
                distributions[:-1], axis=0,
            )))
            if len(distributions) >= 3 else 0.0
        )
        drift_status = (
            "quarantined" if recent_shift >= 0.55
            else "watch" if recent_shift >= 0.35 or instability >= 0.30
            else "stable"
        )
        sample_factor = len(distributions) / (len(distributions) + 4.0)
        consistency_factor = float(np.clip(1.0 - instability, 0.25, 1.0))
        observation_grounded_matches = sum(row[2] for row in match_rows)
        grounding_factor = 0.50 + 0.50 * (
            observation_grounded_matches / len(match_rows)
        )
        drift_factor = (
            0.20 if drift_status == "quarantined"
            else 0.55 if drift_status == "watch" else 1.0
        )
        # Historical evidence is deliberately weaker than a single live update.
        trust = float(np.clip(
            0.60 * sample_factor * consistency_factor * drift_factor
            * grounding_factor,
            0.0,
            0.45,
        ))
        prior = _normalise(trust * empirical + (1.0 - trust) * uniform)
        profiles[opponent] = OpponentMetaProfile(
            opponent_team_id=opponent,
            matches=len(distributions),
            decision_snapshots=sum(row[1] for row in match_rows),
            observation_grounded_matches=observation_grounded_matches,
            posterior_fallback_matches=(
                len(match_rows) - observation_grounded_matches
            ),
            empirical_posterior={
                name: float(empirical[index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
            prior={
                name: float(prior[index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
            trust=trust,
            between_match_instability=instability,
            drift_status=drift_status,
            last_match_posterior={
                name: float(distributions[-1][index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
        )
    return OpponentMetaBeliefMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        profiles=profiles,
        source_logs=len(payloads),
        compatible_matches=compatible_matches,
    )


def load_opponent_meta_belief_memory(
    base_dir: str | Path,
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
) -> OpponentMetaBeliefMemory:
    log_dir = Path(base_dir) / "data" / "persistence" / "cognitive_log"
    payloads = []
    if log_dir.is_dir():
        def chronology(path: Path) -> tuple[int, str]:
            try:
                modified = path.stat().st_mtime_ns
            except OSError:
                modified = 0
            return modified, path.name

        paths = sorted(
            log_dir.glob("*.json"),
            key=chronology,
        )
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return compile_opponent_meta_belief_memory(
        payloads,
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
    )
