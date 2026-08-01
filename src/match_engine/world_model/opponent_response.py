"""Validated action-conditioned opponent response dynamics in belief space."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    normalise_distribution,
    normalized_entropy,
)


OPPONENT_RESPONSE_VERSION = 1
RESPONSE_ACTIONS = ("hold", "pass", "cross", "shot")
_STICKINESS = {
    "hold": 0.88,
    "pass": 0.82,
    "cross": 0.78,
    "shot": 0.74,
}


def _posterior(raw: Any) -> np.ndarray | None:
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
    return normalise_distribution(values)


def structural_response(
    current: np.ndarray, action: str,
) -> np.ndarray:
    """Transparent fallback: persistence plus action-scaled regime hazard."""
    current = normalise_distribution(current)
    uniform = np.full(current.shape, 1.0 / current.size)
    stickiness = _STICKINESS.get(str(action), 0.80)
    return normalise_distribution(
        stickiness * current + (1.0 - stickiness) * uniform
    )


@dataclass(frozen=True)
class OpponentResponseProfile:
    action: str
    transition_matches: int
    transition_pairs: int
    training_matches: int
    validation_matches: int
    validation_pairs: int
    validation_brier: float
    baseline_brier: float
    validation_skill: float
    active: bool
    trust: float
    transition_matrix: tuple[tuple[float, ...], ...]

    def to_dict(self, *, include_matrix: bool = True) -> dict[str, Any]:
        payload = {
            "action": self.action,
            "transition_matches": self.transition_matches,
            "transition_pairs": self.transition_pairs,
            "training_matches": self.training_matches,
            "validation_matches": self.validation_matches,
            "validation_pairs": self.validation_pairs,
            "validation_brier": self.validation_brier,
            "baseline_brier": self.baseline_brier,
            "validation_skill": self.validation_skill,
            "active": self.active,
            "trust": self.trust,
        }
        if include_matrix:
            payload["transition_matrix"] = [
                list(row) for row in self.transition_matrix
            ]
        return payload


@dataclass
class OpponentResponseMemory:
    checkpoint_signature: str
    environment_signature: str
    profiles: dict[str, OpponentResponseProfile]
    source_logs: int
    compatible_matches: int

    def predict(
        self,
        current_posterior: dict[str, float],
        action: str,
    ) -> dict[str, Any]:
        action = str(action).lower()
        current = _posterior(current_posterior)
        if current is None:
            current = np.full(
                len(OPPONENT_HYPOTHESES),
                1.0 / len(OPPONENT_HYPOTHESES),
            )
        fallback = structural_response(current, action)
        profile = self.profiles.get(action)
        learned = fallback
        trust = 0.0
        if profile is not None and profile.active:
            matrix = np.asarray(profile.transition_matrix, dtype=float)
            learned = normalise_distribution(current @ matrix)
            trust = profile.trust
        response = normalise_distribution(
            (1.0 - trust) * fallback + trust * learned
        )
        source = (
            "validated_action_conditioned_response_memory"
            if trust > 0.0 else "structural_sticky_response_prior"
        )
        return {
            "version": OPPONENT_RESPONSE_VERSION,
            "action": action,
            "source": source,
            "learned_active": trust > 0.0,
            "trust": trust,
            "current_posterior": {
                name: float(current[index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
            "response_posterior": {
                name: float(response[index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
            "structural_posterior": {
                name: float(fallback[index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
            "learned_posterior": {
                name: float(learned[index])
                for index, name in enumerate(OPPONENT_HYPOTHESES)
            },
            "normalized_entropy": normalized_entropy(response),
            "profile": (
                profile.to_dict(include_matrix=False)
                if profile is not None else {
                    "action": action,
                    "active": False,
                    "reason": "no_compatible_response_history",
                }
            ),
            "causal_interpretation": False,
            "policy": (
                "Learned response dynamics are observational and influence "
                "planning only after match-held-out gain over the structural prior."
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "version": OPPONENT_RESPONSE_VERSION,
            "checkpoint_signature": self.checkpoint_signature,
            "environment_signature": self.environment_signature,
            "source_logs": self.source_logs,
            "compatible_matches": self.compatible_matches,
            "active_actions": sum(profile.active for profile in self.profiles.values()),
            "profiles": {
                action: profile.to_dict(include_matrix=False)
                for action, profile in sorted(self.profiles.items())
            },
        }


def _match_pairs(
    payload: dict[str, Any],
    *,
    checkpoint_signature: str,
    environment_signature: str,
    maximum_gap_s: float,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    adoption = payload.get("world_model_decision_adoption") or {}
    compatible = []
    for record in adoption.get("records") or []:
        if str(record.get("checkpoint_signature")) != checkpoint_signature:
            continue
        if str(record.get(
            "environment_signature", "environment_unspecified",
        )) != environment_signature:
            continue
        context = record.get("opponent_belief_context") or {}
        current = _posterior(context.get("posterior"))
        opponent = str(context.get("opponent_team_id", "")).strip()
        if current is None or not opponent:
            continue
        compatible.append((
            str(record.get("team_id", "")),
            opponent,
            float(record.get("created_t_sec", 0.0)),
            str(record.get("intervention_actual_action") or "").lower(),
            current,
        ))
    compatible.sort(key=lambda item: (item[0], item[1], item[2]))
    pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for left, right in zip(compatible, compatible[1:]):
        team, opponent, created, action, current = left
        next_team, next_opponent, next_created, _next_action, next_belief = right
        gap = next_created - created
        if (
            team != next_team
            or opponent != next_opponent
            or action not in RESPONSE_ACTIONS
            or gap <= 0.0
            or gap > maximum_gap_s
        ):
            continue
        pairs.setdefault(action, []).append((current, next_belief))
    return pairs


def _fit_matrix(
    match_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    action: str,
) -> np.ndarray:
    size = len(OPPONENT_HYPOTHESES)
    counts = 1.5 * (
        _STICKINESS[action] * np.eye(size)
        + (1.0 - _STICKINESS[action]) / size
    )
    for pairs in match_pairs:
        if not pairs:
            continue
        contribution = sum(
            np.outer(current, following) for current, following in pairs
        ) / len(pairs)
        counts += contribution
    return np.stack([
        normalise_distribution(row) for row in counts
    ])


def _validation_error(
    matrix: np.ndarray,
    validation: list[list[tuple[np.ndarray, np.ndarray]]],
    action: str,
) -> tuple[float, float, int]:
    model_match_errors = []
    baseline_match_errors = []
    pairs_total = 0
    for pairs in validation:
        if not pairs:
            continue
        pairs_total += len(pairs)
        model_errors = []
        baseline_errors = []
        for current, actual in pairs:
            model = normalise_distribution(current @ matrix)
            baseline = structural_response(current, action)
            model_errors.append(float(np.mean(np.square(model - actual))))
            baseline_errors.append(float(np.mean(np.square(baseline - actual))))
        model_match_errors.append(float(np.mean(model_errors)))
        baseline_match_errors.append(float(np.mean(baseline_errors)))
    return (
        float(np.mean(model_match_errors)) if model_match_errors else 0.0,
        float(np.mean(baseline_match_errors)) if baseline_match_errors else 0.0,
        pairs_total,
    )


def compile_opponent_response_memory(
    logs: Iterable[dict[str, Any]],
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
    maximum_gap_s: float = 900.0,
) -> OpponentResponseMemory:
    payloads = list(logs)
    rows_by_action: dict[str, list[list[tuple[np.ndarray, np.ndarray]]]] = {
        action: [] for action in RESPONSE_ACTIONS
    }
    compatible_matches = 0
    for payload in payloads:
        match = _match_pairs(
            payload,
            checkpoint_signature=str(checkpoint_signature),
            environment_signature=str(environment_signature),
            maximum_gap_s=max(30.0, float(maximum_gap_s)),
        )
        if match:
            compatible_matches += 1
        for action, pairs in match.items():
            if pairs:
                rows_by_action[action].append(pairs)

    profiles = {}
    for action, match_rows in rows_by_action.items():
        match_count = len(match_rows)
        split = max(1, match_count // 2)
        training = match_rows[:split]
        validation = match_rows[split:]
        training_matrix = _fit_matrix(training, action)
        model_brier, baseline_brier, validation_pairs = _validation_error(
            training_matrix, validation, action,
        )
        validation_skill = (
            1.0 - model_brier / max(1e-12, baseline_brier)
            if validation_pairs else 0.0
        )
        active = bool(
            len(training) >= 2
            and len(validation) >= 2
            and validation_pairs >= 2
            and baseline_brier > 1e-10
            and validation_skill >= 0.02
        )
        refit = _fit_matrix(match_rows, action) if active else training_matrix
        sample_factor = match_count / (match_count + 6.0)
        skill_factor = float(np.clip(validation_skill, 0.0, 0.50)) / 0.50
        trust = float(np.clip(
            0.50 * sample_factor * skill_factor if active else 0.0,
            0.0,
            0.50,
        ))
        profiles[action] = OpponentResponseProfile(
            action=action,
            transition_matches=match_count,
            transition_pairs=sum(len(pairs) for pairs in match_rows),
            training_matches=len(training) if match_count else 0,
            validation_matches=len(validation),
            validation_pairs=validation_pairs,
            validation_brier=model_brier,
            baseline_brier=baseline_brier,
            validation_skill=validation_skill,
            active=active,
            trust=trust,
            transition_matrix=tuple(tuple(float(value) for value in row) for row in refit),
        )
    return OpponentResponseMemory(
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
        profiles=profiles,
        source_logs=len(payloads),
        compatible_matches=compatible_matches,
    )


def load_opponent_response_memory(
    base_dir: str | Path,
    *,
    checkpoint_signature: str,
    environment_signature: str = "environment_unspecified",
) -> OpponentResponseMemory:
    log_dir = Path(base_dir) / "data" / "persistence" / "cognitive_log"
    payloads = []
    if log_dir.is_dir():
        def chronology(path: Path) -> tuple[int, str]:
            try:
                modified = path.stat().st_mtime_ns
            except OSError:
                modified = 0
            return modified, path.name

        for path in sorted(log_dir.glob("*.json"), key=chronology):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return compile_opponent_response_memory(
        payloads,
        checkpoint_signature=str(checkpoint_signature),
        environment_signature=str(environment_signature),
    )


def opponent_response_diagnostics(
    logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Score registered response forecasts against the next observed belief."""
    match_model_errors: list[float] = []
    match_structural_errors: list[float] = []
    predictions = learned_predictions = 0
    composition_errors: list[float] = []
    submitted_audits: list[dict[str, Any]] = []
    for payload in logs:
        adoption = payload.get("world_model_decision_adoption") or {}
        model_errors = []
        structural_errors = []
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for record in adoption.get("records") or []:
            belief = record.get("opponent_belief_context") or {}
            key = (
                str(record.get("team_id", "")),
                str(belief.get("opponent_team_id", "")),
                str(record.get("checkpoint_signature", "runtime_unspecified")),
                str(record.get(
                    "environment_signature", "environment_unspecified",
                )),
            )
            groups.setdefault(key, []).append(record)
        pairs = []
        for records in groups.values():
            records.sort(key=lambda record: float(record.get("created_t_sec", 0.0)))
            pairs.extend(zip(records, records[1:]))
        for current, following in pairs:
            current_belief = current.get("opponent_belief_context") or {}
            following_belief = following.get("opponent_belief_context") or {}
            response_context = current.get("opponent_response_context") or {}
            prediction = response_context.get("prediction") or {}
            action = str(current.get("intervention_actual_action") or "")
            gap = float(following.get("created_t_sec", 0.0)) - float(
                current.get("created_t_sec", 0.0)
            )
            if (
                action not in RESPONSE_ACTIONS
                or str(prediction.get("action")) != action
                or gap <= 0.0
                or gap > 900.0
            ):
                continue
            predicted = _posterior(prediction.get("response_posterior"))
            structural = _posterior(prediction.get("structural_posterior"))
            actual = _posterior(following_belief.get("posterior"))
            if predicted is None or actual is None:
                continue
            if structural is None:
                current_posterior = _posterior(current_belief.get("posterior"))
                if current_posterior is None:
                    continue
                structural = structural_response(current_posterior, action)
            model_errors.append(float(np.mean(np.square(predicted - actual))))
            structural_errors.append(float(np.mean(np.square(structural - actual))))
            composition_errors.append(abs(float(predicted.sum()) - 1.0))
            predictions += 1
            learned_predictions += bool(
                prediction.get("learned_active")
                and prediction.get("source")
                == "validated_action_conditioned_response_memory"
            )
        if model_errors:
            match_model_errors.append(float(np.mean(model_errors)))
            match_structural_errors.append(float(np.mean(structural_errors)))
        for cognitive in payload.get("cognitive_plans") or []:
            plan = cognitive.get("plan") or {}
            audit = plan.get("opponent_response_hypothesis_audit")
            if isinstance(audit, dict) and isinstance(audit.get("hypothesis"), dict):
                submitted_audits.append(audit)
    model_brier = (
        float(np.mean(match_model_errors)) if match_model_errors else 0.0
    )
    structural_brier = (
        float(np.mean(match_structural_errors))
        if match_structural_errors else 0.0
    )
    return {
        "version": OPPONENT_RESPONSE_VERSION,
        "available": predictions > 0,
        "realized_predictions": predictions,
        "validated_learned_predictions": learned_predictions,
        "structural_predictions": predictions - learned_predictions,
        "scored_matches": len(match_model_errors),
        "match_clustered_brier": model_brier,
        "structural_baseline_brier": structural_brier,
        "skill_vs_structural": (
            1.0 - model_brier / max(1e-12, structural_brier)
            if predictions and structural_brier > 1e-12 else 0.0
        ),
        "mean_composition_identity_error": (
            float(np.mean(composition_errors)) if composition_errors else 0.0
        ),
        "llm_hypotheses_submitted": len(submitted_audits),
        "llm_hypotheses_accepted": sum(
            bool(audit.get("accepted")) for audit in submitted_audits
        ),
        "all_llm_hypotheses_non_persistent": all(
            audit.get("can_update_response_memory") is False
            for audit in submitted_audits
        ),
        "causal_interpretation": False,
    }
