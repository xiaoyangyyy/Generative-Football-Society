"""Numeric tactical change detection and non-controlling LLM explanations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    TACTICAL_FEATURES,
    normalise_distribution,
    tactic_feature_vector,
)


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = normalise_distribution(left)
    right = normalise_distribution(right)
    middle = 0.5 * (left + right)

    def divergence(values: np.ndarray) -> float:
        mask = values > 1e-12
        return float(np.sum(values[mask] * np.log(
            values[mask] / np.clip(middle[mask], 1e-12, None)
        )))

    return float(np.clip(
        0.5 * (divergence(left) + divergence(right)) / math.log(2.0),
        0.0,
        1.0,
    ))


def change_point_evidence(
    previous: dict[str, Any],
    *,
    predictive_prior: np.ndarray,
    provisional_posterior: np.ndarray,
    observed: np.ndarray,
    likelihood: np.ndarray,
    now: float,
) -> dict[str, Any]:
    candidate_index = int(np.argmax(likelihood))
    candidate = OPPONENT_HYPOTHESES[candidate_index]
    if not previous:
        zeros = {feature: 0.0 for feature in TACTICAL_FEATURES}
        return {
            "version": 1,
            "status": "initializing",
            "score": 0.0,
            "instantaneous_score": 0.0,
            "feature_shift": 0.0,
            "posterior_divergence": 0.0,
            "predictive_surprise": 0.0,
            "feature_deltas": dict(zeros),
            "instantaneous_feature_deltas": dict(zeros),
            "candidate_hypothesis": candidate,
            "candidate_streak": 0,
            "previous_map_hypothesis": "none",
            "regime_hypothesis": candidate,
            "confirmed": False,
            "regime_id": 0,
            "run_length_updates": 1,
            "confirmed_changes": 0,
            "last_confirmed_s": None,
        }
    previous_observed = np.asarray(
        previous.get("observed_feature_vector") or observed,
        dtype=float,
    )
    feature_deltas = observed - previous_observed
    scaled_rms = float(np.sqrt(np.mean(np.square(feature_deltas / 0.18))))
    feature_shift = float(np.clip(
        1.0 - math.exp(-0.5 * scaled_rms * scaled_rms), 0.0, 1.0,
    ))
    divergence = _jensen_shannon(predictive_prior, provisional_posterior)
    relative_likelihood = likelihood / max(1e-12, float(np.max(likelihood)))
    predictive_support = float(np.sum(predictive_prior * relative_likelihood))
    surprise = float(np.clip(1.0 - predictive_support, 0.0, 1.0))
    instantaneous_score = float(np.clip(
        0.45 * feature_shift + 0.35 * divergence + 0.20 * surprise,
        0.0,
        1.0,
    ))
    prior_change = previous.get("change_point") or {}
    previous_map = str(previous.get("map_hypothesis", "none"))
    regime_hypothesis = str(prior_change.get(
        "regime_hypothesis", previous_map,
    ))
    candidate_changed = candidate != regime_hypothesis
    candidate_streak = (
        int(prior_change.get("candidate_streak", 0)) + 1
        if candidate_changed
        and candidate == str(prior_change.get("candidate_hypothesis", ""))
        else 1 if candidate_changed else 0
    )
    score = instantaneous_score
    if (
        candidate_changed
        and candidate == str(prior_change.get("candidate_hypothesis", ""))
    ):
        score = max(score, 0.95 * float(prior_change.get("score", 0.0)))
    prior_deltas = prior_change.get("feature_deltas") or {}
    evidence_deltas = np.asarray([
        (
            float(prior_deltas.get(feature, 0.0))
            if abs(float(prior_deltas.get(feature, 0.0)))
            > abs(float(feature_deltas[index]))
            and candidate == str(prior_change.get("candidate_hypothesis", ""))
            else float(feature_deltas[index])
        )
        for index, feature in enumerate(TACTICAL_FEATURES)
    ], dtype=float)
    confirmed = bool(
        candidate_changed
        and (score >= 0.85 or (score >= 0.55 and candidate_streak >= 2))
    )
    status = (
        "confirmed" if confirmed
        else "watch" if candidate_changed and score >= 0.40
        else "stable"
    )
    previous_regime = int(prior_change.get("regime_id", 0))
    return {
        "version": 1,
        "status": status,
        "score": score,
        "instantaneous_score": instantaneous_score,
        "feature_shift": feature_shift,
        "posterior_divergence": divergence,
        "predictive_surprise": surprise,
        "feature_deltas": {
            feature: float(evidence_deltas[index])
            for index, feature in enumerate(TACTICAL_FEATURES)
        },
        "instantaneous_feature_deltas": {
            feature: float(feature_deltas[index])
            for index, feature in enumerate(TACTICAL_FEATURES)
        },
        "candidate_hypothesis": candidate,
        "candidate_streak": candidate_streak,
        "previous_map_hypothesis": regime_hypothesis,
        "regime_hypothesis": candidate if confirmed else regime_hypothesis,
        "confirmed": confirmed,
        "regime_id": previous_regime + (1 if confirmed else 0),
        "run_length_updates": (
            0 if confirmed
            else int(prior_change.get("run_length_updates", 1)) + 1
        ),
        "confirmed_changes": int(prior_change.get(
            "confirmed_changes", 0,
        )) + (1 if confirmed else 0),
        "last_confirmed_s": (
            now if confirmed else prior_change.get("last_confirmed_s")
        ),
    }


def validate_change_claim(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    def preset(field: str) -> str | None:
        value = str(raw.get(field, "")).strip().lower().replace(
            "-", "_"
        ).replace(" ", "_")
        return value if value in OPPONENT_HYPOTHESES else None

    source = preset("from_preset")
    target = preset("to_preset")
    if source is None or target is None or source == target:
        return None
    try:
        confidence = float(np.clip(float(raw.get("confidence", 0.0)), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = raw.get("evidence_features") or []
    if not isinstance(evidence, (list, tuple)):
        evidence = []
    grounded = list(dict.fromkeys(
        str(feature) for feature in evidence
        if str(feature) in TACTICAL_FEATURES
    ))[:4]
    return {
        "from_preset": source,
        "to_preset": target,
        "confidence": confidence,
        "evidence_features": grounded,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def audit_change_claim(
    belief: dict[str, Any], raw_claim: Any, *, version: int,
) -> dict[str, Any]:
    claim = validate_change_claim(raw_claim)
    if claim is None:
        return {
            "version": version,
            "accepted": False,
            "reason": "missing_or_invalid_change_claim",
            "can_trigger_change_point": False,
        }
    change = belief.get("change_point") or {}
    deltas = change.get("feature_deltas") or {}
    changed_features = {
        feature for feature in TACTICAL_FEATURES
        if abs(float(deltas.get(feature, 0.0))) >= 0.05
    }
    cited = set(claim["evidence_features"])
    grounded = cited & changed_features
    source_vector = tactic_feature_vector(claim["from_preset"])
    target_vector = tactic_feature_vector(claim["to_preset"])
    directional = 0
    for feature in grounded:
        index = TACTICAL_FEATURES.index(feature)
        expected_delta = target_vector[index] - source_vector[index]
        observed_delta = float(deltas.get(feature, 0.0))
        if abs(expected_delta) >= 0.03 and expected_delta * observed_delta > 0.0:
            directional += 1
    grounding_fraction = len(grounded) / max(1, len(cited))
    directional_fraction = directional / max(1, len(grounded))
    detector_alignment = bool(
        claim["from_preset"] == str(change.get("previous_map_hypothesis", ""))
        and claim["to_preset"] == str(change.get("candidate_hypothesis", ""))
    )
    status_factor = (
        1.0 if change.get("status") == "confirmed"
        else 0.60 if change.get("status") == "watch" else 0.0
    )
    ranked = {item["tactical_preset"]: item for item in belief["hypotheses"]}
    target_likelihood = float(ranked[claim["to_preset"]]["observation_likelihood"])
    maximum_likelihood = max(
        float(item["observation_likelihood"]) for item in belief["hypotheses"]
    )
    likelihood_support = target_likelihood / max(1e-12, maximum_likelihood)
    evidence_score = float(np.clip(
        claim["confidence"] * grounding_fraction * directional_fraction
        * likelihood_support * status_factor
        * (1.0 if detector_alignment else 0.0),
        0.0,
        1.0,
    ))
    accepted = evidence_score >= 0.15
    return {
        "version": version,
        "accepted": accepted,
        "reason": (
            "numerically_supported_change_explanation"
            if accepted else "change_claim_not_supported"
        ),
        "claim": claim,
        "detector_status": str(change.get("status", "unavailable")),
        "detector_alignment": detector_alignment,
        "changed_features": sorted(changed_features),
        "grounded_features": sorted(grounded),
        "grounding_fraction": grounding_fraction,
        "directional_fraction": directional_fraction,
        "observation_likelihood_support": likelihood_support,
        "evidence_score": evidence_score,
        "can_trigger_change_point": False,
        "causal_interpretation": False,
    }
