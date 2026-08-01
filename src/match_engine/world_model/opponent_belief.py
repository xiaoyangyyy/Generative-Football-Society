"""Bayesian opponent-tactic beliefs and model-grounded LLM hypotheses.

The opponent's tactical intent is latent even when some control values are
observable.  This module keeps a slowly changing posterior over interpretable
tactical archetypes.  Numeric observations provide the likelihood; an LLM may
only contribute bounded pseudo-evidence tied to named observable features.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.tactical_catalog import TACTICAL_PRESETS, resolve_tactical_preset


OPPONENT_BELIEF_VERSION = 1
TACTICAL_FEATURES = (
    "pressing_intensity",
    "risk_budget",
    "line_height",
    "rotation_aggressiveness",
)
OPPONENT_HYPOTHESES = (
    "balanced",
    "gegenpress",
    "possession_control",
    "counter_attack",
    "low_block",
    "wing_play",
    "direct_vertical",
)


def _normalise(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    clean = np.clip(clean, 0.0, None)
    total = float(clean.sum())
    if total <= 1e-12:
        return np.full(clean.shape, 1.0 / max(1, clean.size), dtype=float)
    return clean / total


def _entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-12, 1.0)
    denominator = math.log(max(2, values.size))
    return float(-np.sum(values * np.log(values)) / denominator)


def tactic_feature_vector(preset_name: str) -> np.ndarray:
    preset = TACTICAL_PRESETS[resolve_tactical_preset(preset_name)]
    return np.asarray([
        float(np.clip(preset.get(feature, 0.5), 0.0, 1.0))
        for feature in TACTICAL_FEATURES
    ], dtype=float)


def _team_pair(state, observer_team_id: str):
    if str(observer_team_id) == str(state.home.team_id):
        return state.home, state.away
    if str(observer_team_id) == str(state.away.team_id):
        return state.away, state.home
    raise ValueError(f"unknown observer team: {observer_team_id}")


def _observed_tactics(team) -> np.ndarray:
    controls = team.coach.tactical_current or team.coach.tactical_base or {}
    return np.asarray([
        float(np.clip(controls.get(feature, 0.5), 0.0, 1.0))
        for feature in TACTICAL_FEATURES
    ], dtype=float)


def _likelihoods(observed: np.ndarray, *, sigma: float = 0.18) -> np.ndarray:
    centres = np.stack([
        tactic_feature_vector(name) for name in OPPONENT_HYPOTHESES
    ])
    squared = np.mean(np.square((centres - observed) / max(0.05, sigma)), axis=1)
    # A tempered likelihood avoids collapsing the belief after one trigger.
    log_likelihood = -0.5 * squared
    log_likelihood -= float(np.max(log_likelihood))
    return np.exp(log_likelihood)


def _summary(
    *,
    observer_team_id: str,
    opponent_team_id: str,
    posterior: np.ndarray,
    likelihood: np.ndarray,
    observed: np.ndarray,
    evidence_count: int,
    last_update_s: float,
    switch_probability: float,
    source: str,
) -> dict[str, Any]:
    order = np.argsort(-posterior)
    hypotheses = [
        {
            "tactical_preset": OPPONENT_HYPOTHESES[index],
            "probability": float(posterior[index]),
            "observation_likelihood": float(likelihood[index]),
            "feature_vector": tactic_feature_vector(
                OPPONENT_HYPOTHESES[index]
            ).tolist(),
        }
        for index in order
    ]
    return {
        "version": OPPONENT_BELIEF_VERSION,
        "observer_team_id": str(observer_team_id),
        "opponent_team_id": str(opponent_team_id),
        "evidence_count": int(evidence_count),
        "last_update_s": float(last_update_s),
        "source": source,
        "features": list(TACTICAL_FEATURES),
        "observed_feature_vector": observed.tolist(),
        "posterior": {
            name: float(posterior[index])
            for index, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "hypotheses": hypotheses,
        "map_hypothesis": hypotheses[0]["tactical_preset"],
        "map_probability": hypotheses[0]["probability"],
        "normalized_entropy": _entropy(posterior),
        "switch_probability": float(np.clip(switch_probability, 0.0, 1.0)),
        "policy": (
            "Treat as a latent-state posterior, not a discovered fact. LLM "
            "hypotheses are bounded by observable-feature likelihood."
        ),
    }


def update_opponent_belief(state, observer_team_id: str) -> dict[str, Any]:
    """Advance and condition a match-local posterior on observed controls."""
    _observer, opponent = _team_pair(state, observer_team_id)
    key = f"{observer_team_id}->{opponent.team_id}"
    store = getattr(state, "_wm_opponent_beliefs", None)
    if store is None:
        store = {}
        state._wm_opponent_beliefs = store
    previous = store.get(key) or {}
    previous_posterior = _normalise(np.asarray([
        (previous.get("posterior") or {}).get(name, 1.0)
        for name in OPPONENT_HYPOTHESES
    ], dtype=float))
    now = float(getattr(state, "clock_seconds", 0.0))
    if previous and abs(float(previous.get("last_update_s", -1.0)) - now) < 1e-9:
        return previous

    # A sticky Markov prior represents tactical continuity while retaining
    # probability mass for a formation or intent switch.
    uniform = np.full(len(OPPONENT_HYPOTHESES), 1.0 / len(OPPONENT_HYPOTHESES))
    elapsed = max(0.0, now - float(previous.get("last_update_s", now)))
    transition_mass = float(np.clip(0.06 + elapsed / 1800.0, 0.06, 0.22))
    predictive_prior = (1.0 - transition_mass) * previous_posterior + transition_mass * uniform
    observed = _observed_tactics(opponent)
    likelihood = _likelihoods(observed)
    posterior = _normalise(predictive_prior * np.power(likelihood, 0.65))
    switch_probability = 0.5 * float(np.abs(posterior - previous_posterior).sum())
    summary = _summary(
        observer_team_id=str(observer_team_id),
        opponent_team_id=str(opponent.team_id),
        posterior=posterior,
        likelihood=likelihood,
        observed=observed,
        evidence_count=int(previous.get("evidence_count", 0)) + 1,
        last_update_s=now,
        switch_probability=switch_probability,
        source="markov_prior_plus_observed_tactical_likelihood",
    )
    summary["previous_map_hypothesis"] = previous.get("map_hypothesis", "none")
    summary["llm_audits"] = list(previous.get("llm_audits") or [])[-7:]
    summary["llm_influence_used"] = 0.0
    store[key] = summary
    return summary


def validate_llm_opponent_hypothesis(raw: Any) -> dict[str, Any] | None:
    """Return a constrained hypothesis or None; never infer omitted evidence."""
    if not isinstance(raw, dict) or not raw.get("tactical_preset"):
        return None
    requested = str(raw.get("tactical_preset", "")).strip().lower().replace(
        "-", "_"
    ).replace(" ", "_")
    if requested not in OPPONENT_HYPOTHESES:
        return None
    resolved = resolve_tactical_preset(requested)
    if resolved not in OPPONENT_HYPOTHESES:
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
        "tactical_preset": resolved,
        "confidence": confidence,
        "evidence_features": grounded,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def assimilate_llm_opponent_hypothesis(
    state,
    observer_team_id: str,
    raw_hypothesis: Any,
) -> dict[str, Any]:
    """Fuse bounded LLM pseudo-evidence into the current numeric posterior."""
    belief = update_opponent_belief(state, observer_team_id)
    hypothesis = validate_llm_opponent_hypothesis(raw_hypothesis)
    if hypothesis is None:
        return {
            "version": OPPONENT_BELIEF_VERSION,
            "accepted": False,
            "reason": "missing_or_invalid_hypothesis",
            "posterior": dict(belief["posterior"]),
        }
    index = OPPONENT_HYPOTHESES.index(hypothesis["tactical_preset"])
    ranked = {item["tactical_preset"]: item for item in belief["hypotheses"]}
    likelihoods = np.asarray([
        ranked[name]["observation_likelihood"] for name in OPPONENT_HYPOTHESES
    ], dtype=float)
    likelihood_support = float(likelihoods[index] / max(1e-12, likelihoods.max()))
    grounding = len(hypothesis["evidence_features"]) / len(TACTICAL_FEATURES)
    proposed_influence = float(np.clip(
        0.20 * hypothesis["confidence"] * grounding * likelihood_support,
        0.0,
        0.20,
    ))
    influence_used = float(np.clip(
        belief.get("llm_influence_used", 0.0), 0.0, 0.20,
    ))
    influence = min(proposed_influence, max(0.0, 0.20 - influence_used))
    prior = _normalise(np.asarray([
        belief["posterior"][name] for name in OPPONENT_HYPOTHESES
    ], dtype=float))
    evidence_distribution = np.zeros_like(prior)
    evidence_distribution[index] = 1.0
    posterior = _normalise((1.0 - influence) * prior + influence * evidence_distribution)
    accepted = influence >= 0.01
    audit = {
        "version": OPPONENT_BELIEF_VERSION,
        "accepted": accepted,
        "reason": "model_grounded_update" if accepted else "insufficient_grounding",
        "hypothesis": hypothesis,
        "observation_likelihood_support": likelihood_support,
        "grounding_fraction": grounding,
        "bounded_influence": influence,
        "proposed_influence": proposed_influence,
        "observation_influence_used_before": influence_used,
        "observation_influence_used_after": influence_used + influence,
        "prior_probability": float(prior[index]),
        "posterior_probability": float(posterior[index]),
        "prior_entropy": _entropy(prior),
        "posterior_entropy": _entropy(posterior),
        "posterior": {
            name: float(posterior[position])
            for position, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "causal_interpretation": False,
    }
    if accepted:
        likelihood = np.asarray([
            ranked[name]["observation_likelihood"] for name in OPPONENT_HYPOTHESES
        ])
        observed = np.asarray(belief["observed_feature_vector"], dtype=float)
        updated = _summary(
            observer_team_id=belief["observer_team_id"],
            opponent_team_id=belief["opponent_team_id"],
            posterior=posterior,
            likelihood=likelihood,
            observed=observed,
            evidence_count=belief["evidence_count"],
            last_update_s=belief["last_update_s"],
            switch_probability=belief["switch_probability"],
            source="numeric_posterior_plus_bounded_llm_pseudo_evidence",
        )
        updated["previous_map_hypothesis"] = belief.get(
            "previous_map_hypothesis", "none"
        )
        updated["llm_audits"] = [
            *list(belief.get("llm_audits") or [])[-6:], audit,
        ]
        updated["llm_influence_used"] = influence_used + influence
        key = f"{observer_team_id}->{belief['opponent_team_id']}"
        state._wm_opponent_beliefs[key] = updated
    return audit


def reweight_counterfactual_candidates(
    packet: dict[str, Any], posterior: dict[str, float]
) -> dict[str, Any]:
    """Recompute robust action values after a grounded hypothesis update."""
    candidates = packet.get("candidates") or []
    changed = False
    for candidate in candidates:
        values = candidate.get("opponent_hypothesis_values") or {}
        pairs = [
            (float(posterior.get(name, 0.0)), float(values[name]))
            for name in OPPONENT_HYPOTHESES if name in values
        ]
        if not pairs:
            continue
        weights = _normalise(np.asarray([item[0] for item in pairs]))
        outcomes = np.asarray([item[1] for item in pairs], dtype=float)
        mean = float(np.sum(weights * outcomes))
        std = float(np.sqrt(np.sum(weights * np.square(outcomes - mean))))
        order = np.argsort(outcomes)
        cumulative = np.cumsum(weights[order])
        tail_index = min(
            len(order) - 1, int(np.searchsorted(cumulative, 0.10, side="left")),
        )
        tail = float(outcomes[order[tail_index]])
        robust = mean - 0.30 * std - 0.10 * max(0.0, mean - tail)
        candidate["opponent_belief_expected_value"] = mean
        candidate["opponent_belief_value_std"] = std
        candidate["opponent_belief_tail_value"] = tail
        candidate["risk_adjusted_value"] = robust
        changed = True
    eligible = [
        item for item in candidates if float(item.get("effective_confidence", 0.0)) > 0.0
    ]
    best = max(eligible, key=lambda item: item["risk_adjusted_value"], default=None)
    if changed:
        packet["recommended_action"] = best["action"] if best else "none"
        packet["opponent_belief_reweighted"] = True
    return packet


def opponent_belief_diagnostics(logs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive diagnostics; hidden intent has no direct truth label."""
    audits = []
    sensitivities = []
    belief_snapshots = []
    for payload in logs:
        for record in payload.get("cognitive_plans") or []:
            plan = record.get("plan") or {}
            audit = plan.get("opponent_belief_audit")
            if (
                isinstance(audit, dict)
                and isinstance(audit.get("hypothesis"), dict)
            ):
                audits.append(audit)
            packet = ((record.get("trigger") or {}).get("facts") or {}).get(
                "world_model_decision_support"
            ) or {}
            for candidate in packet.get("candidates") or []:
                try:
                    sensitivities.append(float(candidate.get(
                        "opponent_belief_value_std", 0.0
                    )))
                except (TypeError, ValueError):
                    continue
        adoption = payload.get("world_model_decision_adoption") or {}
        for record in adoption.get("records") or []:
            context = record.get("opponent_belief_context") or {}
            if context.get("posterior"):
                belief_snapshots.append(context)
    accepted = [audit for audit in audits if audit.get("accepted")]
    regimes: dict[str, int] = {}
    for context in belief_snapshots:
        regime = str(context.get("map_hypothesis", "unknown"))
        regimes[regime] = regimes.get(regime, 0) + 1
    return {
        "version": OPPONENT_BELIEF_VERSION,
        "hypotheses_submitted": len(audits),
        "hypotheses_accepted": len(accepted),
        "acceptance_rate": len(accepted) / max(1, len(audits)),
        "mean_bounded_llm_influence": float(np.mean([
            float(audit.get("bounded_influence", 0.0)) for audit in audits
        ])) if audits else 0.0,
        "mean_counterfactual_action_sensitivity": (
            float(np.mean(sensitivities)) if sensitivities else 0.0
        ),
        "decision_belief_snapshots": len(belief_snapshots),
        "decisions_by_map_hypothesis": regimes,
        "mean_decision_belief_entropy": float(np.mean([
            float(context.get("normalized_entropy", 1.0))
            for context in belief_snapshots
        ])) if belief_snapshots else 0.0,
        "available": bool(audits or sensitivities or belief_snapshots),
        "calibrated_against_hidden_truth": False,
        "limitation": (
            "Opponent intent is latent; these diagnostics audit grounding and "
            "decision sensitivity, not classification accuracy."
        ),
    }
