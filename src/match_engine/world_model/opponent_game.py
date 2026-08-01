"""Two-ply belief-space game planning and bounded LLM branch hypotheses."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    normalise_distribution,
    normalized_entropy,
)
from src.match_engine.world_model.opponent_response import (
    OPPONENT_RESPONSE_VERSION,
    RESPONSE_ACTIONS,
    structural_response,
)


OPPONENT_GAME_VERSION = 1


def _posterior(raw: dict[str, float]) -> np.ndarray:
    return normalise_distribution(np.asarray([
        float(raw.get(name, 0.0)) for name in OPPONENT_HYPOTHESES
    ], dtype=float))


def robust_hypothesis_value(
    values: dict[str, float], posterior: dict[str, float],
) -> dict[str, float]:
    pairs = [
        (float(posterior.get(name, 0.0)), float(values[name]))
        for name in OPPONENT_HYPOTHESES if name in values
    ]
    if not pairs:
        return {"mean": -1.0, "std": 0.0, "tail_10": -1.0, "robust": -1.0}
    weights = normalise_distribution(np.asarray([item[0] for item in pairs]))
    outcomes = np.asarray([item[1] for item in pairs], dtype=float)
    mean = float(np.sum(weights * outcomes))
    std = float(np.sqrt(np.sum(weights * np.square(outcomes - mean))))
    order = np.argsort(outcomes)
    cumulative = np.cumsum(weights[order])
    index = min(
        len(order) - 1,
        int(np.searchsorted(cumulative, 0.10, side="left")),
    )
    tail = float(outcomes[order[index]])
    return {
        "mean": mean,
        "std": std,
        "tail_10": tail,
        "robust": mean - 0.30 * std - 0.10 * max(0.0, mean - tail),
    }


def _response_prediction(
    memory,
    current_posterior: dict[str, float],
    action: str,
) -> dict[str, Any]:
    if memory is not None and callable(getattr(memory, "predict", None)):
        return memory.predict(current_posterior, action)
    current = _posterior(current_posterior)
    response = structural_response(current, action)
    return {
        "version": OPPONENT_RESPONSE_VERSION,
        "action": action,
        "source": "structural_sticky_response_prior",
        "learned_active": False,
        "trust": 0.0,
        "current_posterior": dict(current_posterior),
        "response_posterior": {
            name: float(response[index])
            for index, name in enumerate(OPPONENT_HYPOTHESES)
        },
        "normalized_entropy": normalized_entropy(response),
        "profile": {
            "action": action, "active": False,
            "reason": "response_memory_unavailable",
        },
        "causal_interpretation": False,
    }


def attach_second_order_game(
    candidates: list[dict[str, Any]],
    belief: dict[str, Any] | None,
    response_memory=None,
    *,
    discount: float = 0.55,
    response_overrides: dict[str, dict[str, float]] | None = None,
    continuation_value_matrices: (
        dict[str, dict[str, dict[str, float]]] | None
    ) = None,
    trajectory_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate candidate ranking with a partial-observability-correct second ply."""
    if not belief or not belief.get("posterior"):
        return {"version": OPPONENT_GAME_VERSION, "available": False}
    eligible = [
        candidate for candidate in candidates
        if float(candidate.get("effective_confidence", 0.0)) > 0.0
        and candidate.get("opponent_hypothesis_values")
    ]
    if not eligible:
        return {"version": OPPONENT_GAME_VERSION, "available": False}
    bounded_discount = float(np.clip(discount, 0.0, 0.80))
    overrides = response_overrides or {}
    trajectory_matrices = continuation_value_matrices or {}
    branches = []
    learned_branches = 0
    for candidate in eligible:
        action = str(candidate["action"])
        prediction = _response_prediction(
            response_memory, dict(belief["posterior"]), action,
        )
        if action in overrides:
            prediction = dict(prediction)
            prediction["response_posterior"] = dict(overrides[action])
            prediction["normalized_entropy"] = normalized_entropy(
                _posterior(overrides[action])
            )
            prediction["source"] = (
                str(prediction["source"]) + "+bounded_llm_branch_hypothesis"
            )
        learned_branches += bool(prediction.get("learned_active"))
        continuation_options = []
        for continuation in eligible:
            continuation_action = str(continuation["action"])
            hypothesis_values = (
                trajectory_matrices.get(action, {}).get(continuation_action)
                or continuation["opponent_hypothesis_values"]
            )
            metrics = robust_hypothesis_value(
                hypothesis_values,
                prediction["response_posterior"],
            )
            continuation_options.append({
                "action": continuation_action,
                "evaluation_source": (
                    "validated_predicted_state_rollout_blend"
                    if action in trajectory_matrices
                    else "current_state_payoff_proxy"
                ),
                **metrics,
            })
        best_continuation = max(
            continuation_options,
            key=lambda option: option["robust"],
        )
        immediate = float(candidate.get(
            "single_ply_risk_adjusted_value",
            candidate.get("risk_adjusted_value", -1.0),
        ))
        candidate["single_ply_risk_adjusted_value"] = immediate
        two_ply = (
            immediate + bounded_discount * best_continuation["robust"]
        ) / (1.0 + bounded_discount)
        current = _posterior(belief["posterior"])
        response = _posterior(prediction["response_posterior"])
        response_shift = 0.5 * float(np.abs(response - current).sum())
        information = float(np.clip(
            prediction["normalized_entropy"] * response_shift,
            0.0,
            1.0,
        ))
        candidate["opponent_response_prediction"] = prediction
        candidate["continuation_options"] = continuation_options
        candidate["best_continuation_action"] = best_continuation["action"]
        candidate["best_continuation_value"] = best_continuation["robust"]
        candidate["two_ply_risk_adjusted_value"] = two_ply
        candidate["opponent_response_information"] = information
        candidate["trajectory_continuation_hypothesis_values"] = dict(
            trajectory_matrices.get(action, {})
        )
        candidate["risk_adjusted_value"] = two_ply
        branches.append({
            "first_action": action,
            "response_source": prediction["source"],
            "response_trust": float(prediction.get("trust", 0.0)),
            "response_entropy": float(prediction["normalized_entropy"]),
            "best_continuation_action": best_continuation["action"],
            "single_ply_value": immediate,
            "continuation_value": best_continuation["robust"],
            "two_ply_value": two_ply,
        })
    best = max(eligible, key=lambda candidate: candidate["risk_adjusted_value"])
    return {
        "version": OPPONENT_GAME_VERSION,
        "available": True,
        "scope": (
            "validated_predicted_state_two_ply_belief_policy"
            if trajectory_matrices else "two_ply_belief_space_policy_proxy"
        ),
        "discount": bounded_discount,
        "recommended_action": str(best["action"]),
        "learned_response_branches": learned_branches,
        "branches": branches,
        "response_memory": (
            response_memory.summary()
            if response_memory is not None
            and callable(getattr(response_memory, "summary", None))
            else {
                "version": OPPONENT_RESPONSE_VERSION,
                "active_actions": 0,
                "reason": "response_memory_unavailable",
            }
        ),
        "trajectory_rollout": dict(trajectory_audit or {
            "active": False,
            "reason": "predicted_state_rollout_not_requested",
        }),
        "limitations": [
            (
                "Second-ply values blend validated predicted-state rollouts with the current-state proxy."
                if trajectory_matrices else
                "Second-ply continuation reuses the current-state payoff matrix."
            ),
            "Opponent response transitions are observational, not causal.",
            "The continuation action is chosen against one response belief, not hidden truth.",
        ],
    }


def validate_llm_response_hypothesis(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("if_action", "")).strip().lower()
    response = str(raw.get("response_preset", "")).strip().lower().replace(
        "-", "_"
    ).replace(" ", "_")
    if action not in RESPONSE_ACTIONS or response not in OPPONENT_HYPOTHESES:
        return None
    try:
        confidence = float(np.clip(float(raw.get("confidence", 0.0)), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "if_action": action,
        "response_preset": response,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def apply_llm_response_hypothesis(
    packet: dict[str, Any],
    raw_hypothesis: Any,
    response_memory=None,
) -> dict[str, Any]:
    """Stress one evaluated game branch with bounded, model-supported mass."""
    hypothesis = validate_llm_response_hypothesis(raw_hypothesis)
    if hypothesis is None:
        return {
            "version": OPPONENT_GAME_VERSION,
            "accepted": False,
            "reason": "missing_or_invalid_response_hypothesis",
        }
    candidate = next((
        item for item in packet.get("candidates") or []
        if str(item.get("action")) == hypothesis["if_action"]
    ), None)
    if candidate is None or not candidate.get("opponent_response_prediction"):
        return {
            "version": OPPONENT_GAME_VERSION,
            "accepted": False,
            "reason": "response_branch_not_evaluated",
            "hypothesis": hypothesis,
        }
    prediction = candidate["opponent_response_prediction"]
    posterior = _posterior(prediction["response_posterior"])
    index = OPPONENT_HYPOTHESES.index(hypothesis["response_preset"])
    support = float(posterior[index] / max(1e-12, float(np.max(posterior))))
    evidence_authority = 1.0 if prediction.get("learned_active") else 0.50
    influence = float(np.clip(
        0.15 * hypothesis["confidence"] * support * evidence_authority,
        0.0,
        0.15,
    ))
    accepted = influence >= 0.01
    stressed = posterior.copy()
    if accepted:
        one_hot = np.zeros_like(stressed)
        one_hot[index] = 1.0
        stressed = normalise_distribution(
            (1.0 - influence) * stressed + influence * one_hot
        )
        overrides = {
            hypothesis["if_action"]: {
                name: float(stressed[position])
                for position, name in enumerate(OPPONENT_HYPOTHESES)
            },
        }
        matrices = {
            str(item.get("action")): dict(item.get(
                "trajectory_continuation_hypothesis_values"
            ) or {})
            for item in packet.get("candidates") or []
            if item.get("trajectory_continuation_hypothesis_values")
        }
        game = attach_second_order_game(
            packet.get("candidates") or [],
            packet.get("opponent_belief"),
            response_memory,
            response_overrides=overrides,
            continuation_value_matrices=matrices,
            trajectory_audit=(packet.get("second_order_game") or {}).get(
                "trajectory_rollout"
            ),
        )
        packet["second_order_game"] = game
        packet["recommended_action"] = game.get(
            "recommended_action", packet.get("recommended_action", "none")
        )
    audit = {
        "version": OPPONENT_GAME_VERSION,
        "accepted": accepted,
        "reason": (
            "bounded_model_supported_response_stress"
            if accepted else "insufficient_response_model_support"
        ),
        "hypothesis": hypothesis,
        "model_support": support,
        "evidence_tier": (
            "validated_response_memory"
            if prediction.get("learned_active") else "structural_response_prior"
        ),
        "bounded_influence": influence,
        "can_update_response_memory": False,
        "causal_interpretation": False,
    }
    packet["llm_response_hypothesis_audit"] = audit
    return audit
