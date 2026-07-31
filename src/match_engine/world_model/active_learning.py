"""Risk-constrained active learning for world-model policy decisions."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


DEFAULT_ACTIVE_LEARNING_CONFIG = {
    "enabled": True,
    "max_regret": 0.08,
    "turnover_margin": 0.12,
    "min_information_value": 0.45,
    "budget_fraction": 0.25,
    "budget_window": 20,
    "intervention_scale": 0.50,
}


def _bounded(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(np.clip(number, low, high))


def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, number))


def normalize_active_learning_config(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = {**DEFAULT_ACTIVE_LEARNING_CONFIG, **(config or {})}
    return {
        "enabled": bool(raw["enabled"]),
        "max_regret": _bounded(raw["max_regret"], 0.0, 0.30, 0.08),
        "turnover_margin": _bounded(
            raw["turnover_margin"], 0.0, 0.40, 0.12,
        ),
        "min_information_value": _bounded(
            raw["min_information_value"], 0.0, 1.0, 0.45,
        ),
        "budget_fraction": _bounded(
            raw["budget_fraction"], 0.0, 0.50, 0.25,
        ),
        "budget_window": _bounded_int(raw["budget_window"], 4, 100, 20),
        "intervention_scale": _bounded(
            raw["intervention_scale"], 0.0, 1.0, 0.50,
        ),
    }


def _context_key(context: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(context.get(key, "unknown")) for key in (
        "team_id", "opponent_team_id", "zone", "score_state", "match_phase",
    ))


def _record_context(record: dict[str, Any]) -> tuple[str, ...]:
    baseline = record.get("outcome_baseline") or {}
    return _context_key({
        "team_id": record.get("team_id", "unknown"),
        "opponent_team_id": baseline.get("opponent_team_id", "unknown"),
        "zone": baseline.get("zone", "unknown"),
        "score_state": baseline.get("score_state", "unknown"),
        "match_phase": baseline.get("match_phase", "unknown"),
    })


def _coverage_counts(
    records: Iterable[dict[str, Any]], context: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    target_context = _context_key(context)
    action_counts: dict[str, int] = {}
    contextual_counts: dict[str, int] = {}
    for record in records:
        action = str(record.get("intervention_actual_action") or "")
        if not action:
            continue
        action_counts[action] = action_counts.get(action, 0) + 1
        if _record_context(record) == target_context:
            contextual_counts[action] = contextual_counts.get(action, 0) + 1
    return action_counts, contextual_counts


def _forecast_information(
    candidate: dict[str, Any],
) -> tuple[float, float, float]:
    predictions = candidate.get("multi_horizon_predictions") or {}
    epistemic_values = [
        _bounded(
            item.get("epistemic_uncertainty", item.get("uncertainty", 1.0)),
            0.0, 1.0, 1.0,
        )
        for item in predictions.values()
        if isinstance(item, dict)
    ]
    aleatoric_values = [
        _bounded(item.get("aleatoric_uncertainty", 0.0), 0.0, 1.0, 0.0)
        for item in predictions.values()
        if isinstance(item, dict)
    ]
    utilities = [
        float(item.get("raw_policy_utility", item.get("policy_utility", 0.0)))
        for item in predictions.values()
        if isinstance(item, dict)
    ]
    epistemic = (
        float(np.mean(epistemic_values)) if epistemic_values
        else _bounded(
            candidate.get(
                "learnable_uncertainty",
                candidate.get("epistemic_uncertainty", candidate.get(
                    "uncertainty", 1.0,
                )),
            ),
            0.0, 1.0, 1.0,
        )
    )
    epistemic = max(
        epistemic,
        _bounded(candidate.get("learnable_uncertainty", 0.0), 0.0, 1.0, 0.0),
    )
    aleatoric = (
        float(np.mean(aleatoric_values)) if aleatoric_values
        else _bounded(
            candidate.get("aleatoric_uncertainty", 0.0), 0.0, 1.0, 0.0,
        )
    )
    disagreement = (
        float(np.clip((max(utilities) - min(utilities)) / 0.25, 0.0, 1.0))
        if len(utilities) >= 2 else 0.0
    )
    return epistemic, aleatoric, disagreement


def _cross_match_coverage(candidate: dict[str, Any]) -> tuple[int, int]:
    """Read only compatible, non-quarantined memory evidence."""
    action_samples = contextual_samples = 0
    for prediction in (
        candidate.get("multi_horizon_predictions") or {}
    ).values():
        memory = prediction.get("residual_memory") or {}
        scope = str(memory.get("scope", ""))
        if scope in {
            "", "insufficient_contextual_history",
            "quarantined_distribution_drift",
        }:
            continue
        samples = _bounded_int(memory.get("samples", 0), 0, 10**9, 0)
        action_samples = max(action_samples, samples)
        if scope in {"team_opponent_context", "action_context"}:
            contextual_samples = max(contextual_samples, samples)
    return action_samples, contextual_samples


def _budget_status(
    records: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    window = records[-config["budget_window"]:]
    explored = sum(
        (record.get("active_learning") or {}).get("decision_mode") == "explore"
        for record in window
    )
    # One initial exploration is permitted; afterwards enforce the rolling rate.
    allowed = max(1, int(math.ceil(
        config["budget_fraction"] * max(1, len(window) + 1)
    )))
    return {
        "window_decisions": len(window),
        "exploration_decisions": explored,
        "allowed_exploration_decisions": allowed,
        "available": explored < allowed,
    }


def build_active_learning_advice(
    candidates: list[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    context: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate candidate information value and select a safe experiment."""
    settings = normalize_active_learning_config(config)
    history = list(records)
    budget = _budget_status(history, settings)
    eligible_candidates = [
        item for item in candidates
        if float(item.get("effective_confidence", 0.0)) > 0.0
    ]
    exploit = max(
        eligible_candidates,
        key=lambda item: float(item.get("risk_adjusted_value", -1.0)),
        default=None,
    )
    action_counts, contextual_counts = _coverage_counts(history, context)
    best_value = float(exploit.get("risk_adjusted_value", -1.0)) if exploit else -1.0
    best_turnover = float(
        (exploit.get("event_probabilities") or {}).get("turnover", 1.0)
    ) if exploit else 1.0
    scored = []
    for candidate in candidates:
        action = str(candidate.get("action", "unknown"))
        epistemic, aleatoric, disagreement = _forecast_information(candidate)
        memory_action_samples, memory_context_samples = (
            _cross_match_coverage(candidate)
        )
        action_samples = action_counts.get(action, 0) + memory_action_samples
        context_samples = (
            contextual_counts.get(action, 0) + memory_context_samples
        )
        action_novelty = 1.0 / math.sqrt(1.0 + action_samples)
        context_novelty = 1.0 / math.sqrt(1.0 + context_samples)
        information_value = float(np.clip(
            0.55 * epistemic
            + 0.20 * context_novelty
            + 0.15 * action_novelty
            + 0.10 * disagreement,
            0.0, 1.0,
        ))
        value = float(candidate.get("risk_adjusted_value", -1.0))
        regret = max(0.0, best_value - value)
        turnover = float(
            (candidate.get("event_probabilities") or {}).get("turnover", 1.0)
        )
        safe = bool(
            exploit is not None
            and float(candidate.get("effective_confidence", 0.0)) > 0.0
            and regret <= settings["max_regret"]
            and turnover <= min(0.75, best_turnover + settings["turnover_margin"])
        )
        evidence = {
            "action": action,
            "information_value": information_value,
            "model_uncertainty": epistemic,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "multi_horizon_disagreement": disagreement,
            "action_samples": action_samples,
            "context_action_samples": context_samples,
            "estimated_regret": regret,
            "safe_to_explore": safe,
        }
        candidate["active_learning"] = dict(evidence)
        scored.append(evidence)
    alternatives = [
        item for item in scored
        if item["safe_to_explore"]
        and exploit is not None
        and item["action"] != str(exploit.get("action"))
    ]
    target = max(
        alternatives, key=lambda item: item["information_value"], default=None,
    )
    eligible = bool(
        settings["enabled"]
        and budget["available"]
        and target is not None
        and target["information_value"] >= settings["min_information_value"]
    )
    if not settings["enabled"]:
        reason = "disabled"
    elif exploit is None:
        reason = "quality_gate_closed"
    elif not budget["available"]:
        reason = "exploration_budget_exhausted"
    elif target is None:
        reason = "no_safe_non_greedy_action"
    elif target["information_value"] < settings["min_information_value"]:
        reason = "information_value_below_threshold"
    else:
        reason = "safe_information_gain_opportunity"
    return {
        "version": 1,
        "eligible": eligible,
        "reason": reason,
        "exploit_action": str(exploit.get("action")) if exploit else "none",
        "exploration_action": target["action"] if eligible and target else "none",
        "expected_information_value": (
            target["information_value"] if eligible and target else 0.0
        ),
        "estimated_regret": target["estimated_regret"] if target else None,
        "budget": budget,
        "constraints": settings,
        "candidate_evidence": scored,
        "policy": (
            "Exploration is optional, bounded, randomized downstream, and must "
            "never override the LLM or the simulator action sampler."
        ),
    }


def _prediction_uncertainty_component(
    record: dict[str, Any], action: str, horizon: str, field: str,
) -> float | None:
    prediction = (
        (record.get("action_outcome_predictions") or {})
        .get(action, {})
        .get(horizon)
    )
    if not isinstance(prediction, dict):
        return None
    fallback = prediction.get("uncertainty", 1.0) if field == (
        "epistemic_uncertainty"
    ) else 0.0
    return _bounded(prediction.get(field, fallback), 0.0, 1.0, fallback)


def active_learning_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    """Measure exploration execution, coverage, and later uncertainty change."""
    clusters = [list(cluster) for cluster in record_clusters]
    records = [record for cluster in clusters for record in cluster]
    exploration = [
        record for record in records
        if (record.get("active_learning") or {}).get("decision_mode") == "explore"
    ]
    executed = [
        record for record in exploration
        if record.get("intervention_actual_action")
        == (record.get("active_learning") or {}).get("exploration_action")
    ]
    epistemic_reductions = []
    aleatoric_changes = []
    for cluster in clusters:
        for index, record in enumerate(cluster):
            learning = record.get("active_learning") or {}
            if learning.get("decision_mode") != "explore":
                continue
            action = str(learning.get("exploration_action", ""))
            if record.get("intervention_actual_action") != action:
                continue
            current_context = _record_context(record)
            predictions = (
                (record.get("action_outcome_predictions") or {}).get(action, {})
            )
            for later in cluster[index + 1:]:
                if _record_context(later) != current_context:
                    continue
                for horizon in predictions:
                    before = _prediction_uncertainty_component(
                        record, action, horizon, "epistemic_uncertainty",
                    )
                    after = _prediction_uncertainty_component(
                        later, action, horizon, "epistemic_uncertainty",
                    )
                    before_aleatoric = _prediction_uncertainty_component(
                        record, action, horizon, "aleatoric_uncertainty",
                    )
                    after_aleatoric = _prediction_uncertainty_component(
                        later, action, horizon, "aleatoric_uncertainty",
                    )
                    if before is not None and after is not None:
                        epistemic_reductions.append(before - after)
                    if before_aleatoric is not None and after_aleatoric is not None:
                        aleatoric_changes.append(after_aleatoric - before_aleatoric)
                break
    actions = sorted({
        str(record.get("intervention_actual_action"))
        for record in executed if record.get("intervention_actual_action")
    })
    contexts = {_record_context(record) for record in executed}
    return {
        "version": 1,
        "exploration_decisions": len(exploration),
        "executed_explorations": len(executed),
        "execution_rate": len(executed) / max(1, len(exploration)),
        "explored_actions": actions,
        "explored_action_count": len(actions),
        "explored_context_count": len(contexts),
        "uncertainty_reduction_pairs": len(epistemic_reductions),
        "epistemic_reduction_pairs": len(epistemic_reductions),
        "mean_later_uncertainty_reduction": (
            float(np.mean(epistemic_reductions))
            if epistemic_reductions else 0.0
        ),
        "mean_later_epistemic_reduction": (
            float(np.mean(epistemic_reductions))
            if epistemic_reductions else 0.0
        ),
        "mean_later_aleatoric_change": (
            float(np.mean(aleatoric_changes)) if aleatoric_changes else 0.0
        ),
        "positive_uncertainty_reduction_rate": (
            sum(value > 0.0 for value in epistemic_reductions)
            / max(1, len(epistemic_reductions))
        ),
        "interpretation": (
            "Uncertainty reduction is descriptive until model updates are "
            "randomized independently of policy exploration."
        ),
    }
