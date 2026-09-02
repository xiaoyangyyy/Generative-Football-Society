"""Auditable adoption records for direct world-model action guidance."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


_MAX_SAMPLED_RECORDS = 96
_DIRECTLY_VALIDATED_ACTIONS = frozenset({"pass", "shot", "cross"})
_REFERENCE_ACTION = "hold"
_PROBABILITY_POLICY_VERSION = "validated_action_simplex_v3"
_EXPECTED_CHANGE_ESTIMATOR = "shared_uniform_inverse_cdf_overlap_v1"


def mask_infeasible_action_probabilities(
    labels: Sequence[str], probabilities: Sequence[float],
    feasible_actions: set[str],
) -> np.ndarray:
    """Return a normalized distribution with zero mass on invalid actions."""
    probs = np.asarray(probabilities, dtype=float).copy()
    if probs.shape != (len(labels),):
        raise ValueError("action probability shape does not match labels")
    feasible = np.asarray([
        str(action) in feasible_actions for action in labels
    ], dtype=bool)
    if not feasible.any():
        raise ValueError("at least one action must be feasible")
    probs = np.where(feasible, np.clip(probs, 0.0, None), 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = feasible.astype(float)
        total = float(probs.sum())
    return probs / total


def sample_action_from_uniform(
    labels: Sequence[str], probabilities: Sequence[float], draw: float,
) -> str:
    """Deterministically sample a categorical action from one shared draw."""
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("action probabilities must have positive finite mass")
    probs /= total
    u = float(np.clip(draw, 0.0, np.nextafter(1.0, 0.0)))
    index = min(len(labels) - 1, int(np.searchsorted(
        np.cumsum(probs), u, side="right",
    )))
    return str(labels[index])


def shared_uniform_action_change_probability(
    base_probabilities: Sequence[float],
    adjusted_probabilities: Sequence[float],
) -> float:
    """Return the exact mismatch probability under the sampler's shared draw.

    Total variation is the minimum mismatch probability over all couplings. The
    simulator uses one specific coupling: both categorical distributions are
    inverse-CDF sampled with the same uniform draw and fixed action order. For
    three or more actions its mismatch probability can differ from total
    variation, so compute the overlap of the corresponding CDF intervals.
    """
    base = np.asarray(base_probabilities, dtype=float)
    adjusted = np.asarray(adjusted_probabilities, dtype=float)
    if base.ndim != 1 or base.size < 1 or adjusted.shape != base.shape:
        raise ValueError("shared-uniform probability shapes are invalid")
    if not np.isfinite(base).all() or not np.isfinite(adjusted).all():
        raise ValueError("shared-uniform probabilities must be finite")
    base = np.clip(base, 0.0, None)
    adjusted = np.clip(adjusted, 0.0, None)
    base_total = float(base.sum())
    adjusted_total = float(adjusted.sum())
    if base_total <= 0.0 or adjusted_total <= 0.0:
        raise ValueError("shared-uniform probabilities need positive mass")
    base /= base_total
    adjusted /= adjusted_total
    base_upper = np.cumsum(base)
    adjusted_upper = np.cumsum(adjusted)
    base_lower = np.concatenate(([0.0], base_upper[:-1]))
    adjusted_lower = np.concatenate(([0.0], adjusted_upper[:-1]))
    overlap = np.maximum(
        0.0,
        np.minimum(base_upper, adjusted_upper)
        - np.maximum(base_lower, adjusted_lower),
    ).sum()
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def mix_direct_action_probabilities(
    state,
    *,
    labels: Sequence[str],
    probabilities: Sequence[float],
) -> np.ndarray:
    """Blend every validated feasible action preference into one simplex.

    Each open action gate contributes an advantage and its own bounded
    authority. The strongest authority controls the total interpolation
    budget while relative authorities shape the model target distribution.
    """
    base = np.asarray(probabilities, dtype=float).copy()
    record = getattr(state, "_wm_pending_direct_action_adoption", None)
    if not isinstance(record, dict) or base.shape != (len(labels),):
        return base
    base = np.clip(base, 0.0, None)
    base_total = float(base.sum())
    if not np.isfinite(base_total) or base_total <= 0.0:
        return base
    base /= base_total
    feasible = set(str(action) for action in record.get("feasible_actions", labels))
    gates = record.get("quality_gates") or {}
    signals: list[tuple[int, dict[str, Any], float, float]] = []
    for index, action in enumerate(labels):
        gate = gates.get(str(action)) or {}
        if (
            str(action) not in feasible
            or str(action) not in _DIRECTLY_VALIDATED_ACTIONS
            or not gate.get("open", False)
        ):
            continue
        try:
            advantage = float(gate.get("model_advantage", 0.0))
            confidence = float(gate.get(
                "decision_confidence", gate.get("confidence", 0.0),
            ))
            certainty = float(gate.get(
                "decision_certainty", gate.get("certainty", confidence),
            ))
            policy_blend = float(gate.get("policy_blend", 0.0))
        except (TypeError, ValueError):
            continue
        authority = float(np.clip(
            policy_blend * min(confidence, certainty), 0.0, 0.35,
        ))
        if np.isfinite(advantage) and abs(advantage) > 1e-12 and authority > 0.0:
            signals.append((index, gate, advantage, authority))
    if not signals:
        return base

    blend_weight = max(item[3] for item in signals)
    log_preference = np.zeros(len(labels), dtype=float)
    for index, _gate, advantage, authority in signals:
        log_preference[index] = float(np.clip(
            (advantage / 0.10) * (authority / blend_weight), -4.0, 4.0,
        ))
    target = base * np.exp(log_preference)
    target_total = float(target.sum())
    if not np.isfinite(target_total) or target_total <= 0.0:
        return base
    target /= target_total
    mixed = (1.0 - blend_weight) * base + blend_weight * target
    mixed = np.clip(mixed, 0.0, None)
    mixed /= max(1e-12, float(mixed.sum()))
    record["probability_policy_version"] = _PROBABILITY_POLICY_VERSION
    record["applied_policy_actions"] = [
        str(labels[index]) for index, _gate, _advantage, _authority in signals
    ]
    for index, gate, _advantage, authority in signals:
        gate["model_target_action_probability"] = float(target[index])
        gate["applied_action_authority"] = float(authority)
        gate["applied_probability_blend_weight"] = float(blend_weight)
        if str(labels[index]) == "pass":
            gate["model_target_pass_probability"] = float(target[index])
    signal_indices = {item[0] for item in signals}
    record["redistribution_recipient_actions"] = [
        str(action) for index, action in enumerate(labels)
        if (
            str(action) in feasible
            and index not in signal_indices
            and float(mixed[index] - base[index]) > 1e-12
        )
    ]
    return mixed


def register_action_policy_opportunity(
    state,
    *,
    team_id: str,
    t_sec: float,
    feasible_actions: set[str],
    base_utilities: Sequence[float],
    adjusted_utilities: Sequence[float],
    labels: Sequence[str],
    model_adjustments: dict[str, float],
    quality_gates: dict[str, dict[str, Any]],
) -> str:
    """Register one current-action opportunity without claiming adoption yet."""
    store = getattr(state, "_wm_direct_action_adoption", None)
    if store is None:
        store = {
            "opportunities": 0, "influenced_opportunities": 0,
            "resolved": 0, "adopted": 0, "attributable_adoptions": 0,
            "attribution_eligible_opportunities": 0,
            "counterfactual_action_changes": 0,
            "expected_counterfactual_action_changes": 0.0,
            "probability_shift_sum": 0.0,
            "primary_signal_probability_shift_sum": 0.0,
            "reference_redistribution_opportunities": 0,
            "reference_realized_actions": 0,
            "reference_counterfactual_changes": 0,
            "reference_probability_gain_sum": 0.0,
            "action_signal_breakdown": {}, "records": [],
        }
        state._wm_direct_action_adoption = store
    base = np.asarray(base_utilities, dtype=float)
    adjusted = np.asarray(adjusted_utilities, dtype=float)
    feasible_indices = [
        index for index, action in enumerate(labels) if action in feasible_actions
    ]
    policy_signals: list[tuple[int, float]] = []
    for index in feasible_indices:
        action = str(labels[index])
        if action not in _DIRECTLY_VALIDATED_ACTIONS:
            continue
        gate = quality_gates.get(action) or {}
        if not gate.get("open", False):
            continue
        confidence = float(gate.get(
            "decision_confidence", gate.get("confidence", 0.0),
        ))
        certainty = float(gate.get(
            "decision_certainty", gate.get("certainty", confidence),
        ))
        blend = float(gate.get("policy_blend", 0.0))
        advantage = float(gate.get("model_advantage", 0.0))
        if (
            np.isfinite(advantage) and abs(advantage) > 1e-12
            and confidence > 0.0 and certainty > 0.0 and blend > 0.0
        ):
            policy_signals.append((index, advantage))
    positive_policy = [item for item in policy_signals if item[1] > 0.0]
    if positive_policy:
        recommended_index = max(positive_policy, key=lambda item: item[1])[0]
    else:
        recommended_index = None
    primary_signal_index = (
        max(policy_signals, key=lambda item: abs(item[1]))[0]
        if policy_signals else None
    )
    recommended = (
        str(labels[recommended_index]) if recommended_index is not None else "none"
    )
    # Internal utility movement is diagnostic. An opportunity is influenced
    # only when the probability controller accepts an authorized direct signal.
    influenced = bool(policy_signals)
    opportunity_id = (
        f"direct:{team_id}:{float(t_sec):.3f}:{int(store['opportunities'])}"
    )
    record = {
        "opportunity_id": opportunity_id, "team_id": str(team_id),
        "t_sec": float(t_sec),
        "feasible_actions": sorted(str(action) for action in feasible_actions),
        "recommended_action": recommended,
        "primary_signal_action": (
            str(labels[primary_signal_index])
            if primary_signal_index is not None else "none"
        ),
        "signal_mode": (
            "direct_preference" if positive_policy
            else "suppression_only" if policy_signals else "none"
        ),
        "base_utilities": {
            str(action): float(base[index]) for index, action in enumerate(labels)
        },
        "adjusted_utilities": {
            str(action): float(adjusted[index]) for index, action in enumerate(labels)
        },
        "model_adjustments": {
            str(action): float(model_adjustments.get(str(action), 0.0))
            for action in labels
        },
        "quality_gates": quality_gates, "influenced": influenced,
        "base_probability": None, "adjusted_probability": None,
        "recommended_probability_delta": None, "actual_action": None,
        "primary_signal_probability_delta": None,
        "sampling_uniform": None, "counterfactual_baseline_action": None,
        "policy_changed_action": None,
        "total_variation_distance": None,
        "probability_policy_version": _PROBABILITY_POLICY_VERSION,
        "applied_policy_actions": [],
        "redistribution_recipient_actions": [],
        "reference_action": _REFERENCE_ACTION,
        "reference_action_role": "counterfactual_baseline_only",
        "reference_action_directly_authorized": False,
        "adopted": None, "attribution_eligible": None,
        "resolution": "pending_sample",
    }
    store["opportunities"] += 1
    store["influenced_opportunities"] += int(influenced)
    records = store["records"]
    records.append(record)
    if len(records) > _MAX_SAMPLED_RECORDS:
        del records[0]
    state._wm_pending_direct_action_adoption = record
    return opportunity_id


def record_action_policy_sample(
    state,
    *,
    actual_action: str,
    labels: Sequence[str],
    base_probabilities: Sequence[float],
    adjusted_probabilities: Sequence[float],
    sampling_uniform: float | None = None,
    counterfactual_baseline_action: str | None = None,
    externally_overridden: bool = False,
    cointervention: bool = False,
) -> None:
    """Resolve the latest direct policy opportunity against the actual action."""
    record = getattr(state, "_wm_pending_direct_action_adoption", None)
    if not isinstance(record, dict) or record.get("resolution") != "pending_sample":
        return
    store = getattr(state, "_wm_direct_action_adoption", None)
    if not isinstance(store, dict):
        return
    base = np.asarray(base_probabilities, dtype=float)
    adjusted = np.asarray(adjusted_probabilities, dtype=float)
    record["base_probability"] = {
        str(action): float(base[index]) for index, action in enumerate(labels)
    }
    record["adjusted_probability"] = {
        str(action): float(adjusted[index]) for index, action in enumerate(labels)
    }
    recommended = str(record["recommended_action"])
    recommended_index = labels.index(recommended) if recommended in labels else None
    delta = (
        float(adjusted[recommended_index] - base[recommended_index])
        if recommended_index is not None else 0.0
    )
    primary_signal = str(record.get("primary_signal_action") or "none")
    primary_signal_index = (
        labels.index(primary_signal) if primary_signal in labels else None
    )
    primary_signal_delta = (
        float(adjusted[primary_signal_index] - base[primary_signal_index])
        if primary_signal_index is not None else 0.0
    )
    actual = str(actual_action).lower()
    adopted = recommended != "none" and actual == recommended
    attribution_eligible = bool(
        record["influenced"] and not externally_overridden and not cointervention
    )
    counterfactual = (
        str(counterfactual_baseline_action).lower()
        if counterfactual_baseline_action is not None else None
    )
    changed_action = bool(
        attribution_eligible and counterfactual is not None
        and actual != counterfactual
    )
    total_variation = float(0.5 * np.abs(adjusted - base).sum())
    shared_uniform_change = shared_uniform_action_change_probability(
        base, adjusted,
    )
    reference_index = (
        labels.index(_REFERENCE_ACTION)
        if _REFERENCE_ACTION in labels else None
    )
    reference_delta = (
        float(adjusted[reference_index] - base[reference_index])
        if reference_index is not None else 0.0
    )
    reference_redistributed = bool(
        reference_delta > 1e-12
        and _REFERENCE_ACTION
        in (record.get("redistribution_recipient_actions") or [])
    )
    record.update({
        "recommended_probability_delta": delta, "actual_action": actual,
        "primary_signal_probability_delta": primary_signal_delta,
        "sampling_uniform": (
            float(sampling_uniform) if sampling_uniform is not None else None
        ),
        "counterfactual_baseline_action": counterfactual,
        "policy_changed_action": changed_action,
        "total_variation_distance": total_variation,
        "shared_uniform_change_probability": shared_uniform_change,
        "adopted": adopted, "attribution_eligible": attribution_eligible,
        "resolution": (
            "external_schedule_override" if externally_overridden
            else "cointervention_not_attributable" if cointervention
            else "sampled_after_world_model_adjustment"
        ),
        "reference_action_effect": {
            "action": _REFERENCE_ACTION,
            "role": "counterfactual_baseline_only",
            "directly_authorized": False,
            "probability_delta": reference_delta,
            "received_redistributed_probability": reference_redistributed,
            "realized_as_actual_action": bool(actual == _REFERENCE_ACTION),
            "policy_changed_to_reference": bool(
                changed_action and actual == _REFERENCE_ACTION
            ),
        },
    })
    store["resolved"] += 1
    store["adopted"] += int(adopted)
    store.setdefault("attribution_eligible_opportunities", 0)
    store["attribution_eligible_opportunities"] += int(attribution_eligible)
    store["attributable_adoptions"] += int(adopted and attribution_eligible)
    store.setdefault("counterfactual_action_changes", 0)
    store["counterfactual_action_changes"] += int(changed_action)
    store.setdefault("expected_counterfactual_action_changes", 0.0)
    if attribution_eligible:
        store["expected_counterfactual_action_changes"] += shared_uniform_change
    # Keep the legacy public metric semantically narrow: it measures only a
    # direct positive recommendation. Suppression has its own signed signal.
    store["probability_shift_sum"] += abs(delta)
    store.setdefault("primary_signal_probability_shift_sum", 0.0)
    store["primary_signal_probability_shift_sum"] += abs(primary_signal_delta)
    if reference_redistributed:
        store.setdefault("reference_redistribution_opportunities", 0)
        store.setdefault("reference_probability_gain_sum", 0.0)
        store.setdefault("reference_realized_actions", 0)
        store.setdefault("reference_counterfactual_changes", 0)
        store["reference_redistribution_opportunities"] += 1
        store["reference_probability_gain_sum"] += reference_delta
        store["reference_realized_actions"] += int(actual == _REFERENCE_ACTION)
        store["reference_counterfactual_changes"] += int(
            changed_action and actual == _REFERENCE_ACTION
        )
    breakdown = store.setdefault("action_signal_breakdown", {})
    gates = record.get("quality_gates") or {}
    for action in record.get("applied_policy_actions") or []:
        if action not in labels:
            continue
        index = labels.index(action)
        probability_delta = float(adjusted[index] - base[index])
        gate = gates.get(action) or {}
        bucket = breakdown.setdefault(str(action), {
            "signal_opportunities": 0,
            "positive_guidance": 0,
            "negative_guidance": 0,
            "realized_as_actual_action": 0,
            "locally_changed_to_action": 0,
            "probability_delta_sum": 0.0,
            "absolute_probability_shift_sum": 0.0,
            "authority_sum": 0.0,
        })
        bucket["signal_opportunities"] += 1
        bucket["positive_guidance"] += int(probability_delta > 1e-12)
        bucket["negative_guidance"] += int(probability_delta < -1e-12)
        bucket["realized_as_actual_action"] += int(actual == action)
        bucket["locally_changed_to_action"] += int(changed_action and actual == action)
        bucket["probability_delta_sum"] += probability_delta
        bucket["absolute_probability_shift_sum"] += abs(probability_delta)
        bucket["authority_sum"] += float(gate.get("applied_action_authority", 0.0))
    state._wm_pending_direct_action_adoption = None


def record_pass_target_policy_sample(
    state,
    *,
    candidate_ids: Sequence[str],
    base_probabilities: Sequence[float],
    adjusted_probabilities: Sequence[float],
    selected_index: int,
    counterfactual_index: int,
    sampling_uniform: float,
    evidence: dict[str, Any],
) -> None:
    """Attach exact receiver/target adoption evidence to the high-level action."""
    record = getattr(state, "_wm_pending_direct_action_adoption", None)
    store = getattr(state, "_wm_direct_action_adoption", None)
    if not isinstance(record, dict) or not isinstance(store, dict):
        return
    base = np.asarray(base_probabilities, dtype=float)
    adjusted = np.asarray(adjusted_probabilities, dtype=float)
    total_variation = float(0.5 * np.abs(adjusted - base).sum())
    shared_uniform_change = shared_uniform_action_change_probability(
        base, adjusted,
    )
    changed = bool(selected_index != counterfactual_index)
    applied = bool(evidence.get("applied", False) and total_variation > 1e-12)
    selected_id = str(candidate_ids[selected_index])
    counterfactual_id = str(candidate_ids[counterfactual_index])
    recommended_index = int(np.argmax(adjusted - base)) if applied else selected_index
    record["pass_target_policy"] = {
        **evidence,
        "candidate_ids": [str(item) for item in candidate_ids],
        "base_probability": [float(value) for value in base],
        "adjusted_probability": [float(value) for value in adjusted],
        "sampling_uniform": float(sampling_uniform),
        "selected_candidate_id": selected_id,
        "counterfactual_baseline_candidate_id": counterfactual_id,
        "recommended_candidate_id": str(candidate_ids[recommended_index]),
        "policy_changed_target": bool(applied and changed),
        "total_variation_distance": total_variation,
        "shared_uniform_change_probability": shared_uniform_change,
    }
    store.setdefault("pass_target_opportunities", 0)
    store.setdefault("pass_target_influenced_opportunities", 0)
    store.setdefault("pass_target_changes", 0)
    store.setdefault("pass_target_expected_changes", 0.0)
    store["pass_target_opportunities"] += 1
    store["pass_target_influenced_opportunities"] += int(applied)
    store["pass_target_changes"] += int(applied and changed)
    if applied:
        store["pass_target_expected_changes"] += shared_uniform_change


def direct_action_adoption_diagnostics(state) -> dict[str, Any]:
    """Return bounded records plus aggregate direct-policy adoption metrics."""
    store = getattr(state, "_wm_direct_action_adoption", None)
    if not isinstance(store, dict):
        return {
            "available": False,
            "reason": "no_direct_world_model_action_opportunities",
            "opportunities": 0, "influenced_opportunities": 0,
            "resolved": 0, "adopted": 0, "attributable_adoptions": 0,
            "attribution_eligible_opportunities": 0,
            "counterfactual_action_changes": 0,
            "expected_counterfactual_action_changes": 0.0,
            "counterfactual_change_rate": 0.0,
            "expected_counterfactual_change_rate": 0.0,
            "pass_target_opportunities": 0,
            "pass_target_influenced_opportunities": 0,
            "pass_target_changes": 0,
            "pass_target_expected_changes": 0.0,
            "adoption_rate": 0.0,
            "probability_policy_version": _PROBABILITY_POLICY_VERSION,
            "expected_change_estimator": _EXPECTED_CHANGE_ESTIMATOR,
            "action_signal_breakdown": {},
            "reference_action_breakdown": {
                "action": _REFERENCE_ACTION,
                "role": "counterfactual_baseline_only",
                "direct_signal_opportunities": 0,
                "redistribution_opportunities": 0,
                "realized_actions": 0,
                "counterfactual_changes": 0,
                "mean_probability_gain": 0.0,
            },
            "mean_recommended_probability_shift": 0.0,
            "mean_primary_signal_probability_shift": 0.0,
            "records": [],
        }
    resolved = int(store["resolved"])
    action_breakdown = {}
    for action, raw in (store.get("action_signal_breakdown") or {}).items():
        count = max(0, int(raw.get("signal_opportunities", 0)))
        action_breakdown[str(action)] = {
            **raw,
            "mean_probability_delta": float(
                raw.get("probability_delta_sum", 0.0) / max(1, count)
            ),
            "mean_absolute_probability_shift": float(
                raw.get("absolute_probability_shift_sum", 0.0) / max(1, count)
            ),
            "mean_applied_authority": float(
                raw.get("authority_sum", 0.0) / max(1, count)
            ),
        }
    return {
        "available": True,
        "opportunities": int(store["opportunities"]),
        "influenced_opportunities": int(store["influenced_opportunities"]),
        "resolved": resolved, "adopted": int(store["adopted"]),
        "attributable_adoptions": int(store["attributable_adoptions"]),
        "attribution_eligible_opportunities": int(
            store.get("attribution_eligible_opportunities", 0)
        ),
        "counterfactual_action_changes": int(
            store.get("counterfactual_action_changes", 0)
        ),
        "expected_counterfactual_action_changes": float(
            store.get("expected_counterfactual_action_changes", 0.0)
        ),
        "counterfactual_change_rate": float(
            store.get("counterfactual_action_changes", 0)
            / max(1, store.get("attribution_eligible_opportunities", 0))
        ),
        "expected_counterfactual_change_rate": float(
            store.get("expected_counterfactual_action_changes", 0.0)
            / max(1, store.get("attribution_eligible_opportunities", 0))
        ),
        "pass_target_opportunities": int(
            store.get("pass_target_opportunities", 0)
        ),
        "pass_target_influenced_opportunities": int(
            store.get("pass_target_influenced_opportunities", 0)
        ),
        "pass_target_changes": int(store.get("pass_target_changes", 0)),
        "pass_target_expected_changes": float(
            store.get("pass_target_expected_changes", 0.0)
        ),
        "probability_policy_version": _PROBABILITY_POLICY_VERSION,
        "expected_change_estimator": _EXPECTED_CHANGE_ESTIMATOR,
        "action_signal_breakdown": action_breakdown,
        "reference_action_breakdown": {
            "action": _REFERENCE_ACTION,
            "role": "counterfactual_baseline_only",
            "direct_signal_opportunities": 0,
            "redistribution_opportunities": int(
                store.get("reference_redistribution_opportunities", 0)
            ),
            "realized_actions": int(store.get("reference_realized_actions", 0)),
            "counterfactual_changes": int(
                store.get("reference_counterfactual_changes", 0)
            ),
            "mean_probability_gain": float(
                store.get("reference_probability_gain_sum", 0.0)
                / max(1, store.get("reference_redistribution_opportunities", 0))
            ),
        },
        "adoption_rate": float(store["adopted"] / max(1, resolved)),
        "mean_recommended_probability_shift": float(
            store["probability_shift_sum"] / max(1, resolved)
        ),
        "mean_primary_signal_probability_shift": float(
            store.get("primary_signal_probability_shift_sum", 0.0)
            / max(1, resolved)
        ),
        "records_retained": len(store["records"]),
        "records_truncated": int(store["opportunities"]) > len(store["records"]),
        "records": list(store["records"]),
    }
