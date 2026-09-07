"""Actor-centred transition utility shared by training and runtime planning."""

from __future__ import annotations

from typing import Any

import numpy as np


POLICY_UTILITY_VERSION = "actor_centered_transition_utility_v1"
GOAL_DELTA_WEIGHT = 1.0
TERRITORIAL_DELTA_WEIGHT = 0.35
POSSESSION_DELTA_WEIGHT = 0.15
POLICY_UTILITY_LIMIT = 2.0
CONTINUATION_SUPPORT_VERSION = 1
CONTINUATION_ACTION_DIM = 18
CONTINUATION_ACTION_KINDS = (
    "pass", "shot", "cross", "hold", "intercept", "tackle", "other",
)
PLANNABLE_CONTINUATION_ACTIONS = frozenset({
    "pass", "shot", "cross", "hold",
})
MINIMUM_CONTINUATION_SAMPLES = 32
MINIMUM_CONTINUATION_GROUPS = 4
MINIMUM_SUPPORTED_CONTINUATION_MASS = 0.95


def _validate_last_dimension(value: Any, *, name: str) -> None:
    shape = getattr(value, "shape", ())
    if not shape or int(shape[-1]) < 210:
        raise ValueError(f"{name} must end with at least 210 observation features")


def transition_policy_utility_numpy(
    current_observation: np.ndarray,
    future_observation: np.ndarray,
    *,
    attacking_home: np.ndarray | bool | None = None,
) -> dict[str, np.ndarray]:
    """Return bounded realized utility from the original actor's perspective.

    Every component is a transition delta. In particular, possession is not an
    absolute reward: keeping the ball has zero delta while losing it is
    negative. This makes same-state persistence an exact zero-utility baseline.
    """
    current = np.asarray(current_observation, dtype=np.float64)
    future = np.asarray(future_observation, dtype=np.float64)
    _validate_last_dimension(current, name="current_observation")
    _validate_last_dimension(future, name="future_observation")
    if current.shape != future.shape:
        raise ValueError("policy utility observations must have equal shapes")
    if attacking_home is None:
        if current.shape[-1] < 307:
            raise ValueError("attack flag is unavailable from the observation")
        attack = current[..., -1] > 0.5
    else:
        attack = np.asarray(attacking_home, dtype=bool)
    direction = np.where(attack, 1.0, -1.0)
    goal_delta = direction * 5.0 * (
        (future[..., 204] - future[..., 205])
        - (current[..., 204] - current[..., 205])
    )
    territorial_delta = direction * (
        future[..., 200] - current[..., 200]
    )
    current_possession = np.where(
        attack, current[..., 209], 1.0 - current[..., 209],
    )
    future_possession = np.where(
        attack, future[..., 209], 1.0 - future[..., 209],
    )
    possession_delta = future_possession - current_possession
    raw = (
        GOAL_DELTA_WEIGHT * goal_delta
        + TERRITORIAL_DELTA_WEIGHT * territorial_delta
        + POSSESSION_DELTA_WEIGHT * possession_delta
    )
    utility = np.clip(raw, -POLICY_UTILITY_LIMIT, POLICY_UTILITY_LIMIT)
    return {
        "policy_utility": utility,
        "goal_diff_delta": goal_delta,
        "territorial_delta": territorial_delta,
        "possession_delta": possession_delta,
    }


def transition_policy_utility_tensor(
    current_observation,
    future_observation,
    *,
    attacking_home=None,
):
    """Differentiable torch equivalent used by the auxiliary training loss."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by trainer
        raise RuntimeError("PyTorch is required for tensor policy utility") from exc
    current = current_observation
    future = future_observation
    _validate_last_dimension(current, name="current_observation")
    _validate_last_dimension(future, name="future_observation")
    if tuple(current.shape) != tuple(future.shape):
        if (
            future.ndim == current.ndim + 1
            and tuple(future.shape[1:]) == tuple(current.shape)
        ):
            current = current.unsqueeze(0).expand_as(future)
        else:
            raise ValueError(
                "policy utility observations must be equal or member-broadcastable"
            )
    attack = (
        current[..., -1] > 0.5
        if attacking_home is None
        else torch.as_tensor(
            attacking_home, dtype=torch.bool, device=current.device,
        )
    )
    direction = torch.where(
        attack,
        torch.ones((), dtype=current.dtype, device=current.device),
        -torch.ones((), dtype=current.dtype, device=current.device),
    )
    goal_delta = direction * 5.0 * (
        (future[..., 204] - future[..., 205])
        - (current[..., 204] - current[..., 205])
    )
    territorial_delta = direction * (
        future[..., 200] - current[..., 200]
    )
    current_possession = torch.where(
        attack, current[..., 209], 1.0 - current[..., 209],
    )
    future_possession = torch.where(
        attack, future[..., 209], 1.0 - future[..., 209],
    )
    possession_delta = future_possession - current_possession
    utility = torch.clamp(
        GOAL_DELTA_WEIGHT * goal_delta
        + TERRITORIAL_DELTA_WEIGHT * territorial_delta
        + POSSESSION_DELTA_WEIGHT * possession_delta,
        -POLICY_UTILITY_LIMIT,
        POLICY_UTILITY_LIMIT,
    )
    return {
        "policy_utility": utility,
        "goal_diff_delta": goal_delta,
        "territorial_delta": territorial_delta,
        "possession_delta": possession_delta,
    }


def build_continuation_support_evidence(
    first_action_kinds: np.ndarray,
    continuation_action_kinds: np.ndarray,
    groups: np.ndarray,
    continuation_actions: np.ndarray,
    *,
    first_action_kind: str,
) -> dict[str, Any]:
    """Build grouped second-action support from leakage-cleaned vectors."""
    first = np.asarray(first_action_kinds).astype("U")
    second = np.asarray(continuation_action_kinds).astype("U")
    group_ids = np.asarray(groups).astype("U")
    vectors = np.asarray(continuation_actions, dtype=np.float64)
    if not (len(first) == len(second) == len(group_ids) == len(vectors)):
        raise ValueError("continuation support arrays must align")
    if vectors.ndim != 2 or vectors.shape[1] != CONTINUATION_ACTION_DIM:
        raise ValueError("continuation action vectors have invalid shape")
    selected = first == str(first_action_kind)
    total = int(selected.sum())
    actions: dict[str, Any] = {}
    supported_mass = 0.0
    for action in CONTINUATION_ACTION_KINDS:
        mask = selected & (second == action)
        samples = int(mask.sum())
        action_groups = int(len(np.unique(group_ids[mask]))) if samples else 0
        probability = float(samples / total) if total else 0.0
        prototype = np.mean(vectors[mask], axis=0).tolist() if samples else []
        supported = bool(
            action in PLANNABLE_CONTINUATION_ACTIONS
            and samples >= MINIMUM_CONTINUATION_SAMPLES
            and action_groups >= MINIMUM_CONTINUATION_GROUPS
            and len(prototype) == CONTINUATION_ACTION_DIM
            and np.isfinite(prototype).all()
        )
        supported_mass += probability if supported else 0.0
        actions[action] = {
            "samples": samples,
            "groups": action_groups,
            "empirical_probability": probability,
            "plannable": action in PLANNABLE_CONTINUATION_ACTIONS,
            "supported": supported,
            "action_prototype": prototype,
        }
    return {
        "version": CONTINUATION_SUPPORT_VERSION,
        "source": "grouped_holdout_observed_second_actions",
        "first_action_kind": str(first_action_kind),
        "total_samples": total,
        "minimum_samples": MINIMUM_CONTINUATION_SAMPLES,
        "minimum_groups": MINIMUM_CONTINUATION_GROUPS,
        "minimum_supported_probability_mass": MINIMUM_SUPPORTED_CONTINUATION_MASS,
        "supported_probability_mass": float(supported_mass),
        "excluded_probability_mass": float(1.0 - supported_mass if total else 1.0),
        "sufficient": bool(
            total and supported_mass >= MINIMUM_SUPPORTED_CONTINUATION_MASS
        ),
        "actions": actions,
    }


def policy_utility_validation_gate(
    evidence: dict[str, Any] | None,
    *,
    action_kind: str,
    minimum_samples: int = 96,
    minimum_groups: int = 6,
    minimum_skill: float = 0.02,
) -> dict[str, Any]:
    """Fail closed unless an exact action branch beats zero persistence."""
    contract = evidence if isinstance(evidence, dict) else {}
    actions = contract.get("actions") or {}
    profile = actions.get(str(action_kind)) or {}
    samples = int(profile.get("samples", 0) or 0)
    groups = int(profile.get("groups", 0) or 0)
    model_mse = float(profile.get("model_mse", float("inf")))
    persistence_mse = float(profile.get(
        "persistence_mse", float("inf"),
    ))
    skill = float(profile.get("skill_vs_persistence", 0.0) or 0.0)
    correlation = float(profile.get(
        "prediction_target_correlation", -1.0,
    ))
    trained = bool(
        contract.get("version") == 1
        and contract.get("target_version") == POLICY_UTILITY_VERSION
        and contract.get("trained_with_policy_utility_objective") is True
        and float(contract.get("configured_loss_weight", 0.0) or 0.0) > 0.0
        and int(contract.get("optimization_steps", 0) or 0) > 0
    )
    finite = all(np.isfinite(value) for value in (
        model_mse, persistence_mse, skill, correlation,
    ))
    active = bool(
        trained
        and finite
        and samples >= minimum_samples
        and groups >= minimum_groups
        and persistence_mse > 1e-12
        and model_mse < persistence_mse
        and skill >= minimum_skill
        and correlation >= 0.0
    )
    sample_factor = samples / (samples + 192.0)
    group_factor = groups / (groups + 12.0)
    skill_factor = float(np.clip(skill / 0.25, 0.0, 1.0))
    authority = float(np.clip(
        sample_factor * group_factor * skill_factor if active else 0.0,
        0.0,
        1.0,
    ))
    return {
        "version": 1,
        "active": active,
        "authorized": active,
        "authority": authority,
        "reason": (
            "grouped_heldout_policy_utility_gain"
            if active else
            "policy_utility_training_contract_missing"
            if not trained else
            "policy_utility_evidence_gate_closed"
        ),
        "action_kind": str(action_kind),
        "target_version": contract.get("target_version"),
        "samples": samples,
        "groups": groups,
        "model_mse": model_mse if np.isfinite(model_mse) else None,
        "persistence_mse": (
            persistence_mse if np.isfinite(persistence_mse) else None
        ),
        "skill_vs_persistence": skill,
        "prediction_target_correlation": (
            correlation if np.isfinite(correlation) else None
        ),
        "minimum_samples": minimum_samples,
        "minimum_groups": minimum_groups,
        "minimum_skill_vs_persistence": minimum_skill,
    }


def continuation_support_validation_gate(
    evidence: dict[str, Any] | None,
    *,
    first_action_kind: str,
    expected_samples: int,
) -> dict[str, Any]:
    """Recompute a normalized continuation policy from recorded evidence."""
    contract = evidence if isinstance(evidence, dict) else {}
    try:
        actions = contract.get("actions") or {}
        total = int(contract.get("total_samples", 0) or 0)
        structural = bool(
            contract.get("version") == CONTINUATION_SUPPORT_VERSION
            and contract.get("source")
            == "grouped_holdout_observed_second_actions"
            and contract.get("first_action_kind") == str(first_action_kind)
            and total == int(expected_samples)
            and set(actions) == set(CONTINUATION_ACTION_KINDS)
            and contract.get("minimum_samples") == MINIMUM_CONTINUATION_SAMPLES
            and contract.get("minimum_groups") == MINIMUM_CONTINUATION_GROUPS
            and contract.get("minimum_supported_probability_mass")
            == MINIMUM_SUPPORTED_CONTINUATION_MASS
        )
        counted = 0
        probability_sum = 0.0
        supported_mass = 0.0
        supported_rows = []
        for action in CONTINUATION_ACTION_KINDS:
            row = actions.get(action) or {}
            samples = int(row.get("samples", 0) or 0)
            action_groups = int(row.get("groups", 0) or 0)
            probability = float(row.get("empirical_probability", -1.0))
            expected_probability = float(samples / total) if total else 0.0
            prototype = np.asarray(
                row.get("action_prototype") or [], dtype=np.float64,
            )
            prototype_valid = bool(
                prototype.shape == (CONTINUATION_ACTION_DIM,)
                and np.isfinite(prototype).all()
            )
            expected_supported = bool(
                action in PLANNABLE_CONTINUATION_ACTIONS
                and samples >= MINIMUM_CONTINUATION_SAMPLES
                and action_groups >= MINIMUM_CONTINUATION_GROUPS
                and prototype_valid
            )
            structural = bool(
                structural
                and samples >= 0
                and 0 <= action_groups <= samples
                and np.isfinite(probability)
                and abs(probability - expected_probability) <= 1e-12
                and row.get("plannable")
                is (action in PLANNABLE_CONTINUATION_ACTIONS)
                and row.get("supported") is expected_supported
            )
            counted += samples
            probability_sum += probability
            if expected_supported:
                supported_mass += expected_probability
                supported_rows.append({
                    "action_kind": action,
                    "samples": samples,
                    "groups": action_groups,
                    "empirical_probability": expected_probability,
                    "action_prototype": prototype.tolist(),
                })
        structural = bool(
            structural
            and counted == total
            and abs(probability_sum - (1.0 if total else 0.0)) <= 1e-12
            and abs(float(contract.get(
                "supported_probability_mass", -1.0,
            )) - supported_mass) <= 1e-12
            and abs(float(contract.get(
                "excluded_probability_mass", -1.0,
            )) - (1.0 - supported_mass if total else 1.0)) <= 1e-12
        )
    except (TypeError, ValueError, OverflowError):
        structural = False
        total = 0
        supported_mass = 0.0
        supported_rows = []
    active = bool(
        structural
        and supported_rows
        and supported_mass >= MINIMUM_SUPPORTED_CONTINUATION_MASS
        and contract.get("sufficient") is True
    )
    policy = []
    if active:
        policy = [
            {
                **row,
                "weight": float(row["empirical_probability"] / supported_mass),
            }
            for row in supported_rows
        ]
    return {
        "version": CONTINUATION_SUPPORT_VERSION,
        "active": active,
        "authorized": active,
        "reason": (
            "grouped_holdout_continuation_support"
            if active else "continuation_support_contract_invalid"
            if not structural else "continuation_support_mass_insufficient"
        ),
        "first_action_kind": str(first_action_kind),
        "total_samples": total,
        "supported_probability_mass": float(supported_mass),
        "minimum_samples": MINIMUM_CONTINUATION_SAMPLES,
        "minimum_groups": MINIMUM_CONTINUATION_GROUPS,
        "minimum_supported_probability_mass": MINIMUM_SUPPORTED_CONTINUATION_MASS,
        "continuation_policy": policy,
    }


def policy_utility_sequence_validation_gate(
    evidence: dict[str, Any] | None,
    *,
    action_kind: str,
    rollout_steps: int = 2,
    minimum_samples: int = 96,
    minimum_groups: int = 6,
    minimum_skill: float = 0.02,
) -> dict[str, Any]:
    """Authorize utility only for an explicitly trained changing-action depth."""
    contract = evidence if isinstance(evidence, dict) else {}
    steps = max(1, int(rollout_steps))
    base = policy_utility_validation_gate(
        contract,
        action_kind=action_kind,
        minimum_samples=minimum_samples,
        minimum_groups=minimum_groups,
        minimum_skill=minimum_skill,
    )
    profile = ((contract.get("actions") or {}).get(str(action_kind)) or {})
    continuation_gate = continuation_support_validation_gate(
        profile.get("continuation_support"),
        first_action_kind=action_kind,
        expected_samples=int(base.get("samples", 0) or 0),
    )
    sequence_contract = bool(
        steps == 2
        and contract.get("objective_scope") == "two_step_policy_utility"
        and contract.get("rollout_steps") == steps
        and contract.get("action_sequence") == "observed_changing_actions"
        and contract.get("trained_with_action_sequence_objective") is True
        and contract.get("grouped_holdout") is True
    )
    active = bool(
        base["authorized"]
        and sequence_contract
        and continuation_gate["authorized"]
    )
    return {
        **base,
        "version": 3,
        "active": active,
        "authorized": active,
        "authority": base["authority"] if active else 0.0,
        "reason": (
            "grouped_heldout_changing_action_policy_utility_gain"
            if active
            else "policy_utility_sequence_training_contract_missing"
            if not sequence_contract
            else str(base["reason"])
            if not base["authorized"]
            else str(continuation_gate["reason"])
        ),
        "rollout_steps": steps,
        "action_sequence": contract.get("action_sequence"),
        "objective_scope": contract.get("objective_scope"),
        "trained_with_action_sequence_objective": bool(
            contract.get("trained_with_action_sequence_objective") is True
        ),
        "continuation_support_authorized": bool(
            continuation_gate["authorized"]
        ),
        "continuation_support_gate": continuation_gate,
        "continuation_policy": list(
            continuation_gate.get("continuation_policy") or []
        ),
    }
