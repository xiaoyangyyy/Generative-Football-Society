"""Actor-centred transition utility shared by training and runtime planning."""

from __future__ import annotations

from typing import Any

import numpy as np


POLICY_UTILITY_VERSION = "actor_centered_transition_utility_v1"
GOAL_DELTA_WEIGHT = 1.0
TERRITORIAL_DELTA_WEIGHT = 0.35
POSSESSION_DELTA_WEIGHT = 0.15
POLICY_UTILITY_LIMIT = 2.0


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
