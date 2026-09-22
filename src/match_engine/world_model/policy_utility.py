"""Actor-centred transition utility shared by training and runtime planning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


POLICY_UTILITY_VERSION = "actor_centered_transition_utility_v1"
GOAL_DELTA_WEIGHT = 1.0
TERRITORIAL_DELTA_WEIGHT = 0.35
POSSESSION_DELTA_WEIGHT = 0.15
POLICY_UTILITY_LIMIT = 2.0
CONTINUATION_SUPPORT_VERSION = 2
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
MINIMUM_POLICY_UTILITY_SAMPLES = 96
MINIMUM_POLICY_UTILITY_GROUPS = 6
MINIMUM_POLICY_UTILITY_SKILL = 0.02


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
    actor_perspective: str = "pooled",
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
    if actor_perspective not in {"pooled", "home", "away"}:
        raise ValueError("continuation actor perspective is invalid")
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
        "actor_perspective": actor_perspective,
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


def _policy_utility_profile_gate(
    contract: dict[str, Any],
    profile: dict[str, Any],
    *,
    action_kind: str,
    actor_perspective: str,
    minimum_samples: int,
    minimum_groups: int,
    minimum_skill: float,
) -> dict[str, Any]:
    """Validate one recorded action-and-perspective utility profile."""
    try:
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
    except (TypeError, ValueError, OverflowError):
        samples = groups = 0
        model_mse = persistence_mse = float("inf")
        skill = 0.0
        correlation = -1.0
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
    expected_skill = (
        (persistence_mse - model_mse) / persistence_mse
        if finite and persistence_mse > 1e-12 else 0.0
    )
    metric_identity_verified = bool(
        finite
        and persistence_mse > 1e-12
        and abs(skill - expected_skill) <= 1e-9
    )
    active = bool(
        trained
        and finite
        and metric_identity_verified
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
        "actor_perspective": actor_perspective,
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
        "metric_identity_verified": metric_identity_verified,
        "minimum_samples": minimum_samples,
        "minimum_groups": minimum_groups,
        "minimum_skill_vs_persistence": minimum_skill,
    }


def _perspective_profiles_consistent(
    root_profile: dict[str, Any],
    perspective_profiles: dict[str, Any],
) -> bool:
    """Replay pooled counts and additive errors from the two actor partitions."""
    if set(perspective_profiles) != {"home", "away"}:
        return False
    try:
        root_samples = int(root_profile.get("samples", 0) or 0)
        root_groups = int(root_profile.get("groups", 0) or 0)
        side_samples = {
            label: int((perspective_profiles[label]).get("samples", 0) or 0)
            for label in ("home", "away")
        }
        side_groups = {
            label: int((perspective_profiles[label]).get("groups", 0) or 0)
            for label in ("home", "away")
        }
        if (
            root_samples <= 0
            or any(value < 0 for value in side_samples.values())
            or sum(side_samples.values()) != root_samples
            or root_groups < max(side_groups.values())
            or root_groups > sum(side_groups.values())
        ):
            return False
        for metric in ("model_mse", "persistence_mse"):
            pooled = float(root_profile.get(metric, float("nan")))
            weighted = sum(
                side_samples[label]
                * float(perspective_profiles[label].get(metric, float("nan")))
                for label in ("home", "away")
            ) / root_samples
            if (
                not np.isfinite(pooled)
                or not np.isfinite(weighted)
                or abs(pooled - weighted) > 1e-9
            ):
                return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def policy_utility_validation_gate(
    evidence: dict[str, Any] | None,
    *,
    action_kind: str,
    attacking_home: bool | None = None,
    minimum_samples: int = MINIMUM_POLICY_UTILITY_SAMPLES,
    minimum_groups: int = MINIMUM_POLICY_UTILITY_GROUPS,
    minimum_skill: float = MINIMUM_POLICY_UTILITY_SKILL,
) -> dict[str, Any]:
    """Fail closed unless an exact action branch beats zero persistence."""
    contract = evidence if isinstance(evidence, dict) else {}
    actions = contract.get("actions") or {}
    root_profile = actions.get(str(action_kind)) or {}
    perspective_profiles = root_profile.get("perspectives") or {}
    perspectives_consistent = _perspective_profiles_consistent(
        root_profile, perspective_profiles,
    )
    if attacking_home is None and set(perspective_profiles) == {"home", "away"}:
        perspective_gates = {
            label: policy_utility_validation_gate(
                contract,
                action_kind=action_kind,
                attacking_home=(label == "home"),
                minimum_samples=minimum_samples,
                minimum_groups=minimum_groups,
                minimum_skill=minimum_skill,
            )
            for label in ("home", "away")
        }
        overall = _policy_utility_profile_gate(
            contract,
            root_profile,
            action_kind=action_kind,
            actor_perspective="pooled",
            minimum_samples=minimum_samples,
            minimum_groups=minimum_groups,
            minimum_skill=minimum_skill,
        )
        active = bool(
            overall["authorized"]
            and perspectives_consistent
            and all(gate["authorized"] for gate in perspective_gates.values())
        )
        return {
            **overall,
            "version": 2,
            "active": active,
            "authorized": active,
            "authority": (
                min(gate["authority"] for gate in perspective_gates.values())
                if active else 0.0
            ),
            "reason": (
                "perspective_conditioned_policy_utility_gain"
                if active else "perspective_conditioned_policy_utility_gate_closed"
            ),
            "perspective_conditioned": True,
            "perspective_profile_identity_verified": perspectives_consistent,
            "perspective_gates": perspective_gates,
        }
    if attacking_home is not None:
        label = "home" if attacking_home else "away"
        result = _policy_utility_profile_gate(
            contract,
            perspective_profiles.get(label) or {},
            action_kind=action_kind,
            actor_perspective=label,
            minimum_samples=minimum_samples,
            minimum_groups=minimum_groups,
            minimum_skill=minimum_skill,
        )
        active = bool(result["authorized"] and perspectives_consistent)
        return {
            **result,
            "version": 2,
            "active": active,
            "authorized": active,
            "authority": result["authority"] if active else 0.0,
            "reason": (
                str(result["reason"])
                if active or not result["authorized"]
                else "perspective_profile_identity_invalid"
            ),
            "perspective_conditioned": bool(perspectives_consistent),
            "perspective_profile_identity_verified": perspectives_consistent,
        }
    return {
        **_policy_utility_profile_gate(
            contract,
            root_profile,
            action_kind=action_kind,
            actor_perspective="pooled",
            minimum_samples=minimum_samples,
            minimum_groups=minimum_groups,
            minimum_skill=minimum_skill,
        ),
        "perspective_conditioned": False,
        "perspective_profile_identity_verified": False,
    }


def _policy_gate_metrics_integrity(gate: Mapping[str, Any]) -> bool:
    try:
        samples = int(gate.get("samples", 0) or 0)
        groups = int(gate.get("groups", 0) or 0)
        model_mse = float(gate.get("model_mse", float("nan")))
        persistence_mse = float(gate.get(
            "persistence_mse", float("nan"),
        ))
        skill = float(gate.get("skill_vs_persistence", float("nan")))
        correlation = float(gate.get(
            "prediction_target_correlation", float("nan"),
        ))
        authority = float(gate.get("authority", float("nan")))
        expected_skill = (persistence_mse - model_mse) / persistence_mse
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
    return bool(
        gate.get("active") is True
        and gate.get("authorized") is True
        and gate.get("metric_identity_verified") is True
        and gate.get("target_version") == POLICY_UTILITY_VERSION
        and gate.get("minimum_samples") == MINIMUM_POLICY_UTILITY_SAMPLES
        and gate.get("minimum_groups") == MINIMUM_POLICY_UTILITY_GROUPS
        and gate.get("minimum_skill_vs_persistence")
        == MINIMUM_POLICY_UTILITY_SKILL
        and samples >= MINIMUM_POLICY_UTILITY_SAMPLES
        and groups >= MINIMUM_POLICY_UTILITY_GROUPS
        and all(np.isfinite(value) for value in (
            model_mse,
            persistence_mse,
            skill,
            correlation,
            authority,
        ))
        and persistence_mse > 1e-12
        and model_mse < persistence_mse
        and abs(skill - expected_skill) <= 1e-9
        and skill >= MINIMUM_POLICY_UTILITY_SKILL
        and correlation >= 0.0
        and 0.0 < authority <= 1.0
    )


def policy_utility_perspective_gate_integrity(
    gate: Mapping[str, Any] | None,
    *,
    actor_perspective: str,
) -> bool:
    """Replay one side's emitted authority fields without source evidence."""
    contract = gate if isinstance(gate, Mapping) else {}
    return bool(
        actor_perspective in {"home", "away"}
        and contract.get("perspective_conditioned") is True
        and contract.get("perspective_profile_identity_verified") is True
        and contract.get("actor_perspective") == actor_perspective
        and _policy_gate_metrics_integrity(contract)
    )


def policy_utility_aggregate_gate_integrity(
    gate: Mapping[str, Any] | None,
) -> bool:
    """Replay pooled and home/away authority arithmetic from an emitted gate."""
    contract = gate if isinstance(gate, Mapping) else {}
    perspective_gates = contract.get("perspective_gates") or {}
    if (
        contract.get("perspective_conditioned") is not True
        or contract.get("perspective_profile_identity_verified") is not True
        or contract.get("actor_perspective") != "pooled"
        or set(perspective_gates) != {"home", "away"}
        or not _policy_gate_metrics_integrity(contract)
        or not all(
            isinstance(perspective_gates.get(label), Mapping)
            and policy_utility_perspective_gate_integrity(
                perspective_gates[label],
                actor_perspective=label,
            )
            for label in ("home", "away")
        )
    ):
        return False
    try:
        root_samples = int(contract["samples"])
        root_groups = int(contract["groups"])
        side_samples = {
            label: int(perspective_gates[label]["samples"])
            for label in ("home", "away")
        }
        side_groups = {
            label: int(perspective_gates[label]["groups"])
            for label in ("home", "away")
        }
        if (
            sum(side_samples.values()) != root_samples
            or root_groups < max(side_groups.values())
            or root_groups > sum(side_groups.values())
        ):
            return False
        for metric in ("model_mse", "persistence_mse"):
            pooled = float(contract[metric])
            weighted = sum(
                side_samples[label] * float(perspective_gates[label][metric])
                for label in ("home", "away")
            ) / root_samples
            if abs(pooled - weighted) > 1e-9:
                return False
        expected_authority = min(
            float(perspective_gates[label]["authority"])
            for label in ("home", "away")
        )
        return abs(float(contract["authority"]) - expected_authority) <= 1e-12
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def continuation_support_validation_gate(
    evidence: dict[str, Any] | None,
    *,
    first_action_kind: str,
    expected_samples: int,
    actor_perspective: str = "pooled",
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
            and contract.get("actor_perspective") == actor_perspective
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
        "actor_perspective": actor_perspective,
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
    attacking_home: bool | None = None,
    minimum_samples: int = MINIMUM_POLICY_UTILITY_SAMPLES,
    minimum_groups: int = MINIMUM_POLICY_UTILITY_GROUPS,
    minimum_skill: float = MINIMUM_POLICY_UTILITY_SKILL,
) -> dict[str, Any]:
    """Authorize utility only for an explicitly trained changing-action depth."""
    contract = evidence if isinstance(evidence, dict) else {}
    steps = max(1, int(rollout_steps))
    profile = ((contract.get("actions") or {}).get(str(action_kind)) or {})
    perspective_profiles = profile.get("perspectives") or {}
    sequence_contract = bool(
        steps == 2
        and contract.get("objective_scope") == "two_step_policy_utility"
        and contract.get("rollout_steps") == steps
        and contract.get("action_sequence") == "observed_changing_actions"
        and contract.get("trained_with_action_sequence_objective") is True
        and contract.get("grouped_holdout") is True
    )
    if attacking_home is None and set(perspective_profiles) == {"home", "away"}:
        perspective_gates = {
            label: policy_utility_sequence_validation_gate(
                contract,
                action_kind=action_kind,
                rollout_steps=steps,
                attacking_home=(label == "home"),
                minimum_samples=minimum_samples,
                minimum_groups=minimum_groups,
                minimum_skill=minimum_skill,
            )
            for label in ("home", "away")
        }
        base = policy_utility_validation_gate(
            contract,
            action_kind=action_kind,
            minimum_samples=minimum_samples,
            minimum_groups=minimum_groups,
            minimum_skill=minimum_skill,
        )
        active = bool(
            base["authorized"]
            and sequence_contract
            and all(gate["authorized"] for gate in perspective_gates.values())
        )
        return {
            **base,
            "version": 4,
            "active": active,
            "authorized": active,
            "authority": (
                min(gate["authority"] for gate in perspective_gates.values())
                if active else 0.0
            ),
            "reason": (
                "perspective_conditioned_changing_action_policy_utility_gain"
                if active
                else "policy_utility_sequence_training_contract_missing"
                if not sequence_contract
                else "perspective_conditioned_sequence_gate_closed"
            ),
            "rollout_steps": steps,
            "action_sequence": contract.get("action_sequence"),
            "objective_scope": contract.get("objective_scope"),
            "trained_with_action_sequence_objective": bool(
                contract.get("trained_with_action_sequence_objective") is True
            ),
            "perspective_conditioned": True,
            "perspective_gates": perspective_gates,
            "continuation_support_authorized": bool(
                all(
                    gate["continuation_support_authorized"]
                    for gate in perspective_gates.values()
                )
            ),
            "continuation_support_gate": {},
            "continuation_policy": [],
        }
    base = policy_utility_validation_gate(
        contract,
        action_kind=action_kind,
        attacking_home=attacking_home,
        minimum_samples=minimum_samples,
        minimum_groups=minimum_groups,
        minimum_skill=minimum_skill,
    )
    label = (
        "home" if attacking_home is True
        else "away" if attacking_home is False
        else "pooled"
    )
    selected_profile = (
        perspective_profiles.get(label) or {}
        if attacking_home is not None else profile
    )
    continuation_gate = continuation_support_validation_gate(
        selected_profile.get("continuation_support"),
        first_action_kind=action_kind,
        expected_samples=int(base.get("samples", 0) or 0),
        actor_perspective=label,
    )
    active = bool(
        base["authorized"]
        and sequence_contract
        and continuation_gate["authorized"]
        and (
            base.get("perspective_conditioned") is True
            if attacking_home is not None else True
        )
    )
    return {
        **base,
        "version": 4,
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
        "actor_perspective": label,
        "attacking_home": attacking_home,
        "continuation_support_authorized": bool(
            continuation_gate["authorized"]
        ),
        "continuation_support_gate": continuation_gate,
        "continuation_policy": list(
            continuation_gate.get("continuation_policy") or []
        ),
    }
