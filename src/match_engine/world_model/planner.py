"""Imagination bonuses for pass and shot selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from src.match_engine.math_utils import finite_float

from src.match_engine.world_model.action_codec import (
    decode_action_kind,
    encode_high_level_action,
    encode_pass_candidate,
    encode_shot_action,
)
from src.match_engine.world_model.config import (
    world_model_outcome_aligned_policy_enabled,
    world_model_plan_enabled,
)
from src.match_engine.world_model.schema import (
    HORIZON_INDEX,
    HORIZON_SCALE_SECONDS,
)
from src.match_engine.world_model.policy_utility import POLICY_UTILITY_VERSION
from src.match_engine.shot_decision import ShotEvidence, shot_counterfactual_value

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState
    from src.match_engine.world_model.inference import WorldModelRuntime


def _planner_authority(runtime, observation: np.ndarray, *, kind: str) -> dict:
    """Compatibility wrapper for runtimes created before authority V2."""
    method = getattr(runtime, "planner_authority", None)
    if callable(method):
        return dict(method(observation, kind=kind))
    confidence = float(runtime.planner_confidence(observation, kind=kind))
    return {
        "authorized": confidence > 0.0,
        "validation_quality": confidence,
        "minimum_validation_quality": 0.0,
        "observation_coverage": 1.0,
        "online_trust": 1.0,
        "decision_confidence": confidence,
    }


def _decision_certainty(runtime) -> float:
    uncertainty = float(getattr(
        runtime, "last_decision_uncertainty",
        getattr(runtime, "last_uncertainty", 1.0),
    ))
    return 1.0 - float(np.clip(uncertainty, 0.0, 1.0))


def _policy_value(
    runtime,
    observation: np.ndarray,
    action: np.ndarray,
    *,
    action_kind: str,
    attacking_home: bool,
    legacy_authority: dict,
    sequence_policy: bool = False,
) -> tuple[float | None, float, dict]:
    """Use M2 value only when its exact action-specific evidence gate is open."""
    if not world_model_outcome_aligned_policy_enabled():
        return (
            float(runtime.score_action(observation, action)),
            _decision_certainty(runtime),
            {
                **legacy_authority,
                "value_source": "legacy_transition_score",
            },
        )
    authority_method = getattr(runtime, "policy_utility_authority", None)
    predictor = getattr(runtime, "predict_policy_utility", None)
    if not callable(authority_method) or not callable(predictor):
        return None, 0.0, {
            **legacy_authority,
            "authorized": False,
            "authority": 0.0,
            "reason": "policy_utility_runtime_contract_missing",
            "value_source": "outcome_aligned_policy_utility",
        }
    policy_authority = dict(authority_method(action_kind))
    authorized = bool(
        legacy_authority.get("authorized")
        and policy_authority.get("authorized")
    )
    authority = min(
        float(legacy_authority.get("decision_confidence", 0.0)),
        float(policy_authority.get("authority", 0.0)),
    ) if authorized else 0.0
    if authority <= 0.0:
        return None, 0.0, {
            **legacy_authority,
            **policy_authority,
            "authorized": False,
            "authority": 0.0,
            "reason": str(policy_authority.get(
                "reason", "policy_utility_evidence_gate_closed",
            )),
            "value_source": (
                "outcome_aligned_two_step_policy_utility"
                if sequence_policy
                else "outcome_aligned_policy_utility"
            ),
        }
    if sequence_policy:
        sequence_authority_method = getattr(
            runtime, "policy_utility_sequence_authority", None,
        )
        sequence_predictor = getattr(
            runtime, "predict_policy_utility_sequence", None,
        )
        two_step_gate_method = getattr(runtime, "two_step_planning_gate", None)
        if not all(callable(method) for method in (
            sequence_authority_method, sequence_predictor, two_step_gate_method,
        )):
            return None, 0.0, {
                **legacy_authority,
                **policy_authority,
                "authorized": False,
                "authority": 0.0,
                "reason": "policy_utility_sequence_runtime_contract_missing",
                "value_source": "outcome_aligned_two_step_policy_utility",
            }
        sequence_authority = dict(sequence_authority_method(
            action_kind, rollout_steps=2,
        ))
        two_step_gate = dict(two_step_gate_method())
        sequence_authorized = bool(
            sequence_authority.get("authorized")
            and two_step_gate.get("active")
        )
        if not sequence_authorized:
            return None, 0.0, {
                **legacy_authority,
                **policy_authority,
                "authorized": False,
                "authority": 0.0,
                "reason": (
                    str(sequence_authority.get("reason"))
                    if not sequence_authority.get("authorized")
                    else str(two_step_gate.get(
                        "reason", "two_step_evidence_gate_closed",
                    ))
                ),
                "value_source": "outcome_aligned_two_step_policy_utility",
                "policy_utility_sequence_gate": sequence_authority,
                "two_step_planning_gate": two_step_gate,
            }
        continuation_policy = list(
            sequence_authority.get("continuation_policy") or []
        )
        try:
            weights = np.asarray([
                float(row["weight"]) for row in continuation_policy
            ], dtype=float)
            action_vectors = [
                np.asarray(row["action_prototype"], dtype=np.float32)
                for row in continuation_policy
            ]
            continuation_kinds = [
                str(row["action_kind"]) for row in continuation_policy
            ]
            policy_valid = bool(
                sequence_authority.get("continuation_support_authorized")
                is True
                and len(weights)
                and len(set(continuation_kinds)) == len(continuation_kinds)
                and np.isfinite(weights).all()
                and bool((weights > 0.0).all())
                and abs(float(weights.sum()) - 1.0) <= 1e-12
                and all(
                    vector.shape == (18,)
                    and np.isfinite(vector).all()
                    and decode_action_kind(vector) == kind
                    for vector, kind in zip(action_vectors, continuation_kinds)
                )
            )
        except (KeyError, TypeError, ValueError):
            weights = np.zeros(0, dtype=float)
            action_vectors = []
            continuation_kinds = []
            policy_valid = False
        if not policy_valid:
            return None, 0.0, {
                **legacy_authority,
                **policy_authority,
                "authorized": False,
                "authority": 0.0,
                "reason": "evidence_supported_continuation_policy_invalid",
                "value_source": "outcome_aligned_two_step_policy_utility",
            }
        if decode_action_kind(action) != action_kind:
            return None, 0.0, {
                **legacy_authority,
                **policy_authority,
                "authorized": False,
                "authority": 0.0,
                "reason": "sequence_first_action_kind_mismatch",
                "value_source": "outcome_aligned_two_step_policy_utility",
            }
        sequence_rows = []
        for weight, continuation_kind, continuation_action, support_row in zip(
            weights, continuation_kinds, action_vectors, continuation_policy,
        ):
            try:
                prediction = dict(sequence_predictor(
                    observation,
                    np.stack((action, continuation_action), axis=0),
                    action_kind=action_kind,
                    attacking_home=attacking_home,
                ))
            except (KeyError, TypeError, ValueError, RuntimeError):
                return None, 0.0, {
                    **legacy_authority,
                    **policy_authority,
                    "authorized": False,
                    "authority": 0.0,
                    "reason": "sequence_prediction_failed_closed",
                    "value_source": "outcome_aligned_two_step_policy_utility",
                }
            prediction_gate = prediction.get("sequence_gate") or {}
            expected_sequence = [
                decode_action_kind(action), str(continuation_kind),
            ]
            if not (
                prediction.get("prediction_source")
                == "explicit_changing_action_sequence_rollout"
                and prediction.get("rollout_steps") == 2
                and prediction.get("planning_mode")
                == "open_loop_evidence_supported_continuation_policy"
                and prediction.get("policy_utility_version")
                == POLICY_UTILITY_VERSION
                and prediction.get("action_sequence") == expected_sequence
                and prediction_gate.get("authorized") is True
                and prediction_gate.get("action_kind") == action_kind
                and prediction_gate.get("rollout_steps") == 2
                and prediction_gate.get("continuation_support_authorized")
                is True
                and prediction_gate.get("continuation_policy")
                == continuation_policy
            ):
                return None, 0.0, {
                    **legacy_authority,
                    **policy_authority,
                    "authorized": False,
                    "authority": 0.0,
                    "reason": "sequence_prediction_identity_or_gate_invalid",
                    "value_source": "outcome_aligned_two_step_policy_utility",
                }
            sequence_rows.append({
                "continuation_action": str(continuation_kind),
                "weight": float(weight),
                "policy_utility": float(prediction["policy_utility"]),
                "uncertainty": float(np.clip(
                    prediction.get("uncertainty", 1.0), 0.0, 1.0,
                )),
                "action_sequence": list(prediction.get(
                    "action_sequence", [],
                )),
                "prediction_source": str(prediction["prediction_source"]),
                "support_samples": int(support_row["samples"]),
                "support_groups": int(support_row["groups"]),
                "empirical_probability": float(
                    support_row["empirical_probability"]
                ),
            })
        values = np.asarray([
            row["policy_utility"] for row in sequence_rows
        ], dtype=float)
        uncertainties = np.asarray([
            row["uncertainty"] for row in sequence_rows
        ], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(uncertainties).all():
            return None, 0.0, {
                **legacy_authority,
                **policy_authority,
                "authorized": False,
                "authority": 0.0,
                "reason": "sequence_prediction_non_finite",
                "value_source": "outcome_aligned_two_step_policy_utility",
            }
        value = float(np.dot(weights, values))
        uncertainty = float(np.clip(np.sqrt(np.dot(
            weights,
            np.square(uncertainties) + np.square(values - value),
        )), 0.0, 1.0))
        return value, 1.0 - uncertainty, {
            **legacy_authority,
            **policy_authority,
            "authorized": True,
            "authority": authority,
            "decision_confidence": authority,
            "decision_certainty": 1.0 - uncertainty,
            "value_source": "outcome_aligned_two_step_policy_utility",
            "policy_utility_version": POLICY_UTILITY_VERSION,
            "predicted_policy_utility": value,
            "rollout_steps": 2,
            "planning_mode": "open_loop_evidence_supported_continuation_policy",
            "comparison_design": "action_vs_zero_persistence_reference",
            "continuation_policy": sequence_rows,
            "policy_utility_sequence_gate": sequence_authority,
            "two_step_planning_gate": two_step_gate,
        }
    horizon_s = float(max(
        0.1,
        action[HORIZON_INDEX] * HORIZON_SCALE_SECONDS
        if len(action) > HORIZON_INDEX else 10.0,
    ))
    prediction = dict(predictor(
        observation,
        action,
        action_kind=action_kind,
        attacking_home=attacking_home,
        horizon_s=horizon_s,
    ))
    uncertainty = float(np.clip(
        prediction.get("uncertainty", 1.0), 0.0, 1.0,
    ))
    certainty = 1.0 - uncertainty
    return float(prediction["policy_utility"]), certainty, {
        **legacy_authority,
        **policy_authority,
        "authorized": True,
        "authority": authority,
        "decision_confidence": authority,
        "decision_certainty": certainty,
        "value_source": "outcome_aligned_policy_utility",
        "policy_utility_version": prediction.get("policy_utility_version"),
        "predicted_policy_utility": float(prediction["policy_utility"]),
    }


def pass_candidate_policy_probabilities(
    runtime: "WorldModelRuntime",
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    meta: List[Tuple],
    attacking_home: bool,
    base_probabilities: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Blend model-ranked executable pass targets into the base policy."""
    base = np.asarray(base_probabilities, dtype=float)
    base = np.clip(base, 0.0, None)
    base /= max(1e-12, float(base.sum()))
    disabled = {
        "applied": False, "reason": "pass_target_policy_disabled",
        "blend_weight": 0.0, "model_values": [],
    }
    if not world_model_plan_enabled() or not meta:
        return base, disabled
    obs = runtime.encode_state(state, attacking_home=attacking_home)
    authority = _planner_authority(runtime, obs, kind="pass")
    confidence = float(authority["decision_confidence"])
    if not authority["authorized"] or confidence <= 0.0:
        return base, {
            **disabled, **authority, "reason": "pass_quality_gate_closed",
        }
    values: list[float] = []
    certainties: list[float] = []
    value_gates: list[dict] = []
    sequence_policy = world_model_outcome_aligned_policy_enabled()
    for candidate in meta:
        recv, kind, tgt, _lane, _press, _omega = candidate[:6]
        success_prior = float(candidate[6]) if len(candidate) > 6 else 0.5
        action = encode_pass_candidate(
            state, carrier, recv, kind, tgt,
            success_p=success_prior,
            horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
        )
        value, certainty, value_gate = _policy_value(
            runtime, obs, action,
            action_kind="pass",
            attacking_home=attacking_home,
            legacy_authority=authority,
            sequence_policy=sequence_policy,
        )
        if value is None:
            return base, {
                **disabled,
                **value_gate,
                "reason": str(value_gate.get(
                    "reason", "policy_utility_evidence_gate_closed",
                )),
            }
        values.append(value)
        certainties.append(certainty)
        value_gates.append(value_gate)
    model_values = np.asarray(values, dtype=float)
    spread = float(np.max(model_values) - np.min(model_values))
    certainty = float(np.dot(base, np.asarray(certainties, dtype=float)))
    if value_gates:
        confidence = min(
            confidence,
            min(float(gate.get(
                "decision_confidence", confidence,
            )) for gate in value_gates),
        )
    blend = float(getattr(runtime.cfg, "planner_blend", 0.0))
    blend_weight = float(np.clip(
        blend * min(confidence, certainty), 0.0, 0.35,
    ))
    if spread <= 1e-9 or blend_weight <= 0.0:
        return base, {
            **authority, "applied": False,
            "reason": "no_reliable_candidate_preference",
            "decision_certainty": certainty,
            "model_value_spread": spread,
            "blend_weight": 0.0,
            "model_values": values,
        }
    center = float(np.dot(base, model_values))
    preference = np.exp(np.clip((model_values - center) / 0.10, -4.0, 4.0))
    target = base * preference
    target /= max(1e-12, float(target.sum()))
    mixed = (1.0 - blend_weight) * base + blend_weight * target
    mixed /= max(1e-12, float(mixed.sum()))
    return mixed, {
        **authority, "applied": True,
        "reason": "validated_executable_pass_target_ranking",
        "decision_certainty": certainty,
        "model_value_spread": spread,
        "blend_weight": blend_weight,
        "model_values": values,
        "value_source": (
            value_gates[0].get("value_source") if value_gates
            else "unavailable"
        ),
        "planning_mode": (
            value_gates[0].get("planning_mode") if value_gates
            else "unavailable"
        ),
        "policy_utility_gate": value_gates[0] if value_gates else {},
    }


def shot_imagination_bonus(
    runtime: "WorldModelRuntime",
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    attacking_home: bool,
    *,
    dist_goal: float,
    blend: float | None = None,
    evidence: ShotEvidence | None = None,
) -> float:
    """Additive utility bonus for choosing shot vs pass/hold."""
    if not world_model_plan_enabled():
        return 0.0
    if evidence is not None and not evidence.enabled:
        return 0.0
    blend = float(blend if blend is not None else runtime.cfg.shot_planner_blend)
    obs = runtime.encode_state(state, attacking_home=attacking_home)
    authority = _planner_authority(runtime, obs, kind="shot")
    confidence = float(authority["decision_confidence"])
    if not authority["authorized"] or confidence <= 0.0:
        return 0.0
    shot_act = encode_shot_action(
        carrier.position,
        xg=max(0.05, 0.35 * (1.0 - dist_goal)),
        horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
    )
    hold_action = encode_high_level_action(
        "hold",
        target=np.asarray(state.ball.position),
        horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
    )
    hold_val = runtime.score_action(obs, hold_action)
    shot_val = runtime.score_shot_action(obs, shot_act, attacking_home=attacking_home)
    certainty = _decision_certainty(runtime)
    advantage = shot_counterfactual_value(
        goal_probability=float(np.clip(shot_val, 0.0, 1.0)),
        rebound_value=0.0, turnover_cost=1.0,
        best_continuation_value=hold_val,
        uncertainty=runtime.last_uncertainty,
    )
    advantage = float(np.clip(advantage, -0.35, 0.35))
    return finite_float(blend * confidence * certainty * advantage, 0.0)


def action_imagination_adjustments(
    runtime: "WorldModelRuntime",
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    attacking_home: bool,
    utils: np.ndarray,
    labels: List[str],
    *,
    dist_goal: float,
    tac: dict | None = None,
    team_shots: int = 0,
    feasible_actions: set[str] | None = None,
) -> np.ndarray:
    """Adjust high-level actions with action-scoped validated evidence."""
    if not world_model_plan_enabled():
        return utils
    out = utils.copy()
    feasible = set(feasible_actions or labels)
    horizon_s = float(getattr(state, "_wm_horizon_s", 10.0))
    shared_pass_target = np.asarray(
        state.ball.position, dtype=np.float32,
    ).copy()
    shared_pass_target[0] = np.clip(
        shared_pass_target[0] + (0.14 if attacking_home else -0.14),
        0.02,
        0.98,
    )
    shared_pass_action = encode_high_level_action(
        "pass", target=shared_pass_target, horizon_s=horizon_s,
    )
    shared_pass_action[13] = 0.72
    shared_hold_action = encode_high_level_action(
        "hold", target=np.asarray(state.ball.position), horizon_s=horizon_s,
    )
    shared_hold_action[13] = 0.90
    adjustments = {str(action): 0.0 for action in labels}
    gates = {
        str(action): {
            "open": False,
            "quality_kind": (
                "reference" if action == "hold" else str(action)
            ),
            "reason": (
                "reference_action_not_directly_promoted" if action == "hold"
                else "no_action_specific_validation"
            ),
            "confidence": 0.0,
            "authority_type": (
                "counterfactual_reference_only" if action == "hold"
                else "action_specific_validation_required"
            ),
            "direct_action_authorized": False,
        }
        for action in labels
    }
    if "pass" in labels and "pass" in feasible:
        pass_index = labels.index("pass")
        observation = runtime.encode_state(state, attacking_home=attacking_home)
        authority = _planner_authority(runtime, observation, kind="pass")
        confidence = float(authority["decision_confidence"])
        gates["pass"]["confidence"] = confidence
        gates["pass"].update(authority)
        if authority["authorized"] and confidence > 0.0:
            pass_action = shared_pass_action
            hold_action = shared_hold_action
            m2_enabled = world_model_outcome_aligned_policy_enabled()
            pass_value, pass_certainty, value_gate = _policy_value(
                runtime, observation, pass_action,
                action_kind="pass",
                attacking_home=attacking_home,
                legacy_authority=authority,
                sequence_policy=m2_enabled,
            )
            if m2_enabled:
                hold_value = 0.0
                hold_certainty = 1.0
                hold_gate = {
                    "authorized": False,
                    "authority": 0.0,
                    "authority_type": "counterfactual_reference_only",
                    "reason": "exact_zero_persistence_reference",
                    "value_source": "same_state_zero_transition_utility",
                    "baseline": "same_state_zero_transition_utility",
                }
            else:
                hold_value, hold_certainty, hold_gate = _policy_value(
                    runtime, observation, hold_action,
                    action_kind="pass",
                    attacking_home=attacking_home,
                    legacy_authority=authority,
                )
            if pass_value is None or hold_value is None:
                gates["pass"].update({
                    **value_gate,
                    "open": False,
                    "direct_action_authorized": False,
                    "reason": str(value_gate.get(
                        "reason", "policy_utility_evidence_gate_closed",
                    )),
                })
            else:
                confidence = min(
                    confidence,
                    float(value_gate.get(
                        "decision_confidence", confidence,
                    )),
                )
                certainty = min(pass_certainty, hold_certainty)
                advantage = float(np.clip(
                    pass_value - hold_value, -0.35, 0.35,
                ))
                adjustment = finite_float(
                    float(runtime.cfg.planner_blend)
                    * confidence * certainty * advantage,
                    0.0,
                )
                adjustment = float(np.clip(adjustment, -0.35, 0.35))
                out[pass_index] += adjustment
                adjustments["pass"] = adjustment
                gates["pass"].update({
                    **value_gate,
                    "open": True,
                    "direct_action_authorized": True,
                    "reason": (
                        "validated_outcome_aligned_pass_vs_zero_persistence"
                        if world_model_outcome_aligned_policy_enabled()
                        else "validated_pass_vs_hold_advantage"
                    ),
                    "pass_value": pass_value,
                    "hold_value": hold_value,
                    "model_advantage": advantage,
                    "certainty": certainty,
                    "decision_certainty": certainty,
                    "decision_confidence": confidence,
                    "policy_blend": float(runtime.cfg.planner_blend),
                    "hold_value_gate": hold_gate,
                })
        else:
            gates["pass"]["reason"] = "pass_quality_gate_closed"
    if "shot" in labels and "shot" in feasible:
        si = labels.index("shot")
        bonus = shot_imagination_bonus(
            runtime, state, carrier, attacking_home, dist_goal=dist_goal
        )
        lb = float((tac or {}).get("low_block", 0.35))
        bonus *= max(0.12, 1.0 - 0.58 * lb)
        bonus *= float(np.exp(-0.05 * team_shots))
        out[si] += bonus
        adjustments["shot"] = float(bonus)
        shot_authority = _planner_authority(
            runtime,
            runtime.encode_state(state, attacking_home=attacking_home),
            kind="shot",
        )
        shot_confidence = float(shot_authority["decision_confidence"])
        shot_certainty = _decision_certainty(runtime)
        shot_blend = float(runtime.cfg.shot_planner_blend)
        authority_weight = shot_blend * shot_confidence * shot_certainty
        effective_advantage = (
            float(np.clip(bonus / authority_weight, -0.35, 0.35))
            if authority_weight > 1e-12 else 0.0
        )
        gates["shot"].update({
            **shot_authority,
            "open": bool(shot_authority["authorized"] and shot_confidence > 0.0),
            "direct_action_authorized": bool(
                shot_authority["authorized"] and shot_confidence > 0.0
            ),
            "confidence": shot_confidence,
            "reason": (
                "validated_shot_vs_continuation_advantage"
                if shot_authority["authorized"] and shot_confidence > 0.0
                else "shot_quality_gate_closed"
            ),
            "decision_certainty": shot_certainty,
            "certainty": shot_certainty,
            "policy_blend": shot_blend,
            "model_advantage": effective_advantage,
        })
    elif "shot" in labels:
        gates["shot"]["reason"] = "action_infeasible"
    if "cross" in labels and "cross" in feasible:
        cross_index = labels.index("cross")
        observation = runtime.encode_state(state, attacking_home=attacking_home)
        cross_authority = _planner_authority(
            runtime, observation, kind="cross",
        )
        cross_confidence = float(cross_authority["decision_confidence"])
        gates["cross"].update({
            **cross_authority,
            "confidence": cross_confidence,
        })
        if cross_authority["authorized"] and cross_confidence > 0.0:
            cross_target = np.asarray(state.ball.position, dtype=np.float32).copy()
            cross_target[0] = 0.90 if attacking_home else 0.10
            cross_target[1] = 0.50
            cross_action = encode_high_level_action(
                "cross", target=cross_target,
                horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
            )
            hold_action = encode_high_level_action(
                "hold", target=np.asarray(state.ball.position),
                horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
            )
            if world_model_outcome_aligned_policy_enabled():
                cross_value, cross_action_certainty, value_gate = _policy_value(
                    runtime, observation, cross_action,
                    action_kind="cross",
                    attacking_home=attacking_home,
                    legacy_authority=cross_authority,
                    sequence_policy=True,
                )
                hold_value = 0.0
                hold_certainty = 1.0
                hold_gate = {
                    "authorized": False,
                    "authority": 0.0,
                    "authority_type": "counterfactual_reference_only",
                    "reason": "exact_zero_persistence_reference",
                    "value_source": "same_state_zero_transition_utility",
                    "baseline": "same_state_zero_transition_utility",
                }
            else:
                cross_value = float(runtime.score_cross_action(
                    observation, cross_action, attacking_home=attacking_home,
                ))
                cross_action_certainty = _decision_certainty(runtime)
                hold_value = float(runtime.score_cross_action(
                    observation, hold_action, attacking_home=attacking_home,
                ))
                hold_certainty = _decision_certainty(runtime)
                value_gate = {
                    **cross_authority,
                    "value_source": "legacy_cross_score",
                }
                hold_gate = dict(value_gate)
            if cross_value is None or hold_value is None:
                gates["cross"].update({
                    **value_gate,
                    "open": False,
                    "direct_action_authorized": False,
                    "reason": str(value_gate.get(
                        "reason", "policy_utility_evidence_gate_closed",
                    )),
                })
                cross_confidence = 0.0
                cross_advantage = 0.0
                cross_adjustment = 0.0
            else:
                cross_confidence = min(
                    cross_confidence,
                    float(value_gate.get(
                        "decision_confidence", cross_confidence,
                    )),
                )
                cross_certainty = min(
                    cross_action_certainty, hold_certainty,
                )
                cross_advantage = float(np.clip(
                    cross_value - hold_value, -0.35, 0.35,
                ))
                cross_blend = float(getattr(
                    runtime.cfg, "cross_planner_blend", 0.20,
                ))
                cross_adjustment = finite_float(
                    cross_blend * cross_confidence * cross_certainty
                    * cross_advantage,
                    0.0,
                )
                cross_adjustment = float(np.clip(
                    cross_adjustment, -0.35, 0.35,
                ))
                out[cross_index] += cross_adjustment
                adjustments["cross"] = cross_adjustment
                gates["cross"].update({
                    **value_gate,
                    "open": True,
                    "direct_action_authorized": True,
                    "reason": (
                        "validated_outcome_aligned_cross_vs_zero_persistence"
                        if world_model_outcome_aligned_policy_enabled()
                        else "validated_cross_vs_continuation_advantage"
                    ),
                    "cross_value": cross_value,
                    "hold_value": hold_value,
                    "model_advantage": cross_advantage,
                    "certainty": cross_certainty,
                    "decision_certainty": cross_certainty,
                    "decision_confidence": cross_confidence,
                    "policy_blend": cross_blend,
                    "hold_value_gate": hold_gate,
                })
        else:
            gates["cross"]["reason"] = "cross_quality_gate_closed"
    elif "cross" in labels:
        gates["cross"]["reason"] = "action_infeasible"
    from src.match_engine.world_model.action_adoption import (
        register_action_policy_opportunity,
    )

    register_action_policy_opportunity(
        state,
        team_id=str(carrier.team_id),
        t_sec=float(getattr(state, "clock_seconds", 0.0)),
        feasible_actions=feasible,
        base_utilities=utils,
        adjusted_utilities=out,
        labels=labels,
        model_adjustments=adjustments,
        quality_gates=gates,
    )
    return out
