"""Imagination bonuses for pass and shot selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from src.match_engine.math_utils import finite_float

from src.match_engine.world_model.action_codec import (
    encode_high_level_action,
    encode_pass_candidate,
    encode_shot_action,
)
from src.match_engine.world_model.config import world_model_plan_enabled
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


def pass_imagination_bonuses(
    runtime: "WorldModelRuntime",
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    meta: List[Tuple],
    attacking_home: bool,
    *,
    blend: float | None = None,
) -> List[float]:
    if not world_model_plan_enabled():
        return [0.0] * len(meta)

    blend = float(blend if blend is not None else runtime.cfg.planner_blend)
    runtime.reset_hidden()
    obs = runtime.encode_state(state, attacking_home=attacking_home)
    authority = _planner_authority(runtime, obs, kind="pass")
    confidence = float(authority["decision_confidence"])
    if not authority["authorized"] or confidence <= 0.0:
        return [0.0] * len(meta)
    hold_action = encode_high_level_action(
        "hold",
        target=np.asarray(state.ball.position),
        horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
    )
    baseline = runtime.score_action(obs, hold_action)
    bonuses: List[float] = []

    for candidate in meta:
        recv, kind, tgt, _lane, _press, _omega = candidate[:6]
        success_prior = float(candidate[6]) if len(candidate) > 6 else 0.5
        act = encode_pass_candidate(
            state,
            carrier,
            recv,
            kind,
            tgt,
            success_p=success_prior,
            horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
        )
        val = runtime.score_action(obs, act)
        certainty = _decision_certainty(runtime)
        advantage = float(np.clip(val - baseline, -0.35, 0.35))
        bonuses.append(finite_float(blend * confidence * certainty * advantage, 0.0))
    return bonuses


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
    for candidate in meta:
        recv, kind, tgt, _lane, _press, _omega = candidate[:6]
        success_prior = float(candidate[6]) if len(candidate) > 6 else 0.5
        action = encode_pass_candidate(
            state, carrier, recv, kind, tgt,
            success_p=success_prior,
            horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
        )
        values.append(float(runtime.score_action(obs, action)))
        certainties.append(_decision_certainty(runtime))
    model_values = np.asarray(values, dtype=float)
    spread = float(np.max(model_values) - np.min(model_values))
    certainty = float(np.dot(base, np.asarray(certainties, dtype=float)))
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
    adjustments = {str(action): 0.0 for action in labels}
    gates = {
        str(action): {
            "open": False,
            "quality_kind": "shot" if action == "shot" else "pass",
            "reason": "no_action_specific_validation",
            "confidence": 0.0,
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
            target = np.asarray(state.ball.position, dtype=np.float32).copy()
            target[0] = np.clip(
                target[0] + (0.14 if attacking_home else -0.14), 0.02, 0.98,
            )
            pass_action = encode_high_level_action(
                "pass", target=target,
                horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
            )
            pass_action[13] = 0.72
            hold_action = encode_high_level_action(
                "hold", target=np.asarray(state.ball.position),
                horizon_s=float(getattr(state, "_wm_horizon_s", 10.0)),
            )
            hold_action[13] = 0.90
            pass_value = float(runtime.score_action(observation, pass_action))
            hold_value = float(runtime.score_action(observation, hold_action))
            certainty = _decision_certainty(runtime)
            advantage = float(np.clip(pass_value - hold_value, -0.35, 0.35))
            adjustment = finite_float(
                float(runtime.cfg.planner_blend) * confidence * certainty * advantage,
                0.0,
            )
            adjustment = float(np.clip(adjustment, -0.35, 0.35))
            out[pass_index] += adjustment
            adjustments["pass"] = adjustment
            gates["pass"].update({
                "open": True,
                "reason": "validated_pass_vs_hold_advantage",
                "pass_value": pass_value,
                "hold_value": hold_value,
                "model_advantage": advantage,
                "certainty": certainty,
                "decision_certainty": certainty,
                "decision_confidence": confidence,
                "policy_blend": float(runtime.cfg.planner_blend),
            })
        else:
            gates["pass"]["reason"] = "pass_quality_gate_closed"
    if "shot" in labels:
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
        gates["shot"].update({
            **shot_authority,
            "open": bool(shot_authority["authorized"] and shot_confidence > 0.0),
            "confidence": shot_confidence,
            "reason": (
                "validated_shot_value" if shot_confidence > 0.0
                else "shot_quality_gate_closed"
            ),
        })
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
