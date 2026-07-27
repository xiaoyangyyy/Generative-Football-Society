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
    confidence = runtime.planner_confidence(obs, kind="pass")
    if confidence <= 0.0:
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
        certainty = 1.0 - float(np.clip(runtime.last_uncertainty, 0.0, 1.0))
        advantage = float(np.clip(val - baseline, -0.35, 0.35))
        bonuses.append(finite_float(blend * confidence * certainty * advantage, 0.0))
    return bonuses


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
    confidence = runtime.planner_confidence(obs, kind="shot")
    if confidence <= 0.0:
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
    certainty = 1.0 - float(np.clip(runtime.last_uncertainty, 0.0, 1.0))
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
) -> np.ndarray:
    """Adjust [pass, shot, cross, hold] utilities using world model imagination."""
    if not world_model_plan_enabled():
        return utils
    out = utils.copy()
    if "shot" in labels:
        si = labels.index("shot")
        bonus = shot_imagination_bonus(
            runtime, state, carrier, attacking_home, dist_goal=dist_goal
        )
        lb = float((tac or {}).get("low_block", 0.35))
        bonus *= max(0.12, 1.0 - 0.58 * lb)
        bonus *= float(np.exp(-0.05 * team_shots))
        out[si] += bonus
    return out
