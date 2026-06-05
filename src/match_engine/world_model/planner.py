"""Imagination bonuses for pass and shot selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from src.match_engine.world_model.action_codec import encode_pass_candidate, encode_shot_action, zero_action
from src.match_engine.world_model.config import world_model_plan_enabled

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
    baseline = runtime.score_action(obs, zero_action())
    bonuses: List[float] = []

    for recv, kind, tgt, _lane, _press, _omega in meta:
        act = encode_pass_candidate(state, carrier, recv, kind, tgt, success_p=0.55)
        val = runtime.score_action(obs, act)
        bonuses.append(blend * (val - baseline))
    return bonuses


def shot_imagination_bonus(
    runtime: "WorldModelRuntime",
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    attacking_home: bool,
    *,
    dist_goal: float,
    blend: float | None = None,
) -> float:
    """Additive utility bonus for choosing shot vs pass/hold."""
    if not world_model_plan_enabled():
        return 0.0
    blend = float(blend if blend is not None else runtime.cfg.shot_planner_blend)
    obs = runtime.encode_state(state, attacking_home=attacking_home)
    shot_act = encode_shot_action(carrier.position, xg=max(0.05, 0.35 * (1.0 - dist_goal)))
    hold_val = runtime.score_action(obs, zero_action())
    shot_val = runtime.score_shot_action(obs, shot_act, attacking_home=attacking_home)
    return blend * (shot_val - hold_val)


def action_imagination_adjustments(
    runtime: "WorldModelRuntime",
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    attacking_home: bool,
    utils: np.ndarray,
    labels: List[str],
    *,
    dist_goal: float,
) -> np.ndarray:
    """Adjust [pass, shot, cross, hold] utilities using world model imagination."""
    if not world_model_plan_enabled():
        return utils
    out = utils.copy()
    if "shot" in labels:
        si = labels.index("shot")
        out[si] += shot_imagination_bonus(
            runtime, state, carrier, attacking_home, dist_goal=dist_goal
        )
    return out
