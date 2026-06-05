"""Apply System 2 plans back into System 1 state."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.cognitive.events import (
    ENTITY_TIER_ASSISTANT,
    ENTITY_TIER_COACH,
    ENTITY_TIER_CROWD,
    ENTITY_TIER_PLAYER,
    ENTITY_TIER_REFEREE,
    CognitiveTriggerEvent,
)
from src.match_engine.cognitive.schemas import (
    validate_assistant_plan,
    validate_coach_plan,
    validate_crowd_plan,
    validate_player_plan,
    validate_referee_plan,
)
from src.match_engine.math_utils import tanh_clip
from src.match_engine.tactical_profile import build_tactical_vector_for_agent, sync_agent_controls_from_vector

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState
    from src.simulation.agent import SocietyAgent


def apply_coach_plan_to_team(
    state: "MatchAffectiveState",
    team_id: str,
    plan: Dict[str, Any],
    agent: Optional["SocietyAgent"] = None,
) -> Dict[str, Any]:
    plan = validate_coach_plan(plan or {})
    team = state.team(team_id)
    cur = dict(team.coach.tactical_current)
    for k, d in (plan.get("controls_delta") or {}).items():
        if k in cur:
            cur[k] = float(np.clip(cur[k] + d, 0.0, 1.0))
    for k, v in (plan.get("tactical_hints") or {}).items():
        if k in cur:
            cur[k] = float(np.clip(0.6 * cur[k] + 0.4 * v, 0.0, 1.0))
    team.coach.tactical_current = cur
    applied = {"tactical_current": {k: cur[k] for k in list(cur.keys())[:6]}}

    if agent is not None:
        for k, d in (plan.get("controls_delta") or {}).items():
            if k in agent.tactical_controls:
                agent.tactical_controls[k] = float(np.clip(agent.tactical_controls[k] + d, 0.0, 1.0))
        tac = build_tactical_vector_for_agent(agent)
        sync_agent_controls_from_vector(agent, tac)
        team.coach.tactical_current = dict(tac)
        agent.tactical_vector = tac
        if plan.get("tactical_preset") and getattr(agent, "coach_profile", None):
            cp = agent.coach_profile
            if hasattr(cp, "preferred_preset"):
                cp.preferred_preset = str(plan["tactical_preset"])
    return applied


def apply_player_plan(player: "PlayerAffectiveState", plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = validate_player_plan(plan)
    eb = plan.get("emotion_bias") or {}
    idx = {"pride": 0, "anger": 1, "fear": 2, "determination": 3}
    for k, i in idx.items():
        if k in eb:
            player.z_emo[i] = float(player.z_emo[i] + eb[k])
    return {"player_id": player.player_id, "emotion_bias": eb}


def apply_referee_plan(state: "MatchAffectiveState", plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = validate_referee_plan(plan)
    ref = state.referee
    ref.strictness_base = float(np.clip(ref.strictness_base + plan.get("strictness_delta", 0), 0.2, 0.95))
    ref.bias_home = float(tanh_clip(ref.bias_home + plan.get("bias_delta", 0)))
    ref.z_calm = float(ref.z_calm + plan.get("calm_delta", 0))
    if plan.get("var_recommendation", "none").lower() in ("review", "var", "check"):
        ref.controversy_integral = float(ref.controversy_integral + 0.15)
    return {
        "strictness_base": ref.strictness_base,
        "bias_home": ref.bias_home,
    }


def apply_assistant_plan(state: "MatchAffectiveState", side: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = validate_assistant_plan(plan)
    ar = None
    for a in getattr(state.referee, "assistants", []) or []:
        if a.side == side:
            ar = a
            break
    if ar is None:
        return {}
    ar.offside_strictness = float(np.clip(ar.offside_strictness + plan.get("offside_strictness_delta", 0), 0.2, 0.9))
    if plan.get("recommend_card_review"):
        state.referee.controversy_integral = float(state.referee.controversy_integral + 0.08)
        state.referee.bias_home = float(tanh_clip(state.referee.bias_home + 0.02))
    return {"side": side, "offside_strictness": ar.offside_strictness}


def apply_crowd_plan(state: "MatchAffectiveState", plan: Dict[str, Any], *, numeric: bool = True) -> Dict[str, Any]:
    plan = validate_crowd_plan(plan)
    if numeric:
        state.crowd.psi = float(tanh_clip(state.crowd.psi + plan.get("psi_pulse", 0)))
        state.crowd.event_intensity_integral += abs(plan.get("psi_pulse", 0))
    return {"psi": state.crowd.psi, "narrative": plan.get("chant_narrative", "")}


def apply_cognitive_plan(
    trig: CognitiveTriggerEvent,
    state: "MatchAffectiveState",
    plan: Dict[str, Any],
    *,
    home_agent: Optional["SocietyAgent"] = None,
    away_agent: Optional["SocietyAgent"] = None,
    crowd_numeric: bool = True,
) -> Dict[str, Any]:
    tier = trig.entity_tier
    if tier == ENTITY_TIER_COACH and trig.team_id:
        ag = home_agent if trig.team_id == state.home.team_id else away_agent
        return apply_coach_plan_to_team(state, trig.team_id, plan, ag)
    if tier == ENTITY_TIER_PLAYER:
        for p in state.home.players + state.away.players:
            if p.player_id == trig.entity_id:
                return apply_player_plan(p, plan)
        return {}
    if tier == ENTITY_TIER_REFEREE:
        return apply_referee_plan(state, plan)
    if tier == ENTITY_TIER_ASSISTANT:
        side = trig.entity_id.split(":")[-1] if ":" in trig.entity_id else "left"
        return apply_assistant_plan(state, side, plan)
    if tier == ENTITY_TIER_CROWD:
        return apply_crowd_plan(state, plan, numeric=crowd_numeric)
    return {}
