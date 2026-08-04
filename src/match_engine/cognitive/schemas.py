"""Validate and clamp cognitive LLM JSON plans."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

FORBIDDEN_KEYS = frozenset(
    {"score", "goals", "xg", "final_score", "winner", "home_goals", "away_goals"}
)


def _clip_delta(v: Any, limit: float = 0.15) -> float:
    try:
        return float(np.clip(float(v), -limit, limit))
    except (TypeError, ValueError):
        return 0.0


def _clip01(v: Any) -> float:
    try:
        return float(np.clip(float(v), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.5


def sanitize_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"error": "invalid_plan_type"}
    for k in FORBIDDEN_KEYS:
        raw.pop(k, None)
    return raw


def validate_coach_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = sanitize_plan(plan)
    out: Dict[str, Any] = {
        "reasoning": str(plan.get("reasoning", ""))[:500],
        "confidence": _clip01(plan.get("confidence", 0.6)),
    }
    cd = plan.get("controls_delta") or {}
    if isinstance(cd, dict):
        out["controls_delta"] = {
            k: _clip_delta(cd[k])
            for k in ("pressing_intensity", "risk_budget", "line_height", "rotation_aggressiveness")
            if k in cd
        }
    hints = plan.get("tactical_hints") or {}
    if isinstance(hints, dict):
        out["tactical_hints"] = {k: _clip01(v) for k, v in hints.items() if isinstance(v, (int, float))}
    if plan.get("tactical_preset"):
        out["tactical_preset"] = str(plan["tactical_preset"])
    if plan.get("sub_intent"):
        out["sub_intent"] = str(plan["sub_intent"])[:120]
    action = str(plan.get("world_model_action", "none")).lower()
    out["world_model_action"] = (
        action if action in {"hold", "pass", "cross", "shot", "none"}
        else "none"
    )
    mode = str(plan.get("world_model_decision_mode", "exploit")).lower()
    out["world_model_decision_mode"] = (
        mode if mode in {"exploit", "explore", "decline"} else "exploit"
    )
    if plan.get("world_model_rationale"):
        out["world_model_rationale"] = str(
            plan["world_model_rationale"]
        )[:300]
    from src.match_engine.world_model.llm_deliberation_focus import (
        validate_llm_deliberation_focus,
    )

    deliberation_focus = validate_llm_deliberation_focus(
        plan.get("world_model_deliberation_focus")
    )
    if deliberation_focus is not None:
        out["world_model_deliberation_focus"] = deliberation_focus
    from src.match_engine.world_model.opponent_belief import (
        validate_llm_opponent_change_claim,
        validate_llm_opponent_hypothesis,
    )

    hypothesis = validate_llm_opponent_hypothesis(
        plan.get("opponent_hypothesis")
    )
    if hypothesis is not None:
        out["opponent_hypothesis"] = hypothesis
    change_claim = validate_llm_opponent_change_claim(
        plan.get("opponent_change_claim")
    )
    if change_claim is not None:
        out["opponent_change_claim"] = change_claim
    from src.match_engine.world_model.opponent_game import (
        validate_llm_response_hypothesis,
    )

    response_hypothesis = validate_llm_response_hypothesis(
        plan.get("opponent_response_hypothesis")
    )
    if response_hypothesis is not None:
        out["opponent_response_hypothesis"] = response_hypothesis
    from src.match_engine.world_model.llm_critic import (
        validate_llm_world_model_critique,
    )

    critique = validate_llm_world_model_critique(
        plan.get("world_model_critique")
    )
    if critique is not None:
        out["world_model_critique"] = critique
    from src.match_engine.world_model.llm_event_hypothesis import (
        validate_llm_event_hypothesis,
    )

    event_hypothesis = validate_llm_event_hypothesis(
        plan.get("world_model_event_hypothesis")
    )
    if event_hypothesis is not None:
        out["world_model_event_hypothesis"] = event_hypothesis
    from src.match_engine.world_model.event_option import (
        validate_llm_event_option,
    )

    event_option = validate_llm_event_option(
        plan.get("world_model_event_option")
    )
    if event_option is not None:
        out["world_model_event_option"] = event_option
    from src.match_engine.world_model.contrastive_explanation import (
        validate_llm_contrastive_claim,
    )

    contrastive_claim = validate_llm_contrastive_claim(
        plan.get("world_model_contrastive_claim")
    )
    if contrastive_claim is not None:
        out["world_model_contrastive_claim"] = contrastive_claim
    from src.match_engine.world_model.risk_certificate import (
        validate_llm_risk_constraint,
    )

    risk_constraint = validate_llm_risk_constraint(
        plan.get("world_model_risk_constraint")
    )
    if risk_constraint is not None:
        out["world_model_risk_constraint"] = risk_constraint
    from src.match_engine.world_model.distributional_claim import (
        validate_llm_distributional_claim,
    )

    distributional_claim = validate_llm_distributional_claim(
        plan.get("world_model_distributional_claim")
    )
    if distributional_claim is not None:
        out["world_model_distributional_claim"] = distributional_claim
    from src.match_engine.world_model.risk_preference import (
        validate_llm_risk_preference,
    )

    risk_preference = validate_llm_risk_preference(
        plan.get("world_model_risk_preference")
    )
    if risk_preference is not None:
        out["world_model_risk_preference"] = risk_preference
    from src.match_engine.world_model.opponent_information_query import (
        validate_llm_opponent_information_query,
    )

    information_query = validate_llm_opponent_information_query(
        plan.get("opponent_information_query")
    )
    if information_query is not None:
        out["opponent_information_query"] = information_query
    from src.match_engine.world_model.opponent_information_policy import (
        validate_llm_opponent_information_policy,
    )

    information_policy = validate_llm_opponent_information_policy(
        plan.get("opponent_information_policy")
    )
    if information_policy is not None:
        out["opponent_information_policy"] = information_policy
    from src.match_engine.world_model.belief_space_meta_planner import (
        validate_llm_belief_space_meta_plan,
    )

    meta_plan = validate_llm_belief_space_meta_plan(
        plan.get("world_model_belief_space_meta_plan")
    )
    if meta_plan is not None:
        out["world_model_belief_space_meta_plan"] = meta_plan
    from src.match_engine.world_model.active_probe import (
        validate_llm_active_probe,
    )

    active_probe = validate_llm_active_probe(
        plan.get("world_model_active_probe")
    )
    if active_probe is not None:
        out["world_model_active_probe"] = active_probe
    from src.match_engine.world_model.active_probe_portfolio import (
        validate_llm_active_probe_portfolio,
    )

    active_probe_portfolio = validate_llm_active_probe_portfolio(
        plan.get("world_model_active_probe_portfolio")
    )
    if active_probe_portfolio is not None:
        out["world_model_active_probe_portfolio"] = active_probe_portfolio
    from src.match_engine.world_model.opponent_information_adaptation import (
        validate_llm_opponent_information_adaptation,
    )

    information_adaptation = validate_llm_opponent_information_adaptation(
        plan.get("opponent_information_adaptation")
    )
    if information_adaptation is not None:
        out["opponent_information_adaptation"] = information_adaptation
    return out


def validate_player_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = sanitize_plan(plan)
    eb = plan.get("emotion_bias") or {}
    if not isinstance(eb, dict):
        eb = {}
    return {
        "narrative": str(plan.get("narrative", ""))[:280],
        "confidence": _clip01(plan.get("confidence", 0.5)),
        "emotion_bias": {
            k: _clip_delta(eb.get(k, 0), 0.25)
            for k in ("pride", "anger", "fear", "determination")
        },
        "risk_delta": _clip_delta(plan.get("risk_delta", 0), 0.12),
    }


def validate_referee_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = sanitize_plan(plan)
    return {
        "reasoning": str(plan.get("reasoning", ""))[:400],
        "strictness_delta": _clip_delta(plan.get("strictness_delta", 0), 0.12),
        "bias_delta": _clip_delta(plan.get("bias_delta", 0), 0.08),
        "var_recommendation": str(plan.get("var_recommendation", "none"))[:40],
        "calm_delta": _clip_delta(plan.get("calm_delta", 0), 0.15),
    }


def validate_assistant_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = sanitize_plan(plan)
    return {
        "offside_strictness_delta": _clip_delta(plan.get("offside_strictness_delta", 0), 0.1),
        "recommend_card_review": bool(plan.get("recommend_card_review", False)),
        "signal": str(plan.get("signal", ""))[:120],
    }


def validate_crowd_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = sanitize_plan(plan)
    return {
        "chant_narrative": str(plan.get("chant_narrative", ""))[:200],
        "psi_pulse": _clip_delta(plan.get("psi_pulse", 0), 0.35),
        "home_boost_delta": _clip_delta(plan.get("home_boost_delta", 0), 0.15),
    }
