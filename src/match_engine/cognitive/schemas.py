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
