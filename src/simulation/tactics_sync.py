"""Sync LLM / JSON coach decisions to full 21-dim tactical vector."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.tactical_catalog import (
    TACTICAL_PRESETS,
    infer_archetype_from_text,
    resolve_tactical_preset,
)
from src.match_engine.tactical_profile import (
    build_tactical_vector_for_agent,
    legacy_controls_from_vector,
    sync_agent_controls_from_vector,
)

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def _finite_metric(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def parse_coach_tactics_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"reasoning": raw}
    return {}


def reconcile_world_model_tactical_choice(
    tactics: Dict[str, Any],
    decision_support: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Constrain an evidence-backed LLM choice to evaluated tactical candidates."""
    data = dict(tactics or {})
    packet = decision_support or {}
    available = bool(packet.get("available"))
    candidates = {
        resolve_tactical_preset(item.get("tactical_preset"))
        for item in packet.get("candidates", [])
        if isinstance(item, dict) and item.get("tactical_preset")
    }
    recommended = resolve_tactical_preset(
        packet.get("recommended_tactical_preset")
    )
    requested = resolve_tactical_preset(data.get("tactical_preset"))
    selected = requested
    constrained = False
    if available and candidates and requested not in candidates:
        selected = recommended if recommended in candidates else sorted(candidates)[0]
        constrained = True
    if available and candidates:
        data["tactical_preset"] = selected
    candidate_evidence = {
        resolve_tactical_preset(item.get("tactical_preset")): item
        for item in packet.get("candidates", [])
        if isinstance(item, dict) and item.get("tactical_preset")
    }

    def evidence_for(name: str) -> Dict[str, float]:
        item = candidate_evidence.get(name, {})
        return {
            key: _finite_metric(item[key])
            for key in (
                "risk_adjusted_value",
                "effective_confidence",
                "uncertainty",
                "fatigue_cost_proxy",
                "structural_risk_proxy",
            )
            if key in item
        }

    ranked = sorted(
        candidate_evidence.values(),
        key=lambda item: _finite_metric(item.get("risk_adjusted_value"), -1.0),
        reverse=True,
    )
    recommendation_margin = 0.0
    if len(ranked) >= 2:
        recommendation_margin = float(
            _finite_metric(ranked[0].get("risk_adjusted_value"))
            - _finite_metric(ranked[1].get("risk_adjusted_value"))
        )
    rationale = str(data.get("world_model_rationale", "") or "")[:300]
    if rationale:
        data["world_model_rationale"] = rationale
    reliability = packet.get("fusion_reliability") or {}
    strategy_memory = packet.get("strategy_memory") or {}
    data["world_model_audit"] = {
        "evidence_available": available,
        "evidence_reason": str(packet.get("reason", "not_provided")),
        "recommended_tactical_preset": recommended if available else "none",
        "requested_tactical_preset": requested,
        "selected_tactical_preset": selected,
        "disagreed_with_recommendation": bool(
            available and selected != recommended
        ),
        "selection_constrained": constrained,
        "packet_version": int(_finite_metric(packet.get("version"))),
        "evaluation_scope": str(packet.get("evaluation_scope", "")),
        "horizon_s": _finite_metric(packet.get("horizon_s")),
        "observation_coverage": _finite_metric(
            packet.get("observation_coverage")
        ),
        "recommendation_confidence": _finite_metric(
            packet.get("recommendation_confidence")
        ),
        "recommendation_margin": recommendation_margin,
        "recommended_evidence": evidence_for(recommended),
        "selected_evidence": evidence_for(selected),
        "balanced_evidence": evidence_for("balanced"),
        "rationale": rationale,
        "historical_evidence_tier": str(
            reliability.get("evidence_tier", "not_available")
        ),
        "historical_guidance": str(reliability.get("guidance", "")),
        "adjusted_recommendation_trust": _finite_metric(
            reliability.get("adjusted_recommendation_trust")
        ),
        "matched_seed_samples": int(_finite_metric(
            reliability.get("matched_seed_samples")
        )),
        "historical_match_records": int(_finite_metric(
            reliability.get("historical_match_records")
        )),
        "opponent_specific_history": int(_finite_metric(
            strategy_memory.get("opponent_specific_matches")
        )),
    }
    return data


def apply_coach_tactics_from_llm(
    agent: "SocietyAgent",
    tactics_payload: Any,
    *,
    decision_support: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply formation, style text, controls, and optional tactical_preset to agent + full vector.
    Returns parsed dict for logging.
    """
    data = reconcile_world_model_tactical_choice(
        parse_coach_tactics_payload(tactics_payload), decision_support,
    )
    if data.get("formation"):
        agent.formation = str(data["formation"]).split(" - ")[0].strip()
    style = str(data.get("style", ""))
    if style:
        agent.style_desc = style
        agent.style_archetype = infer_archetype_from_text(style, agent.formation)

    controls = dict(data.get("controls") or {})
    if controls:
        agent.set_tactical_controls(controls)

    preset = str(data.get("tactical_preset", "") or "")
    cp = getattr(agent, "coach_profile", None)
    if cp is not None and preset:
        resolved = resolve_tactical_preset(preset)
        cp.preferred_preset = resolved
        agent._tactical_preset_locked = True
        from src.data_engine.entity_dynamics import preset_affinities

        aff = preset_affinities(cp.mental, agent.style_desc)
        aff[preset] = aff.get(preset, 0) + 0.35
        s = sum(aff.values())
        cp.preset_affinities = {k: v / s for k, v in aff.items()}

    hints = dict(data.get("tactical_hints") or {})
    from src.match_engine.tactical_profile import compose_llm_tactical_vector

    tac = (
        compose_llm_tactical_vector(agent, hints=hints)
        if getattr(agent, "_tactical_preset_locked", False)
        else build_tactical_vector_for_agent(agent)
    )
    if hints and not getattr(agent, "_tactical_preset_locked", False):
        for k, v in hints.items():
            if k in tac:
                tac[k] = float(np.clip(0.55 * tac[k] + 0.45 * float(v), 0.0, 1.0))
    sync_agent_controls_from_vector(agent, tac)
    agent.tactical_vector = tac
    agent._prematch_world_model_audit = dict(data["world_model_audit"])
    agent.semantic_memory["tactical_vector_keys"] = list(tac.keys())[:6]
    return data
