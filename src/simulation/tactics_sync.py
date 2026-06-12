"""Sync LLM / JSON coach decisions to full 21-dim tactical vector."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.tactical_catalog import TACTICAL_PRESETS, infer_archetype_from_text
from src.match_engine.tactical_profile import (
    build_tactical_vector_for_agent,
    legacy_controls_from_vector,
    sync_agent_controls_from_vector,
)

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def parse_coach_tactics_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"reasoning": raw}
    return {}


def apply_coach_tactics_from_llm(agent: "SocietyAgent", tactics_payload: Any) -> Dict[str, Any]:
    """
    Apply formation, style text, controls, and optional tactical_preset to agent + full vector.
    Returns parsed dict for logging.
    """
    data = parse_coach_tactics_payload(tactics_payload)
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
        from src.match_engine.tactical_catalog import resolve_tactical_preset

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
    agent.semantic_memory["tactical_vector_keys"] = list(tac.keys())[:6]
    return data
