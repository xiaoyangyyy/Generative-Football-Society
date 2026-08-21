"""Build full tactical vectors from agent archetype + coach knobs."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.tactical_catalog import (
    TACTICAL_KEYS,
    TACTICAL_PRESETS,
    infer_archetype_from_text,
    normalize_formation_key,
    resolve_tactical_preset,
)

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def blend_vectors(base: Dict[str, float], override: Dict[str, float], weight: float = 0.55) -> Dict[str, float]:
    w = _clip01(weight)
    out = dict(base)
    for k in TACTICAL_KEYS:
        if k in override:
            out[k] = _clip01((1.0 - w) * base.get(k, 0.5) + w * float(override[k]))
    return out


def _control_blend_weight(agent: "SocietyAgent") -> float:
    """LLM knob influence 鈥?rises with coach tactical knowledge, preset stays primary."""
    cp = getattr(agent, "coach_profile", None)
    tk = float(getattr(cp, "mental", {}).get("tactical_knowledge", 0.5)) if cp else 0.5
    return _clip01(0.10 + 0.16 * tk)


def _apply_formation_adjustments(merged: Dict[str, float], agent: "SocietyAgent") -> Dict[str, float]:
    from src.match_engine.math_utils import sigmoid

    fk = normalize_formation_key(getattr(agent, "formation", ""))
    def_n = float(fk[0]) if fk and fk[0].isdigit() else 4.0
    low_block_sig = float(sigmoid(1.15 * (def_n - 5.0)))
    wide_sig = float(sigmoid(-1.1 * abs(def_n - 3.0)))
    press_sig = float(sigmoid(1.0 * (4.0 - def_n)))
    merged["low_block"] = _clip01(merged["low_block"] + 0.12 * low_block_sig)
    merged["compactness"] = _clip01(merged["compactness"] + 0.10 * low_block_sig)
    merged["wing_focus"] = _clip01(merged["wing_focus"] + 0.10 * wide_sig)
    merged["width_play"] = _clip01(merged["width_play"] + 0.08 * wide_sig)
    merged["pressing_intensity"] = _clip01(merged["pressing_intensity"] + 0.06 * press_sig)
    return {k: float(sigmoid(4.0 * merged.get(k, 0.5) - 2.0)) for k in TACTICAL_KEYS}


def compose_llm_tactical_vector(
    agent: "SocietyAgent",
    *,
    hints: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Preset-coherent vector: named preset is the manifold anchor; four LLM knobs are
    a small tangent adjustment via vector_from_controls (no raw control overwrite).
    """
    cp = getattr(agent, "coach_profile", None)
    raw_preset = str(getattr(cp, "preferred_preset", "") or "") if cp else ""
    preset = resolve_tactical_preset(raw_preset)
    if cp is not None and preset != raw_preset:
        cp.preferred_preset = preset

    return compose_locked_tactical_vector(agent, preset, hints=hints)


def compose_locked_tactical_vector(
    agent: "SocietyAgent",
    preset: str,
    *,
    hints: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compose a preset-anchored vector independently of who selected it."""
    resolved = resolve_tactical_preset(preset)
    if resolved != preset or resolved not in TACTICAL_PRESETS:
        raise ValueError(f"unsupported tactical preset: {preset}")
    base = dict(TACTICAL_PRESETS[resolved])
    controls = dict(getattr(agent, "tactical_controls", {}) or {})
    w_ctrl = _control_blend_weight(agent)
    merged = blend_vectors(base, vector_from_controls(controls), weight=w_ctrl)

    hint_weight = 0.20
    for k, v in (hints or {}).items():
        if k in merged:
            merged = blend_vectors(merged, {k: float(v)}, weight=hint_weight)

    return _apply_formation_adjustments(merged, agent)


def apply_locked_tactical_preset(
    agent: "SocietyAgent", preset: str, *, source: str,
) -> Dict[str, float]:
    """Apply a full preset vector and retain its explicit intervention source."""
    vector = compose_locked_tactical_vector(agent, preset)
    coach = getattr(agent, "coach_profile", None)
    if coach is not None:
        coach.preferred_preset = preset
    agent._tactical_preset_locked = True
    agent.style_archetype = preset
    agent.tactical_vector = vector
    controls = legacy_controls_from_vector(vector)
    if hasattr(agent, "set_tactical_controls"):
        agent.set_tactical_controls(controls)
    else:
        agent.tactical_controls.update(controls)
    memory = getattr(agent, "semantic_memory", None)
    if isinstance(memory, dict):
        memory["tactical_intervention"] = {
            "preset": preset, "source": str(source),
        }
    return vector


def vector_from_controls(controls: Dict[str, float]) -> Dict[str, float]:
    """Map four legacy knobs into full vector adjustments."""
    p = _clip01(controls.get("pressing_intensity", 0.5))
    r = _clip01(controls.get("risk_budget", 0.5))
    h = _clip01(controls.get("line_height", 0.5))
    rot = _clip01(controls.get("rotation_aggressiveness", 0.5))
    v = dict(TACTICAL_PRESETS["balanced"])
    v["pressing_intensity"] = p
    v["risk_budget"] = r
    v["line_height"] = h
    v["rotation_aggressiveness"] = rot
    v["high_press"] = _clip01(0.35 + 0.65 * p)
    v["counterpress"] = _clip01(0.30 + 0.60 * p)
    v["low_block"] = _clip01(0.85 - 0.70 * h)
    v["offside_trap"] = _clip01(0.25 + 0.65 * h)
    v["tempo"] = _clip01(0.35 + 0.45 * p + 0.25 * r)
    v["verticality"] = _clip01(0.35 + 0.55 * r)
    v["possession_orientation"] = _clip01(0.65 - 0.35 * r)
    v["width_play"] = _clip01(0.40 + 0.45 * rot)
    v["wing_focus"] = v["width_play"]
    v["overlap_fullbacks"] = _clip01(0.35 + 0.55 * rot)
    v["compactness"] = _clip01(0.75 - 0.35 * rot)
    return v


def _build_tactical_vector_base(agent: "SocietyAgent") -> Dict[str, float]:
    """Archetype / affinity path 鈥?never delegates to preset-locked compose."""
    arch = infer_archetype_from_text(
        getattr(agent, "style_desc", ""),
        getattr(agent, "formation", "4-3-3"),
    )
    from src.data_engine.entity_dynamics import blend_preset_from_affinities

    cp = getattr(agent, "coach_profile", None)
    if cp is not None and getattr(cp, "preset_affinities", None):
        base = blend_preset_from_affinities(cp.preset_affinities)
        arch = getattr(cp, "preferred_preset", "balanced")
    else:
        if cp is not None:
            preset = resolve_tactical_preset(getattr(cp, "preferred_preset", "") or "")
            if preset in TACTICAL_PRESETS:
                arch = preset
        if arch not in TACTICAL_PRESETS:
            arch = getattr(agent, "style_archetype", "balanced")
        if arch not in TACTICAL_PRESETS:
            arch = "balanced"
        base = dict(TACTICAL_PRESETS[arch])
        if cp is not None:
            w = 0.22 + 0.18 * float(getattr(cp, "mental", {}).get("tactical_knowledge", 0.5))
            coach_base = dict(TACTICAL_PRESETS.get(resolve_tactical_preset(getattr(cp, "preferred_preset", "")), base))
            base = blend_vectors(base, coach_base, weight=w)
    controls = dict(getattr(agent, "tactical_controls", {}) or {})
    merged = blend_vectors(base, vector_from_controls(controls), weight=0.48)
    return _apply_formation_adjustments(merged, agent)


def build_tactical_vector_for_agent(agent: "SocietyAgent") -> Dict[str, float]:
    if getattr(agent, "_tactical_preset_locked", False):
        return compose_llm_tactical_vector(agent)
    return _build_tactical_vector_base(agent)


def legacy_controls_from_vector(tac: Dict[str, float]) -> Dict[str, float]:
    return {
        "pressing_intensity": tac.get("pressing_intensity", 0.5),
        "risk_budget": tac.get("risk_budget", 0.5),
        "line_height": tac.get("line_height", 0.5),
        "rotation_aggressiveness": tac.get("rotation_aggressiveness", 0.5),
    }


def apply_vector_to_team_coach(team, tac: Dict[str, float]) -> None:
    team.coach.tactical_base = dict(tac)
    team.coach.tactical_current = dict(tac)


def sync_agent_controls_from_vector(agent: "SocietyAgent", tac: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    tac = tac or build_tactical_vector_for_agent(agent)
    leg = legacy_controls_from_vector(tac)
    agent.set_tactical_controls(leg)
    agent.style_archetype = infer_archetype_from_text(agent.style_desc, agent.formation)
    return tac
