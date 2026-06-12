"""Preset-coherent LLM tactics and semantic shot bias."""

from __future__ import annotations

from src.match_engine.tactical_catalog import TACTICAL_PRESETS
from src.match_engine.tactical_engine import TacticalMicroEngine
from src.match_engine.tactical_profile import compose_llm_tactical_vector
from src.match_engine.micro_config import MicroMatchConfig


def test_park_the_bus_lowers_shot_bias_vs_gegenpress():
    eng = TacticalMicroEngine(MicroMatchConfig())
    bus = TACTICAL_PRESETS["park_the_bus"]
    press = TACTICAL_PRESETS["gegenpress"]
    assert eng.shot_bias(bus) < eng.shot_bias(press)
    assert eng.shot_bias(bus) < 1.0


def test_low_block_preset_defensive_not_counter_vertical():
    eng = TacticalMicroEngine(MicroMatchConfig())
    lb = TACTICAL_PRESETS["low_block"]
    lbc = TACTICAL_PRESETS["low_block_counter"]
    assert eng.shot_bias(lb) < eng.shot_bias(lbc)
    assert lb["verticality"] < lbc["verticality"]


def test_resolve_low_block_uses_defensive_preset():
    from src.match_engine.tactical_catalog import resolve_tactical_preset

    assert resolve_tactical_preset("low_block") == "low_block"


def test_locked_llm_preset_alias_no_recursion():
    class _Coach:
        preferred_preset = "counter"  # LLM shorthand — not a TACTICAL_PRESETS key
        mental = {"tactical_knowledge": 0.6}

    class _Agent:
        _tactical_preset_locked = True
        coach_profile = _Coach()
        formation = "4-2-3-1"
        style_desc = "counter"
        style_archetype = "counter_attack"
        tactical_controls = {"pressing_intensity": 0.5, "risk_budget": 0.5, "line_height": 0.5, "rotation_aggressiveness": 0.5}

    tac = compose_llm_tactical_vector(_Agent())
    assert _Agent.coach_profile.preferred_preset == "counter_attack"
    assert tac["counter_attack"] > 0.5


def test_build_vector_locked_invalid_preset_uses_compose_not_recursion():
    from src.match_engine.tactical_profile import build_tactical_vector_for_agent

    class _Coach:
        preferred_preset = "possession"
        mental = {"tactical_knowledge": 0.5}

    class _Agent:
        _tactical_preset_locked = True
        coach_profile = _Coach()
        formation = "4-3-3"
        style_desc = ""
        style_archetype = "balanced"
        tactical_controls = {"pressing_intensity": 0.5, "risk_budget": 0.5, "line_height": 0.5, "rotation_aggressiveness": 0.5}

    tac = build_tactical_vector_for_agent(_Agent())
    assert _Agent.coach_profile.preferred_preset == "possession_control"
    assert len(tac) > 0


def test_llm_preset_not_overwritten_by_extreme_controls():
    class _Coach:
        preferred_preset = "park_the_bus"
        mental = {"tactical_knowledge": 0.6}

    class _Agent:
        _tactical_preset_locked = True
        coach_profile = _Coach()
        formation = "5-4-1"
        style_desc = "low block"
        style_archetype = "park_the_bus"
        tactical_controls = {
            "pressing_intensity": 0.85,
            "risk_budget": 0.75,
            "line_height": 0.80,
            "rotation_aggressiveness": 0.70,
        }

    tac = compose_llm_tactical_vector(_Agent())
    assert tac["low_block"] > 0.65
    assert tac["pressing_intensity"] < 0.55
    assert tac["risk_budget"] < 0.45
