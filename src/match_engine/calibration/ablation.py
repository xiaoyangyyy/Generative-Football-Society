"""Ablation presets and calibration pipeline environment switches."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.match_engine.micro_config import MicroMatchConfig

BALANCED_TACTICS = {
    "pressing_intensity": 0.5,
    "risk_budget": 0.5,
    "line_height": 0.5,
    "rotation_aggressiveness": 0.5,
}

_CALIBRATION_ENV_KEYS = (
    "CALIBRATION_MODE",
    "CALIBRATION_NO_BLEND",
    "CALIBRATION_LOGIT_BLEND",
    "CALIBRATION_INTERCEPT_RISK_SCALE",
    "CALIBRATION_ROSTER_STATUS",
    "MATCH_COGNITIVE",
    "MATCH_WORLD_MODEL",
    "MATCH_WM_PLAN",
    "MATCH_WM_OUTCOME_ALIGNED_POLICY",
    "MATCH_WM_CONTROL_SCOPE",
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    env: dict[str, str] = field(default_factory=dict)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    tactical_override: bool = False


PIPELINE_PRESETS: dict[str, AblationSpec] = {
    "M0": AblationSpec(
        name="M0",
        description="Micro physics baseline: blend on, WM off, cognitive off.",
        env={
            "CALIBRATION_MODE": "1",
            "MATCH_COGNITIVE": "0",
            "MATCH_WORLD_MODEL": "0",
            "MATCH_WM_PLAN": "0",
            "MATCH_WM_OUTCOME_ALIGNED_POLICY": "0",
            "MATCH_WM_CONTROL_SCOPE": "none",
        },
    ),
    "M1": AblationSpec(
        name="M1",
        description="M0 + world model planner (additive layer).",
        env={
            "CALIBRATION_MODE": "1",
            "MATCH_COGNITIVE": "0",
            "MATCH_WORLD_MODEL": "1",
            "MATCH_WM_PLAN": "1",
            "MATCH_WM_OUTCOME_ALIGNED_POLICY": "0",
            "MATCH_WM_CONTROL_SCOPE": "both",
        },
    ),
    "M2": AblationSpec(
        name="M2",
        description=(
            "M0 + outcome-aligned, action-specific validated world-model policy."
        ),
        env={
            "CALIBRATION_MODE": "1",
            "MATCH_COGNITIVE": "0",
            "MATCH_WORLD_MODEL": "1",
            "MATCH_WM_PLAN": "1",
            "MATCH_WM_OUTCOME_ALIGNED_POLICY": "1",
            "MATCH_WM_CONTROL_SCOPE": "both",
        },
    ),
    "C1": AblationSpec(
        name="C1",
        description="M0 + in-match cognitive layer (rule/LLM fallback).",
        env={
            "CALIBRATION_MODE": "1",
            "MATCH_COGNITIVE": "1",
            "MATCH_COGNITIVE_SYNC": "0",
            "MATCH_WORLD_MODEL": "0",
            "MATCH_WM_PLAN": "0",
            "MATCH_WM_OUTCOME_ALIGNED_POLICY": "0",
            "MATCH_WM_CONTROL_SCOPE": "none",
        },
    ),
    "T0": AblationSpec(
        name="T0",
        description="Roster-driven eff_status; cognitive and WM off.",
        env={
            "CALIBRATION_MODE": "1",
            "MATCH_COGNITIVE": "0",
            "MATCH_WORLD_MODEL": "0",
            "MATCH_WM_PLAN": "0",
            "MATCH_WM_OUTCOME_ALIGNED_POLICY": "0",
            "MATCH_WM_CONTROL_SCOPE": "none",
            "CALIBRATION_ROSTER_STATUS": "1",
        },
    ),
}

# Subtractive ablations vs M0 (module was ON in baseline, turn OFF).
SUBTRACTIVE_ABLATIONS: dict[str, AblationSpec] = {
    "no_affective": AblationSpec(
        name="no_affective",
        description="Freeze emotion→modulator coupling at neutral defaults.",
        env=dict(PIPELINE_PRESETS["M0"].env),
        config_overrides={"disable_affective_dynamics": True},
    ),
    "no_spatial": AblationSpec(
        name="no_spatial",
        description="Disable spatial field and spatial intelligence.",
        env=dict(PIPELINE_PRESETS["M0"].env),
        config_overrides={"enable_spatial": False},
    ),
    "no_blend": AblationSpec(
        name="no_blend",
        description="StatsBomb logit blend w=0 (pure physics pass completion).",
        env={
            **PIPELINE_PRESETS["M0"].env,
            "CALIBRATION_NO_BLEND": "1",
        },
    ),
    "no_discipline_tick": AblationSpec(
        name="no_discipline_tick",
        description="Calendar fouls/cards instead of tick discipline model.",
        env=dict(PIPELINE_PRESETS["M0"].env),
        config_overrides={"discipline_tick_fouls": False},
    ),
    "no_tactical_bias": AblationSpec(
        name="no_tactical_bias",
        description="Pin both teams to balanced tactical vector.",
        env=dict(PIPELINE_PRESETS["M0"].env),
        tactical_override=True,
    ),
}

# Additive layers vs M0 (module OFF in baseline, turn ON).
ADDITIVE_LAYERS: dict[str, AblationSpec] = {
    "M1": PIPELINE_PRESETS["M1"],
    "C1": PIPELINE_PRESETS["C1"],
}

ABLATION_PRESETS: dict[str, AblationSpec] = {
    **SUBTRACTIVE_ABLATIONS,
    **ADDITIVE_LAYERS,
}


def calibration_env() -> dict[str, str]:
    """Default env for reproducible calibration runs."""
    return dict(PIPELINE_PRESETS["M0"].env)


def apply_ablation(spec: AblationSpec, cfg: MicroMatchConfig | None = None) -> MicroMatchConfig:
    cfg = cfg or MicroMatchConfig()
    for key, val in spec.config_overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    cfg.use_micro_goals = True
    return cfg


def roster_status_mode() -> bool:
    return os.environ.get("CALIBRATION_ROSTER_STATUS", "").strip().lower() in ("1", "true", "yes")


def calibration_mode_active() -> bool:
    return os.environ.get("CALIBRATION_MODE", "").strip().lower() in ("1", "true", "yes")


def _clear_pass_calibration_cache() -> None:
    try:
        from src.match_engine import pass_calibration

        pass_calibration.load_pass_calibration.cache_clear()
    except Exception:
        pass


@contextmanager
def ablation_context(spec: AblationSpec) -> Iterator[MicroMatchConfig]:
    """Temporarily apply ablation env vars; restore on exit."""
    from src.match_engine.calibration.narrative_isolation import (
        apply_benchmark_isolation,
        restore_env as restore_benchmark_env,
    )

    pipeline = spec.name if spec.name in PIPELINE_PRESETS else "M0"
    saved_cal = {k: os.environ.get(k) for k in _CALIBRATION_ENV_KEYS}
    saved_bench = apply_benchmark_isolation(pipeline)
    try:
        for k in _CALIBRATION_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(spec.env)
        _clear_pass_calibration_cache()
        cfg = apply_ablation(spec)
        yield cfg
    finally:
        restore_benchmark_env(saved_bench)
        for k in _CALIBRATION_ENV_KEYS:
            prev = saved_cal.get(k)
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
        _clear_pass_calibration_cache()


def tactical_overrides_for(spec: AblationSpec) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if not spec.tactical_override:
        return None, None
    return dict(BALANCED_TACTICS), dict(BALANCED_TACTICS)
