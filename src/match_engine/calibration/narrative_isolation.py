"""Phase 5 — isolate L0 calibration from narrative layers (LLM / WM / cognitive)."""

from __future__ import annotations

import os

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, calibration_mode_active

_L0_PIPELINES = frozenset({"M0", "T0"})
_NARRATIVE_LAYER_PIPELINES = frozenset({"M1", "C1"})

_CALIBRATION_BENCHMARK_ENV = (
    "CALIBRATION_MODE",
    "MATCH_MICRO",
    "MATCH_MICRO_SCORE",
    "MATCH_COGNITIVE",
    "MATCH_WORLD_MODEL",
    "MATCH_COGNITIVE_SYNC",
    "MATCH_WM_PLAN",
    "MATCH_WM_OUTCOME_ALIGNED_POLICY",
    "MATCH_WM_CONTROL_SCOPE",
    "MATCH_LEGACY_POISSON",
    "XG_SUPPLEMENT",
    "MATCH_SCHEDULED_SHOTS",
)


def narrative_layers_enabled() -> bool:
    """Full tournament narrative (LLM coach/media) — off during calibration benchmarks."""
    if calibration_mode_active():
        return False
    return os.environ.get("NARRATIVE_MODE", "1").strip().lower() not in ("0", "false", "no")


def l0_calibration_pipeline(name: str) -> bool:
    return name in _L0_PIPELINES


def narrative_layer_pipeline(name: str) -> bool:
    return name in _NARRATIVE_LAYER_PIPELINES


def expected_layer_flags(pipeline: str) -> dict[str, str]:
    """Canonical env for a calibration pipeline (M0 = physics only)."""
    spec = PIPELINE_PRESETS.get(pipeline)
    if spec is None:
        raise KeyError(f"Unknown pipeline: {pipeline}")
    base = {
        "CALIBRATION_MODE": "1",
        "MATCH_MICRO": "1",
        "MATCH_MICRO_SCORE": "1",
        "MATCH_LEGACY_POISSON": "0",
        "XG_SUPPLEMENT": "0",
        "MATCH_SCHEDULED_SHOTS": "0",
        "MATCH_COGNITIVE": "0",
        "MATCH_WORLD_MODEL": "0",
    }
    base.update(spec.env)
    return base


def assert_isolated_env(pipeline: str, env: dict[str, str] | None = None) -> None:
    """Raise if narrative layers leak into an L0 pipeline."""
    if not l0_calibration_pipeline(pipeline):
        return
    env = env or {k: os.environ.get(k, "") or "" for k in _CALIBRATION_BENCHMARK_ENV}
    leaks: list[str] = []
    if env.get("MATCH_COGNITIVE", "0") in ("1", "true", "yes"):
        leaks.append("MATCH_COGNITIVE")
    if env.get("MATCH_WORLD_MODEL", "0") in ("1", "true", "yes"):
        leaks.append("MATCH_WORLD_MODEL")
    if leaks:
        raise RuntimeError(
            f"L0 pipeline {pipeline} must not enable narrative layers; found: {', '.join(leaks)}"
        )


def apply_benchmark_isolation(pipeline: str = "M0") -> dict[str, str | None]:
    """
    Set process env for isolated micro benchmark runs.
    Returns previous values for optional restore.
    """
    expected = expected_layer_flags(pipeline)
    saved: dict[str, str | None] = {}
    for key in _CALIBRATION_BENCHMARK_ENV:
        saved[key] = os.environ.get(key)
        os.environ.pop(key, None)
    os.environ.update(expected)
    assert_isolated_env(pipeline)
    return saved


def resolve_tournament_llm():
    """Real LLM for full narrative; deterministic stub when calibration / NARRATIVE_MODE=0."""
    from src.simulation.llm_engine import NullSimulationLLM, SimulationLLM

    if narrative_layers_enabled():
        return SimulationLLM()
    return NullSimulationLLM()


def restore_env(saved: dict[str, str | None]) -> None:
    for key, prev in saved.items():
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev
