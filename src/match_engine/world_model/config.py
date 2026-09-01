"""Environment flags and paths for the latent world model."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping

from src.simulation.runtime import environment_snapshot, env_bool, env_float, env_int


_PLAN_CLOCK_SECONDS: ContextVar[float | None] = ContextVar(
    "world_model_plan_clock_seconds", default=None,
)


@dataclass
class WorldModelConfig:
    latent_dim: int = 96
    hidden_dim: int = 256
    imagination_steps: int = 1
    planner_blend: float = 0.30
    shot_planner_blend: float = 0.25
    cross_planner_blend: float = 0.20
    residual_scale: float = 0.25
    rollout_residual_blend: float = 1.0
    legacy_quality: float = 0.10
    min_planner_quality: float = 0.15
    ensemble_size: int = 3
    transition_ensemble_size: int = 3
    grid_gx: int = 8
    grid_gy: int = 5
    transition_type: str = "gru"  # gru | transformer
    transformer_layers: int = 2
    transformer_heads: int = 4

    def __post_init__(self) -> None:
        if (self.grid_gx, self.grid_gy) != (8, 5):
            raise ValueError("World-model schema v6 requires grid_gx=8 and grid_gy=5")
        if self.latent_dim < 16 or self.hidden_dim < 32:
            raise ValueError("World-model dimensions are too small for structured encoding")
        if self.ensemble_size < 2:
            raise ValueError("World-model uncertainty requires ensemble_size >= 2")
        if self.transition_ensemble_size < 2:
            raise ValueError(
                "World-model dynamics uncertainty requires "
                "transition_ensemble_size >= 2"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "WorldModelConfig":
        return cls(
            latent_dim=env_int(values, "MATCH_WM_LATENT_DIM", 96),
            hidden_dim=env_int(values, "MATCH_WM_HIDDEN_DIM", 256),
            imagination_steps=env_int(values, "MATCH_WM_IMAGINATION_STEPS", 1),
            planner_blend=env_float(values, "MATCH_WM_PLANNER_BLEND", 0.30),
            shot_planner_blend=env_float(values, "MATCH_WM_SHOT_BLEND", 0.25),
            cross_planner_blend=env_float(values, "MATCH_WM_CROSS_BLEND", 0.20),
            residual_scale=env_float(values, "MATCH_WM_RESIDUAL_SCALE", 0.25),
            rollout_residual_blend=env_float(
                values, "MATCH_WM_ROLLOUT_RESIDUAL_BLEND", 1.0,
            ),
            legacy_quality=env_float(values, "MATCH_WM_LEGACY_QUALITY", 0.10),
            min_planner_quality=env_float(values, "MATCH_WM_MIN_QUALITY", 0.15),
            ensemble_size=env_int(values, "MATCH_WM_ENSEMBLE_SIZE", 3),
            transition_ensemble_size=env_int(
                values, "MATCH_WM_TRANSITION_ENSEMBLE_SIZE", 3,
            ),
            grid_gx=env_int(values, "MATCH_WM_GRID_GX", 8),
            grid_gy=env_int(values, "MATCH_WM_GRID_GY", 5),
            transition_type=values.get("MATCH_WM_TRANSITION", "gru").strip().lower() or "gru",
            transformer_layers=env_int(values, "MATCH_WM_TF_LAYERS", 2),
            transformer_heads=env_int(values, "MATCH_WM_TF_HEADS", 4),
        )

    @classmethod
    def from_env(cls) -> "WorldModelConfig":
        return cls.from_mapping(environment_snapshot())


def world_model_enabled() -> bool:
    return env_bool(environment_snapshot(), "MATCH_WORLD_MODEL", False)


def world_model_required() -> bool:
    """Whether an enabled world model must load instead of degrading to fallback."""
    return env_bool(environment_snapshot(), "MATCH_WORLD_MODEL_REQUIRED", False)


def world_model_plan_start_sec() -> float:
    raw = environment_snapshot().get("MATCH_WM_PLAN_START_SEC", "0").strip()
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("MATCH_WM_PLAN_START_SEC must be a finite number") from exc
    if not 0.0 <= value <= 5400.0:
        raise ValueError("MATCH_WM_PLAN_START_SEC must be between 0 and 5400")
    return value


def world_model_branch_at_sec() -> float | None:
    raw = environment_snapshot().get("MATCH_WM_BRANCH_AT_SEC", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("MATCH_WM_BRANCH_AT_SEC must be a finite number") from exc
    if not 0.0 <= value <= 5400.0:
        raise ValueError("MATCH_WM_BRANCH_AT_SEC must be between 0 and 5400")
    return value


def world_model_plan_enabled(state: Any | None = None) -> bool:
    if not world_model_enabled():
        return False
    raw = environment_snapshot().get("MATCH_WM_PLAN", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    clock = (
        float(getattr(state, "clock_seconds", 0.0))
        if state is not None else _PLAN_CLOCK_SECONDS.get()
    )
    if clock is not None:
        return clock >= world_model_plan_start_sec()
    return True


@contextmanager
def world_model_plan_clock(clock_seconds: float):
    token = _PLAN_CLOCK_SECONDS.set(float(clock_seconds))
    try:
        yield
    finally:
        _PLAN_CLOCK_SECONDS.reset(token)


def world_model_record_enabled() -> bool:
    return env_bool(environment_snapshot(), "MATCH_WM_RECORD", False)


def default_checkpoint_path(base_dir: str) -> str:
    custom = environment_snapshot().get("MATCH_WM_CHECKPOINT", "").strip()
    if custom:
        return custom
    return os.path.join(base_dir, "data", "world_model", "latent_wm.pt")


def default_shot_head_path(base_dir: str) -> str:
    custom = environment_snapshot().get("MATCH_WM_SHOT_HEAD", "").strip()
    if custom:
        return custom
    return ""


def default_trace_dir(base_dir: str) -> str:
    custom = environment_snapshot().get("MATCH_WM_TRACE_DIR", "").strip()
    if custom:
        return custom
    return os.path.join(base_dir, "data", "world_model", "traces")
