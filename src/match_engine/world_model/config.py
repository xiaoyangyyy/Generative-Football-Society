"""Environment flags and paths for the latent world model."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class WorldModelConfig:
    latent_dim: int = 96
    hidden_dim: int = 256
    imagination_steps: int = 3
    planner_blend: float = 0.45
    shot_planner_blend: float = 0.40
    grid_gx: int = 8
    grid_gy: int = 5
    transition_type: str = "gru"  # gru | transformer
    transformer_layers: int = 2
    transformer_heads: int = 4

    @classmethod
    def from_env(cls) -> "WorldModelConfig":
        return cls(
            latent_dim=_env_int("MATCH_WM_LATENT_DIM", 96),
            hidden_dim=_env_int("MATCH_WM_HIDDEN_DIM", 256),
            imagination_steps=_env_int("MATCH_WM_IMAGINATION_STEPS", 3),
            planner_blend=_env_float("MATCH_WM_PLANNER_BLEND", 0.45),
            shot_planner_blend=_env_float("MATCH_WM_SHOT_BLEND", 0.40),
            grid_gx=_env_int("MATCH_WM_GRID_GX", 8),
            grid_gy=_env_int("MATCH_WM_GRID_GY", 5),
            transition_type=os.environ.get("MATCH_WM_TRANSITION", "gru").strip().lower() or "gru",
            transformer_layers=_env_int("MATCH_WM_TF_LAYERS", 2),
            transformer_heads=_env_int("MATCH_WM_TF_HEADS", 4),
        )


def world_model_enabled() -> bool:
    return _env_bool("MATCH_WORLD_MODEL", False)


def world_model_plan_enabled() -> bool:
    if not world_model_enabled():
        return False
    raw = os.environ.get("MATCH_WM_PLAN", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def world_model_record_enabled() -> bool:
    return _env_bool("MATCH_WM_RECORD", False)


def default_checkpoint_path(base_dir: str) -> str:
    custom = os.environ.get("MATCH_WM_CHECKPOINT", "").strip()
    if custom:
        return custom
    return os.path.join(base_dir, "data", "world_model", "latent_wm.pt")


def default_trace_dir(base_dir: str) -> str:
    custom = os.environ.get("MATCH_WM_TRACE_DIR", "").strip()
    if custom:
        return custom
    return os.path.join(base_dir, "data", "world_model", "traces")
