from __future__ import annotations

from dataclasses import dataclass


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
    transition_type: str = "gru"
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
            raise ValueError("World-model dynamics uncertainty requires transition_ensemble_size >= 2")
