"""Learned latent spatial world model for micro-match imagination and planning."""

from src.match_engine.world_model.config import (
    WorldModelConfig,
    world_model_enabled,
    world_model_plan_enabled,
    world_model_record_enabled,
)
from src.match_engine.world_model.observation import OBS_DIM, encode_observation
from src.match_engine.world_model.action_codec import ACTION_DIM, encode_pass_candidate, zero_action

from src.match_engine.world_model.inference import WorldModelRuntime

__all__ = [
    "WorldModelConfig",
    "world_model_enabled",
    "world_model_plan_enabled",
    "world_model_record_enabled",
    "OBS_DIM",
    "ACTION_DIM",
    "encode_observation",
    "encode_pass_candidate",
    "zero_action",
    "WorldModelRuntime",
]
