"""Environment-driven config for in-match cognitive (System 2) layer."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Mapping, Tuple

from src.simulation.runtime import environment_snapshot, env_bool, env_float


def _parse_tier_caps(raw: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if not raw.strip():
        return default
    parts = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    return tuple(parts) if parts else default


def _parse_outcome_horizons(
    raw: str,
    default: Tuple[float, ...],
) -> Tuple[float, ...]:
    if not raw.strip():
        return default
    values = []
    for item in raw.split(","):
        try:
            value = float(item.strip())
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0.0:
            values.append(value)
    if not values:
        return default
    return tuple(sorted(set([0.0, *values])))[:6]


@dataclass
class CognitiveMatchConfig:
    enabled: bool = False
    sync_llm: bool = False
    cache_dir: str = ""
    salience_center: float = 0.55
    cooldown_sec: float = 180.0
    crowd_numeric: bool = True
    world_model_action_bridge: bool = True
    world_model_action_bias_max: float = 0.35
    world_model_action_control_rate: float = 0.20
    world_model_action_min_arm_samples: int = 2
    world_model_residual_min_samples: int = 6
    world_model_outcome_horizons_s: Tuple[float, ...] = (0.0, 60.0, 180.0)
    world_model_active_learning: bool = True
    world_model_exploration_budget: float = 0.25
    world_model_exploration_max_regret: float = 0.08
    world_model_exploration_min_information: float = 0.45
    world_model_exploration_strength_scale: float = 0.50
    world_model_trajectory_branch_budget: int = 48
    world_model_contrastive_repair: bool = False
    world_model_contrastive_repair_path_budget: int = 128
    world_model_two_stage_deliberation: bool = True
    # coach, referee, player_per_team, assistant_total, crowd
    tier_caps: Tuple[int, int, int, int, int] = (6, 4, 4, 2, 3)
    half_time_sec: float = 45.0 * 60.0
    second_half_sec: float = 90.0 * 60.0
    half_time_window_sec: float = 120.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, str], base_dir: str = "") -> "CognitiveMatchConfig":
        cache = values.get("MATCH_COGNITIVE_CACHE", "").strip()
        if not cache and base_dir:
            cache = os.path.join(base_dir, "data", "cache", "cognitive")
        return cls(
            enabled=env_bool(values, "MATCH_COGNITIVE", False),
            sync_llm=env_bool(values, "MATCH_COGNITIVE_SYNC", False),
            cache_dir=cache,
            salience_center=env_float(values, "MATCH_COGNITIVE_S0", 0.55),
            cooldown_sec=env_float(values, "MATCH_COGNITIVE_COOLDOWN", 180.0),
            crowd_numeric=env_bool(values, "MATCH_CROWD_LLM_NUMERIC", True),
            world_model_action_bridge=env_bool(
                values, "MATCH_WM_LLM_ACTION_BRIDGE", True,
            ),
            world_model_action_bias_max=max(0.0, min(0.5, env_float(
                values, "MATCH_WM_LLM_ACTION_BIAS_MAX", 0.35,
            ))),
            world_model_action_control_rate=max(0.0, min(0.5, env_float(
                values, "MATCH_WM_LLM_ACTION_CONTROL_RATE", 0.20,
            ))),
            world_model_action_min_arm_samples=max(
                2,
                int(env_float(
                    values, "MATCH_WM_LLM_ACTION_MIN_ARM_SAMPLES", 2.0,
                )),
            ),
            world_model_residual_min_samples=max(
                2,
                int(env_float(
                    values, "MATCH_WM_LLM_RESIDUAL_MIN_SAMPLES", 6.0,
                )),
            ),
            world_model_outcome_horizons_s=_parse_outcome_horizons(
                values.get("MATCH_WM_LLM_OUTCOME_HORIZONS", ""),
                (0.0, 60.0, 180.0),
            ),
            world_model_active_learning=env_bool(
                values, "MATCH_WM_ACTIVE_LEARNING", True,
            ),
            world_model_exploration_budget=max(0.0, min(0.5, env_float(
                values, "MATCH_WM_EXPLORATION_BUDGET", 0.25,
            ))),
            world_model_exploration_max_regret=max(0.0, min(0.30, env_float(
                values, "MATCH_WM_EXPLORATION_MAX_REGRET", 0.08,
            ))),
            world_model_exploration_min_information=max(
                0.0, min(1.0, env_float(
                    values, "MATCH_WM_EXPLORATION_MIN_INFORMATION", 0.45,
                )),
            ),
            world_model_exploration_strength_scale=max(
                0.0, min(1.0, env_float(
                    values, "MATCH_WM_EXPLORATION_STRENGTH_SCALE", 0.50,
                )),
            ),
            world_model_trajectory_branch_budget=max(
                16,
                min(112, int(env_float(
                    values, "MATCH_WM_TRAJECTORY_BRANCH_BUDGET", 48.0,
                ))),
            ),
            world_model_contrastive_repair=env_bool(
                values, "MATCH_WM_LLM_CONTRASTIVE_REPAIR", False,
            ),
            world_model_contrastive_repair_path_budget=max(
                16,
                min(128, int(env_float(
                    values,
                    "MATCH_WM_LLM_CONTRASTIVE_REPAIR_PATH_BUDGET",
                    128.0,
                ))),
            ),
            world_model_two_stage_deliberation=env_bool(
                values, "MATCH_WM_LLM_TWO_STAGE_DELIBERATION", True,
            ),
            tier_caps=_parse_tier_caps(
                values.get("MATCH_COGNITIVE_MAX_PER_TIER", ""),
                (6, 4, 4, 2, 3),
            ),
        )

    @classmethod
    def from_env(cls, base_dir: str = "") -> "CognitiveMatchConfig":
        return cls.from_mapping(environment_snapshot(), base_dir)


def cognitive_enabled() -> bool:
    return CognitiveMatchConfig.from_env().enabled
