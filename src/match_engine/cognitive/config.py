"""Environment-driven config for in-match cognitive (System 2) layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Tuple

from src.simulation.runtime import environment_snapshot, env_bool, env_float


def _parse_tier_caps(raw: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if not raw.strip():
        return default
    parts = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    return tuple(parts) if parts else default


@dataclass
class CognitiveMatchConfig:
    enabled: bool = False
    sync_llm: bool = False
    cache_dir: str = ""
    salience_center: float = 0.55
    cooldown_sec: float = 180.0
    crowd_numeric: bool = True
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
