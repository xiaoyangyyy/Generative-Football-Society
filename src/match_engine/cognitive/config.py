"""Environment-driven config for in-match cognitive (System 2) layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


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
    def from_env(cls, base_dir: str = "") -> "CognitiveMatchConfig":
        cache = os.environ.get("MATCH_COGNITIVE_CACHE", "").strip()
        if not cache and base_dir:
            cache = os.path.join(base_dir, "data", "cache", "cognitive")
        return cls(
            enabled=_env_bool("MATCH_COGNITIVE", False),
            sync_llm=_env_bool("MATCH_COGNITIVE_SYNC", False),
            cache_dir=cache,
            salience_center=_env_float("MATCH_COGNITIVE_S0", 0.55),
            cooldown_sec=_env_float("MATCH_COGNITIVE_COOLDOWN", 180.0),
            crowd_numeric=_env_bool("MATCH_CROWD_LLM_NUMERIC", True),
            tier_caps=_parse_tier_caps(
                os.environ.get("MATCH_COGNITIVE_MAX_PER_TIER", ""),
                (6, 4, 4, 2, 3),
            ),
        )


def cognitive_enabled() -> bool:
    return CognitiveMatchConfig.from_env().enabled
