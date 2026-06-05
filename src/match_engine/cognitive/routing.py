"""Tier caps and admission control for System 2 calls."""

from __future__ import annotations

from typing import Dict

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import (
    ENTITY_TIER_ASSISTANT,
    ENTITY_TIER_COACH,
    ENTITY_TIER_CROWD,
    ENTITY_TIER_PLAYER,
    ENTITY_TIER_REFEREE,
    CognitiveTriggerEvent,
)


class TierBudget:
    def __init__(self, cfg: CognitiveMatchConfig) -> None:
        caps = cfg.tier_caps
        self._caps = {
            ENTITY_TIER_COACH: caps[0],
            ENTITY_TIER_REFEREE: caps[1],
            ENTITY_TIER_PLAYER: caps[2],
            ENTITY_TIER_ASSISTANT: caps[3],
            ENTITY_TIER_CROWD: caps[4],
        }
        self._counts: Dict[str, int] = {k: 0 for k in self._caps}
        self._player_per_team: Dict[str, int] = {}

    def admit(self, trig: CognitiveTriggerEvent) -> bool:
        tier = trig.entity_tier
        cap = self._caps.get(tier, 0)
        if self._counts.get(tier, 0) >= cap:
            return False
        if tier == ENTITY_TIER_PLAYER and trig.team_id:
            key = trig.team_id
            if self._player_per_team.get(key, 0) >= max(1, cap // 2):
                return False
            self._player_per_team[key] = self._player_per_team.get(key, 0) + 1
        self._counts[tier] = self._counts.get(tier, 0) + 1
        return True

    def summary(self) -> Dict[str, int]:
        return dict(self._counts)
