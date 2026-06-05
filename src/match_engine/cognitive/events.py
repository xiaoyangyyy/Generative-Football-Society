"""Cognitive trigger events (System 2 inputs)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


ENTITY_TIER_COACH = "coach"
ENTITY_TIER_PLAYER = "player"
ENTITY_TIER_REFEREE = "referee"
ENTITY_TIER_ASSISTANT = "assistant"
ENTITY_TIER_CROWD = "crowd"


@dataclass
class CognitiveTriggerEvent:
    t_sec: float
    kind: str
    entity_tier: str
    entity_id: str
    team_id: Optional[str] = None
    salience: float = 0.0
    facts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitivePlanRecord:
    """Structured System 2 output + audit metadata."""

    trigger: CognitiveTriggerEvent
    plan: Dict[str, Any]
    applied: bool = False
    error: str = ""
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger.to_dict(),
            "plan": self.plan,
            "applied": self.applied,
            "error": self.error,
            "cached": self.cached,
        }
