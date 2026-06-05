"""In-match cognitive layer: System 1 (continuous) + System 2 (event-triggered LLM)."""

from src.match_engine.cognitive.bus import MatchCognitiveBus
from src.match_engine.cognitive.config import CognitiveMatchConfig, cognitive_enabled
from src.match_engine.cognitive.executor import CognitiveExecutor
from src.match_engine.cognitive.events import CognitivePlanRecord, CognitiveTriggerEvent

__all__ = [
    "MatchCognitiveBus",
    "CognitiveExecutor",
    "CognitiveMatchConfig",
    "cognitive_enabled",
    "CognitiveTriggerEvent",
    "CognitivePlanRecord",
]
