"""Narrative events that propagate generated atmosphere into social state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.simulation.psychological_state import PsychologicalState


@dataclass(frozen=True)
class NarrativeEvent:
    text: str
    signals: dict[str, float]
    targets: tuple[str, ...]
    persistence: float
    stage: str


class NarrativeEventBus:
    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def publish(self, payload: dict[str, Any], agents: list[Any], *, stage: str) -> dict[str, Any]:
        event = NarrativeEvent(
            text=str(payload.get("key_event", "")),
            signals={k: float(v) for k, v in dict(payload.get("signals") or {}).items()},
            targets=tuple(str(v) for v in payload.get("targets", [])),
            persistence=float(np.clip(payload.get("persistence", 0.35), 0.0, 1.0)),
            stage=stage,
        )
        applied: dict[str, Any] = {}
        if event.persistence <= 0.0 or not event.text.strip():
            record = {"event": event, "applied": applied}
            self.history.append(record)
            return record
        for agent in agents:
            if event.targets and agent.name not in event.targets and agent.team_name not in event.targets:
                continue
            hostility = event.signals.get("crowd_hostility", 0.0)
            anxiety = event.signals.get("player_anxiety", 0.0)
            unity = event.signals.get("unity_signal", 0.0)
            amplification = event.signals.get("media_amplification", 0.0)
            social_signal = {
                "sentiment": float(np.clip(unity - 0.55 * hostility - 0.45 * anxiety, -1.0, 1.0)),
                "unity_signal": float(np.clip(unity - 0.25 * anxiety, -1.0, 1.0)),
                "provocation_level": float(np.clip(max(0.0, hostility) * amplification, 0.0, 1.0)),
                "credibility": float(np.clip(0.45 + 0.4 * event.persistence, 0.0, 1.0)),
            }
            feedback = agent.apply_social_signal(social_signal, stage_pressure=event.persistence)
            previous_psychology = getattr(agent, "psychological_state", PsychologicalState())
            psychology = previous_psychology.updated(event.signals, event.persistence)
            agent.psychological_state = psychology
            coach_pressure = event.signals.get("coach_pressure", 0.0)
            agent.roles["Manager"]["pressure"] = float(
                np.clip(agent.roles["Manager"].get("pressure", 0.0) + 0.12 * coach_pressure * event.persistence, 0.0, 1.0)
            )
            agent.record_social_dialogue_event(
                event.text,
                stage,
                "llm_atmosphere",
                social_signal,
            )
            applied[agent.name] = {
                "signal": social_signal,
                "feedback": feedback,
                "psychology_before": previous_psychology,
                "psychology_after": psychology,
                "decision_modifiers": psychology.decision_modifiers(),
            }
        record = {"event": event, "applied": applied}
        self.history.append(record)
        return record
