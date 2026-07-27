"""Causal narrative-to-psychology state with decay and ablation support."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PsychologicalState:
    morale: float = 0.0
    pressure: float = 0.0
    trust: float = 0.0
    risk_appetite: float = 0.0
    conflict: float = 0.0
    audience_activation: float = 0.0

    def updated(self, signals: dict[str, float], persistence: float, *, enabled: bool = True):
        if not enabled:
            return self
        strength = float(np.clip(persistence, 0.0, 1.0))
        hostility = float(signals.get("crowd_hostility", 0.0))
        anxiety = float(signals.get("player_anxiety", 0.0))
        unity = float(signals.get("unity_signal", 0.0))
        media = float(signals.get("media_amplification", 0.0))
        coach = float(signals.get("coach_pressure", 0.0))
        changes = {
            "morale": 0.18 * (unity - anxiety),
            "pressure": 0.20 * (coach + anxiety + 0.4 * hostility),
            "trust": 0.16 * (unity - 0.5 * hostility),
            "risk_appetite": 0.12 * (unity - coach),
            "conflict": 0.15 * (hostility + media - unity),
            "audience_activation": 0.22 * media * abs(hostility),
        }
        return replace(
            self,
            **{
                key: float(np.clip(getattr(self, key) * 0.92 + strength * delta, -1.0, 1.0))
                for key, delta in changes.items()
            },
        )

    def decision_modifiers(self) -> dict[str, float]:
        return {
            "risk": float(np.clip(0.18 * self.risk_appetite - 0.12 * self.pressure, -0.2, 0.2)),
            "execution": float(np.clip(0.10 * self.morale + 0.08 * self.trust - 0.14 * self.pressure, -0.2, 0.2)),
            "coordination": float(np.clip(0.12 * self.trust - 0.12 * self.conflict, -0.2, 0.2)),
        }
