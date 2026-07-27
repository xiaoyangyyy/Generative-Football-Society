"""Coach-to-unit-to-player option decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TacticalOption:
    name: str
    unit: str
    target: Mapping[str, float]
    horizon_seconds: float
    confidence: float


class HierarchicalPolicy:
    UNITS = ("back_line", "midfield", "front_line")

    def decompose(self, controls: Mapping[str, float], *, confidence: float = 1.0) -> tuple[TacticalOption, ...]:
        press = float(np.clip(controls.get("pressing_intensity", 0.5), 0.0, 1.0))
        risk = float(np.clip(controls.get("risk_budget", 0.5), 0.0, 1.0))
        line = float(np.clip(controls.get("line_height", 0.5), 0.0, 1.0))
        rotation = float(np.clip(controls.get("rotation_aggressiveness", 0.5), 0.0, 1.0))
        conf = float(np.clip(confidence, 0.0, 1.0))
        return (
            TacticalOption("hold_or_step", "back_line", {"line": line, "cover": 1.0 - risk}, 20.0, conf),
            TacticalOption("press_and_connect", "midfield", {"press": press, "support": rotation}, 12.0, conf),
            TacticalOption("threaten_depth", "front_line", {"risk": risk, "press": press}, 8.0, conf),
        )

    def player_intent(self, option: TacticalOption, role: str) -> dict[str, float]:
        role = role.upper()
        relevance = {
            "back_line": 1.0 if role in {"GK", "CB", "LB", "RB"} else 0.25,
            "midfield": 1.0 if role in {"DM", "CM", "AM", "LM", "RM"} else 0.35,
            "front_line": 1.0 if role in {"LW", "RW", "ST"} else 0.25,
        }[option.unit]
        return {
            key: float(np.clip(value * relevance * option.confidence, 0.0, 1.0))
            for key, value in option.target.items()
        }
