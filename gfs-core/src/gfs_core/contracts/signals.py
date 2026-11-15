from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class InternalMatchSignals:
    coordination_home: float = 0.6
    coordination_away: float = 0.6
    conflict_home: float = 0.12
    conflict_away: float = 0.12


def normalize_internal_signals(home, away) -> InternalMatchSignals:
    return InternalMatchSignals(
        _bounded(home, "coordination", 0.6, 0.0, 1.0),
        _bounded(away, "coordination", 0.6, 0.0, 1.0),
        _bounded(home, "conflict_heat", 0.12, 0.0, 1.05),
        _bounded(away, "conflict_heat", 0.12, 0.0, 1.05),
    )


def _bounded(payload: Mapping[str, object] | None, key: str, default: float, lower: float, upper: float) -> float:
    source = payload if isinstance(payload, Mapping) else {}
    try:
        value = float(source.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value):
        return default
    return min(upper, max(lower, value))
