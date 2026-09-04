"""Validated, side-specific internal team signals for match runners."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


DEFAULT_COORDINATION = 0.6
DEFAULT_CONFLICT_HEAT = 0.12
MAX_CONFLICT_HEAT = 1.05


@dataclass(frozen=True, slots=True)
class InternalMatchSignals:
    coordination_home: float
    coordination_away: float
    conflict_home: float
    conflict_away: float


def _bounded_signal(
    payload: Mapping[str, object] | None,
    key: str,
    *,
    default: float,
    lower: float,
    upper: float,
) -> float:
    source = payload if isinstance(payload, Mapping) else {}
    try:
        value = float(source.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value):
        return default
    return min(upper, max(lower, value))


def normalize_internal_match_signals(
    internal_home: Mapping[str, object] | None,
    internal_away: Mapping[str, object] | None,
) -> InternalMatchSignals:
    """Return finite, bounded signals without allowing home/away cross-talk."""
    return InternalMatchSignals(
        coordination_home=_bounded_signal(
            internal_home,
            "coordination",
            default=DEFAULT_COORDINATION,
            lower=0.0,
            upper=1.0,
        ),
        coordination_away=_bounded_signal(
            internal_away,
            "coordination",
            default=DEFAULT_COORDINATION,
            lower=0.0,
            upper=1.0,
        ),
        conflict_home=_bounded_signal(
            internal_home,
            "conflict_heat",
            default=DEFAULT_CONFLICT_HEAT,
            lower=0.0,
            upper=MAX_CONFLICT_HEAT,
        ),
        conflict_away=_bounded_signal(
            internal_away,
            "conflict_heat",
            default=DEFAULT_CONFLICT_HEAT,
            lower=0.0,
            upper=MAX_CONFLICT_HEAT,
        ),
    )
