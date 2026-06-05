"""Assistant referee System 1 ODE + offside modulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.match_engine.math_utils import tanh_clip

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState


def assistant_offside_modifier(state: "MatchAffectiveState") -> float:
    """Aggregate AR strictness → added offside risk (±~0.08)."""
    assistants = getattr(state.referee, "assistants", None) or []
    if not assistants:
        return 0.0
    mean_strict = float(np.mean([a.offside_strictness for a in assistants]))
    trust = float(np.mean([a.trust_with_center for a in assistants]))
    return float(0.12 * (mean_strict - 0.5) * trust)


def step_assistant_dynamics(state: "MatchAffectiveState", dt: float) -> None:
    """Slow trust / strictness drift with controversy."""
    assistants = getattr(state.referee, "assistants", None) or []
    if not assistants:
        return
    controversy = float(state.referee.controversy_integral)
    for ar in assistants:
        ar.trust_with_center = float(
            tanh_clip(ar.trust_with_center - 0.02 * controversy * dt + 0.01 * (1.0 - controversy) * dt)
        )
        ar.offside_strictness = float(
            np.clip(ar.offside_strictness + 0.015 * controversy * dt - 0.008 * (1.0 - controversy) * dt, 0.25, 0.85)
        )
