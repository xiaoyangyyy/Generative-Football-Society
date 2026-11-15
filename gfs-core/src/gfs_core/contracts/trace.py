from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .events import MicroEvent
from .micro_state import MatchAffectiveState


@dataclass(frozen=True, slots=True)
class TraceTick:
    match_id: str
    tick: int
    dt: float
    state: MatchAffectiveState
    action: str
    events: tuple[MicroEvent, ...] = ()
    metadata: Mapping[str, object] | None = None
