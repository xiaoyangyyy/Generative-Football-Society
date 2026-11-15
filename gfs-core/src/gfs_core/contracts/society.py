from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TeamProfile:
    team_name: str
    status_score: float = 50.0
    formation: str = "4-3-3"
    tactical: Mapping[str, float] = field(default_factory=dict)
    emotion_profile: Mapping[str, float] = field(default_factory=dict)
    psychology_profile: Mapping[str, float] = field(default_factory=dict)
    fatigue: float = 0.0
    media_exposure: float = 0.5
    conflict_heat: float = 0.12
    team_cohesion: float = 0.6
    coach_name: str = ""
    referee_grievance: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    event_id: str
    match_id: str
    team: str
    subject: str
    kind: str
    summary: str
    match_index: int
    importance: float = 0.5
    valence: float = 0.0
    arousal: float = 0.0
    emotion: Mapping[str, float] = field(default_factory=dict)
    source: str = ""
    tags: tuple[str, ...] = ()
    context: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class SocietyMemory(Protocol):
    def record(self, event: MemoryEvent) -> dict: ...

    def recall(self, team: str, **options) -> list[dict]: ...

    def effects(self, team: str, **options) -> dict: ...

    def snapshot(self) -> dict: ...

    def restore(self, snapshot: dict) -> None: ...

    def apply_to_state(self, state, *, as_of: int | None = None) -> dict: ...

    def status(self) -> dict: ...
