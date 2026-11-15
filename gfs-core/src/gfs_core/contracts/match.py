from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .micro_state import MicroMatchSummary


@dataclass(frozen=True, slots=True)
class MatchScore:
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float


class ScoreSimulator(Protocol):
    def simulate(
        self,
        status_home: float,
        status_away: float,
        *,
        is_knockout: bool,
        rng: np.random.Generator,
    ) -> MatchScore: ...


@runtime_checkable
class MatchRunner(Protocol):
    def run(self, home_profile, away_profile, **options) -> MicroMatchSummary | dict: ...


@runtime_checkable
class MatchWorkflow(Protocol):
    def run(
        self,
        home: str,
        away: str,
        *,
        seed: int = 42,
        fast: bool = False,
        **options,
    ) -> MicroMatchSummary | dict: ...
