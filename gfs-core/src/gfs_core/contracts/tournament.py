from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .society import TeamProfile


class MonteCarloSimulator(Protocol):
    def run(
        self,
        stats: pd.DataFrame,
        *,
        simulations: int,
        seed: int,
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class TournamentMatchResult:
    home_goals: int
    away_goals: int
    home_xg: float
    away_xg: float
    engine: str
    summary: object | None = None


class MatchEngine(Protocol):
    name: str

    def play(
        self,
        home: TeamProfile,
        away: TeamProfile,
        *,
        stage: str,
        match_id: str,
        seed: int,
        knockout: bool,
    ) -> TournamentMatchResult: ...

    def penalties(self, seed: int) -> tuple[int, int]: ...
