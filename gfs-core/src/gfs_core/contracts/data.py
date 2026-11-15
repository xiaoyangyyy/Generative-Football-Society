from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class HistoricalResults:
    results: pd.DataFrame
    shootouts: pd.DataFrame
    raw: Mapping[str, pd.DataFrame]

    @classmethod
    def from_frames(
        cls,
        *,
        results: pd.DataFrame,
        shootouts: pd.DataFrame,
        raw: dict[str, pd.DataFrame],
    ) -> HistoricalResults:
        return cls(results, shootouts, MappingProxyType(dict(raw)))


class ResultsRepository(Protocol):
    def load(self) -> HistoricalResults: ...
