from __future__ import annotations

from pathlib import Path

import pandas as pd

from gfs_core.contracts.data import HistoricalResults
from .cleaning import clean_results
from .identities import normalize_identities

REQUIRED_COLUMNS = {
    "results": {"date", "home_team", "away_team", "home_score", "away_score", "tournament"},
    "goalscorers": {"date", "home_team", "away_team"},
    "shootouts": {"date", "home_team", "away_team", "winner"},
    "former_names": {"former", "current"},
}


class CSVResultsRepository:
    def __init__(self, raw_data_dir: str | Path) -> None:
        self._raw_data_dir = Path(raw_data_dir)
        self._snapshot: HistoricalResults | None = None

    def load(self) -> HistoricalResults:
        if self._snapshot is None:
            raw = {
                kind: self._read_csv(kind)
                for kind in ("results", "goalscorers", "shootouts", "former_names")
            }
            results = normalize_identities(
                clean_results(raw["results"]), raw["former_names"],
            )
            self._snapshot = HistoricalResults.from_frames(
                results=results,
                shootouts=raw["shootouts"].copy(),
                raw=raw,
            )
        return self._snapshot

    def _read_csv(self, kind: str) -> pd.DataFrame:
        path = self._raw_data_dir / f"{kind}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"required {kind} dataset not found: {path}")
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS[kind].difference(frame.columns))
        if missing:
            raise ValueError(
                f"invalid {kind} dataset; missing columns: {', '.join(missing)}"
            )
        return frame
