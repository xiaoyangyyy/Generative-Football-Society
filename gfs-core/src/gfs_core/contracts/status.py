from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gfs_core.contracts.data import HistoricalResults


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    table: pd.DataFrame
    team_matches: pd.DataFrame
    source: HistoricalResults

    def ranked(self, limit: int = 20) -> pd.DataFrame:
        if limit < 1:
            raise ValueError("status ranking limit must be positive")
        columns = [
            "total_matches",
            "historical_base",
            "modern_power",
            "final_status_score",
            "global_exposure_gate",
            "sample_confidence",
            "tier",
        ]
        return self.table.sort_values(
            "final_status_score", ascending=False,
        ).loc[:, columns].head(limit)
