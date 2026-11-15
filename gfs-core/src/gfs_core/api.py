from __future__ import annotations

from pathlib import Path

import pandas as pd

from gfs_core.bootstrap.application import CoreApplication
from gfs_core.contracts.status import StatusSnapshot


def load_status_snapshot(
    project_root: str | Path | None = None,
    *,
    current_year: int = 2026,
) -> StatusSnapshot:
    return CoreApplication(
        project_root, current_year=current_year,
    ).load_status()


def load_status_table(
    project_root: str | Path | None = None,
    *,
    current_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    snapshot = load_status_snapshot(project_root, current_year=current_year)
    return snapshot.table, snapshot.team_matches, dict(snapshot.source.raw)


def run_monte_carlo(
    simulations: int = 1000,
    *,
    project_root: str | Path | None = None,
    seed: int = 42,
) -> list[dict[str, object]]:
    return CoreApplication(project_root).run_monte_carlo(
        simulations=simulations, seed=seed,
    )


def run_micro_match(
    home: str = "Brazil",
    away: str = "Argentina",
    *,
    project_root: str | Path | None = None,
    seed: int = 42,
    fast: bool = False,
    stage_name: str = "match",
):
    return CoreApplication(project_root).run_micro_match(
        home,
        away,
        seed=seed,
        fast=fast,
        stage_name=stage_name,
    )
