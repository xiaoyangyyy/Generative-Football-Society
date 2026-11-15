from __future__ import annotations

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.status import StatusSnapshot

from .scoring import compute_team_status


class RankingPlugin:
    metadata = PluginMetadata(
        id="gfs.core.status.ranking",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.status",
        provides=("team-ranking",),
        requires=("gfs.core.data.results",),
    )

    def __init__(self, *, current_year: int = 2026) -> None:
        self._current_year = current_year

    def activate(self, context: PluginContext) -> None:
        source = context.require("results").load()
        table, team_matches = compute_team_status(
            source.results,
            source.shootouts,
            current_year=self._current_year,
        )
        context.provide(
            "team-ranking",
            StatusSnapshot(table, team_matches, source),
        )

    def deactivate(self) -> None:
        pass
