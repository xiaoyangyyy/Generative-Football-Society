from __future__ import annotations

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.status import StatusSnapshot

from .exposure import add_exposure_gate, add_sample_confidence


class ConfidencePlugin:
    metadata = PluginMetadata(
        id="gfs.core.status.confidence",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.status",
        provides=("sample-confidence",),
        requires=("gfs.core.status.ranking",),
    )

    def __init__(self, *, threshold: int = 80) -> None:
        self._threshold = threshold

    def activate(self, context: PluginContext) -> None:
        ranking: StatusSnapshot = context.require("team-ranking")
        table = add_exposure_gate(ranking.table, ranking.team_matches)
        table = add_sample_confidence(table, threshold=self._threshold)
        context.provide(
            "sample-confidence",
            StatusSnapshot(table, ranking.team_matches, ranking.source),
        )

    def deactivate(self) -> None:
        pass
