from __future__ import annotations

from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from .csv_repository import CSVResultsRepository


class ResultsPlugin:
    metadata = PluginMetadata(
        id="gfs.core.data.results",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.data",
        provides=("results",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._repository = CSVResultsRepository(Path(project_root) / "data" / "raw")

    def activate(self, context: PluginContext) -> None:
        context.provide("results", self._repository)

    def deactivate(self) -> None:
        pass
