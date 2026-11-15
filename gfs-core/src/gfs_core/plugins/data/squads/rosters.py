from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata

from .repository import RosterRepository


class RostersPlugin:
    metadata = PluginMetadata(
        id="gfs.core.data.rosters",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.data",
        provides=("rosters",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._repository = RosterRepository(project_root)

    def activate(self, context: PluginContext) -> None:
        context.provide("rosters", self._repository)

    def deactivate(self) -> None:
        pass
