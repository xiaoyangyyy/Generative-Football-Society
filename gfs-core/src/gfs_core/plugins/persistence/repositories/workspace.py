from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.workspace import WorkspaceRepository


class WorkspacePlugin:
    metadata = PluginMetadata(
        id="gfs.core.persistence.workspace",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.persistence",
        provides=("workspace-repository",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._repository = WorkspaceRepository(project_root)

    def activate(self, context: PluginContext) -> None:
        context.provide("workspace-repository", self._repository)

    def deactivate(self) -> None:
        pass


__all__ = ["WorkspacePlugin", "WorkspaceRepository"]
