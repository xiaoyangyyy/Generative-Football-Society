from gfs_core.contracts import PluginContext, PluginMetadata

from .service import SpatialService


class SpatialPlugin:
    metadata = PluginMetadata(
        id="gfs.core.match.spatial",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.match",
        provides=("spatial",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("spatial", SpatialService())

    def deactivate(self) -> None:
        pass
