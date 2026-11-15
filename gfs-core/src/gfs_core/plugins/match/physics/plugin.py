from gfs_core.contracts import PluginContext, PluginMetadata

from .service import PhysicsService


class PhysicsPlugin:
    metadata = PluginMetadata(
        id="gfs.core.match.physics",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.match",
        provides=("physics",),
        requires=("gfs.core.match.spatial",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("physics", PhysicsService(context.require("spatial")))

    def deactivate(self) -> None:
        pass
