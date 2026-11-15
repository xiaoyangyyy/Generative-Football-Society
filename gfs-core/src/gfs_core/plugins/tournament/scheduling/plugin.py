from gfs_core.contracts import PluginContext, PluginMetadata

from .groups import WORLD_CUP_2026_GROUPS


class SchedulingPlugin:
    metadata = PluginMetadata(
        id="gfs.core.tournament.scheduling",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.tournament",
        provides=("scheduling",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("scheduling", WORLD_CUP_2026_GROUPS)

    def deactivate(self) -> None:
        pass
