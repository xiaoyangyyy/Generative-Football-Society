from gfs_core.contracts import PluginContext, PluginMetadata

from .engine import WorldCupEngine


class AdvancementPlugin:
    metadata = PluginMetadata(
        id="gfs.core.tournament.advancement",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.tournament",
        provides=("advancement", "monte-carlo"),
        requires=(
            "gfs.core.tournament.scheduling",
            "gfs.core.match.scoring",
            "gfs.core.status.confidence",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        engine = WorldCupEngine(
            context.require("scheduling"),
            context.require("scoring"),
        )
        context.provide("advancement", engine)
        context.provide("monte-carlo", engine)

    def deactivate(self) -> None:
        pass
