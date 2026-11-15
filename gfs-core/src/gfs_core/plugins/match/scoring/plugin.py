from gfs_core.contracts import PluginContext, PluginMetadata

from .service import MacroScoreService


class ScoringPlugin:
    metadata = PluginMetadata(
        id="gfs.core.match.scoring",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.match",
        provides=("scoring",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("scoring", MacroScoreService())

    def deactivate(self) -> None:
        pass
