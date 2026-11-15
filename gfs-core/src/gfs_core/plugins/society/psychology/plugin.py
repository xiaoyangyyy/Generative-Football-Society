from gfs_core.contracts import PluginContext, PluginMetadata

from .service import PsychologyService


class PsychologyPlugin:
    metadata = PluginMetadata(
        id="gfs.core.society.psychology",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.society",
        provides=("psychology",),
        requires=("gfs.core.match.actions",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("psychology", PsychologyService(context.require("actions")))

    def deactivate(self) -> None:
        pass
