from gfs_core.contracts import PluginContext, PluginMetadata


class ParticipantsBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.participants",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.participants",
        provides=("participants-bundle",),
        requires=(
            "gfs.studio.participants.sessions",
            "gfs.studio.participants.delivery",
            "gfs.studio.participants.scoring",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("participants-bundle", {
            "sessions": context.require("participant-sessions"),
            "delivery": context.require("participant-delivery"),
            "scoring": context.require("participant-scoring"),
        })

    def deactivate(self) -> None:
        pass
