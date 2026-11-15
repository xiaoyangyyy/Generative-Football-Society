from gfs_core.contracts import PluginContext, PluginMetadata


class CognitionBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.cognition",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.cognition",
        provides=("cognition-bundle",),
        requires=(
            "gfs.studio.cognition.gateway",
            "gfs.studio.cognition.roles",
            "gfs.studio.cognition.reflection",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("cognition-bundle", {
            "gateway": context.require("provider-gateway"),
            "coach": context.require("coach-role"),
            "player": context.require("player-role"),
            "media": context.require("media-role"),
            "reflection": context.require("reflection"),
        })

    def deactivate(self) -> None:
        pass
