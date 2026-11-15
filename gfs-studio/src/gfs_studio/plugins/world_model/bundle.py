from gfs_core.contracts import PluginContext, PluginMetadata


class WorldModelBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.world-model",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.world_model",
        provides=("world-model-bundle",),
        requires=(
            "gfs.studio.world-model.latent",
            "gfs.studio.world-model.frame",
            "gfs.studio.world-model.planning",
            "gfs.studio.world-model.policies",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("world-model-bundle", {
            "latent": context.require("latent-models"),
            "frame": context.require("frame-models"),
            "planning": context.require("planning"),
            "policies": context.require("policies"),
        })

    def deactivate(self) -> None:
        pass
