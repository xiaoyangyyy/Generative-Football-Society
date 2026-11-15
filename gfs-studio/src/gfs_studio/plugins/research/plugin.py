from gfs_core.contracts import PluginContext, PluginMetadata


class ResearchBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.research",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.research",
        provides=("research-bundle",),
        requires=(
            "gfs.studio.research.calibration",
            "gfs.studio.research.ablation",
            "gfs.studio.research.experiments",
            "gfs.studio.research.reproduction",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("research-bundle", {
            "calibration": context.require("calibration"),
            "ablation": context.require("ablation"),
            "experiments": context.require("experiments"),
            "reproduction": context.require("reproduction"),
        })

    def deactivate(self) -> None:
        pass
