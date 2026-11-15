from gfs_core.contracts import PluginContext, PluginMetadata


class TrainingBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.training",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.training",
        provides=("training-bundle",),
        requires=("gfs.studio.training.datasets", "gfs.studio.training.trainers", "gfs.studio.training.evaluation", "gfs.studio.training.checkpoints", "gfs.studio.training.trace-export"),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("training-bundle", {
            "datasets": context.require("datasets"),
            "trace_export": context.require("trace-export"),
            "trainers": context.require("trainers"),
            "evaluation": context.require("model-evaluation"),
            "checkpoints": context.require("checkpoint-registry"),
        })

    def deactivate(self) -> None:
        pass
