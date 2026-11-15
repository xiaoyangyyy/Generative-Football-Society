from gfs_core.contracts import PluginContext, PluginMetadata

from .export import DatasetService


class TraceExportPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.training.trace-export",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.training",
        provides=("trace-export",),
        requires=(
            "gfs.core.match",
            "gfs.core.status.confidence",
            "gfs.core.data.tactics",
            "gfs.core.match.scoring",
            "gfs.studio.world-model.latent",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("trace-export", DatasetService(
            context.require("sample-confidence"),
            context.require("tactics"),
            context.require("scoring"),
            context.require("match-runner"),
            context.require("trace-codec"),
        ))

    def deactivate(self) -> None:
        pass
