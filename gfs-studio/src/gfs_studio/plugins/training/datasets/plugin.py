from gfs_core.contracts import PluginContext, PluginMetadata

from .manifest import TraceDataset


class DatasetPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.training.datasets",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.training",
        provides=("datasets",),
        requires=(),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("datasets", TraceDataset)

    def deactivate(self) -> None:
        pass
