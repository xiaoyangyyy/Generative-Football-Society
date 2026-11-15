from gfs_core.contracts import PluginContext, PluginMetadata

from .reports import JsonReportService


class JsonPresentationPlugin:
    metadata = PluginMetadata(
        id="gfs.core.presentation.json",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.presentation",
        provides=("json-reports",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("json-reports", JsonReportService())

    def deactivate(self) -> None:
        pass
