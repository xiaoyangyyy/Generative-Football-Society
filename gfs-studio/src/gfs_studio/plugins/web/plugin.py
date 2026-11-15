from gfs_core.contracts import PluginContext, PluginMetadata


class WebBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.web",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.web",
        provides=("web-bundle",),
        requires=(
            "gfs.studio.web.gateway",
            "gfs.studio.web.api",
            "gfs.studio.web.views",
            "gfs.studio.web.authentication",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("web-bundle", {
            "gateway": context.require("http-gateway"),
            "api": context.require("http-api"),
            "views": context.require("web-views"),
            "authentication": context.require("authentication"),
        })

    def deactivate(self) -> None:
        pass
