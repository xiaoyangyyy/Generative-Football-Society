from gfs_core.contracts import PluginContext, PluginMetadata


class ProductBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.product",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.product",
        provides=("product-bundle",),
        requires=(
            "gfs.studio.product.studio",
            "gfs.studio.product.manager",
            "gfs.studio.product.comparison",
            "gfs.studio.product.recovery",
            "gfs.studio.product.tasks",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("product-bundle", {
            "workflows": context.require("studio-workflows"),
            "manager": context.require("manager"),
            "comparison": context.require("comparison"),
            "recovery": context.require("product-recovery"),
            "tasks": context.require("product-tasks"),
        })

    def deactivate(self) -> None:
        pass
