from gfs_core.contracts import PluginContext, PluginMetadata


class BundlePlugin:
    def __init__(
        self,
        metadata: PluginMetadata,
        capability: str,
        bindings: dict[str, str],
    ) -> None:
        self.metadata = metadata
        self.capability = capability
        self.bindings = bindings

    def activate(self, context: PluginContext) -> None:
        context.provide(
            self.capability,
            {
                name: context.require(capability)
                for name, capability in self.bindings.items()
            },
        )

    def deactivate(self) -> None:
        pass
