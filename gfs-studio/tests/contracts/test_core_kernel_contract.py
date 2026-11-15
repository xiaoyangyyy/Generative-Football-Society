from gfs_core.contracts import Plugin, PluginMetadata
from gfs_core.kernel import PluginKernel, PluginRegistry


class StudioPlugin:
    def __init__(self):
        self.metadata = PluginMetadata(
            id="gfs.studio.example",
            version="1.0.0",
            api=1,
            contract="gfs_studio.contracts.product",
            provides=("studio-example",),
        )

    def activate(self, context):
        context.provide("studio-example", self)

    def deactivate(self):
        pass


def test_studio_plugin_uses_core_contract_and_kernel():
    plugin = StudioPlugin()
    assert isinstance(plugin, Plugin)
    kernel = PluginKernel(PluginRegistry([plugin])).start()
    assert kernel.context.require("studio-example") is plugin
    kernel.stop()
    assert kernel.context.capabilities == ()
