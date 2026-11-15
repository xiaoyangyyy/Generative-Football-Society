from gfs_core.contracts import PluginContext, PluginMetadata


class EvidenceBundlePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.evidence",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.evidence",
        provides=("evidence-bundle",),
        requires=(
            "gfs.studio.evidence.receipts",
            "gfs.studio.evidence.registry",
            "gfs.studio.evidence.release",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("evidence-bundle", {
            "receipts": context.require("receipts"),
            "registry": context.require("identity-registry"),
            "release": context.require("release-gates"),
        })

    def deactivate(self) -> None:
        pass
