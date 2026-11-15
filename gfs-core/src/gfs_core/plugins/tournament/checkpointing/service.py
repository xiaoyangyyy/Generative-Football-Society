from gfs_core.contracts import PluginContext, PluginMetadata

from .carryover import CrossMatchService


class CheckpointService:
    def __init__(self, workspace, recovery) -> None:
        self.workspace = workspace
        self.recovery_service = recovery
        self.carryover = CrossMatchService()

    def save(self, snapshot) -> str:
        return self.workspace.write_checkpoint(snapshot)

    def load(self, *, expected_sha256=None):
        return self.workspace.read_checkpoint(expected_sha256=expected_sha256)

    def recover(self, resource) -> str:
        return self.recovery_service.recover(resource)


class CheckpointPlugin:
    metadata = PluginMetadata(
        id="gfs.core.tournament.checkpointing",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.tournament",
        provides=("checkpointing",),
        requires=("gfs.core.persistence.recovery",),
    )

    def __init__(self, workspace) -> None:
        self._workspace = workspace

    def activate(self, context: PluginContext) -> None:
        context.provide(
            "checkpointing",
            CheckpointService(self._workspace, context.require("recovery")),
        )

    def deactivate(self) -> None:
        pass
