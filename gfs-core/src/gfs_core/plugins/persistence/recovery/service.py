from gfs_core.contracts import PluginContext, PluginMetadata


class RecoveryService:
    def __init__(self, workspace) -> None:
        self.workspace = workspace

    def recover(self, resource) -> str:
        try:
            journal = self.workspace.read("transaction.json")
        except FileNotFoundError:
            return "no_recovery_required"
        state = journal.get("state")
        if state == "committed":
            return "already_committed"
        if state == "rolled_back":
            return "already_rolled_back"
        if state != "prepared":
            raise ValueError("unknown transaction journal state")
        checkpoint = self.workspace.read_checkpoint(
            expected_sha256=journal.get("checkpoint_sha256")
        )
        resource.restore(checkpoint)
        self.workspace.write_checkpoint(checkpoint)
        if dict(resource.snapshot()) != checkpoint:
            raise RuntimeError("resource recovery did not converge")
        identity = str(journal.get("identity") or "unknown")
        self.workspace.write("transaction.json", {
            **journal,
            "state": "rolled_back",
        })
        self.workspace.write_receipt(identity, {
            "schema_version": 1,
            "state": "rolled_back",
            "identity": identity,
            "rollback_verified": True,
        })
        return "rolled_back"


class RecoveryPlugin:
    metadata = PluginMetadata(
        id="gfs.core.persistence.recovery",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.persistence",
        provides=("recovery",),
        requires=("gfs.core.persistence.transactions",),
    )

    def __init__(self, workspace) -> None:
        self._workspace = workspace

    def activate(self, context: PluginContext) -> None:
        context.provide("recovery", RecoveryService(self._workspace))

    def deactivate(self) -> None:
        pass
