from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.persistence import RollbackError


class MatchTransactionService:
    def __init__(self, workspace) -> None:
        self.workspace = workspace
        self._active_identity: str | None = None

    def execute(self, identity: str, resource, operation):
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 180:
            raise ValueError("transaction identity is invalid")
        if self._active_identity is not None:
            raise RuntimeError("nested match transactions are forbidden")
        with self.workspace.lease("match-transaction.lock"):
            self._active_identity = identity
            try:
                snapshot = deepcopy(dict(resource.snapshot()))
                checkpoint_hash = self.workspace.write_checkpoint(snapshot)
                self.workspace.write("transaction.json", {
                    "schema_version": 1,
                    "state": "prepared",
                    "identity": identity,
                    "checkpoint_sha256": checkpoint_hash,
                })
                try:
                    result = operation()
                    self.workspace.write_checkpoint(resource.snapshot())
                    resource.verify(result)
                except BaseException as original_error:
                    self._rollback(
                        identity,
                        resource,
                        snapshot,
                        checkpoint_hash,
                        original_error,
                    )
                    raise
                self.workspace.write("transaction.json", {
                    "schema_version": 1,
                    "state": "committed",
                    "identity": identity,
                    "checkpoint_sha256": checkpoint_hash,
                })
                self.workspace.write_receipt(identity, {
                    "schema_version": 1,
                    "state": "committed",
                    "identity": identity,
                    "recorded_at": _now(),
                })
                return result
            finally:
                self._active_identity = None

    def _rollback(self, identity, resource, snapshot, checkpoint_hash, original_error):
        rollback_error = None
        try:
            resource.restore(snapshot)
            self.workspace.write_checkpoint(snapshot)
            if dict(resource.snapshot()) != snapshot:
                raise RuntimeError("resource rollback did not converge")
            self.workspace.write("transaction.json", {
                "schema_version": 1,
                "state": "rolled_back",
                "identity": identity,
                "checkpoint_sha256": checkpoint_hash,
            })
        except BaseException as error:
            rollback_error = error
        self.workspace.write_receipt(identity, {
            "schema_version": 1,
            "state": "rolled_back" if rollback_error is None else "rollback_failed",
            "identity": identity,
            "rollback_verified": rollback_error is None,
            "original_error_type": type(original_error).__name__,
            "rollback_error_type": (
                None if rollback_error is None else type(rollback_error).__name__
            ),
            "recorded_at": _now(),
        })
        if rollback_error is not None:
            raise RollbackError(original_error, rollback_error) from original_error


class TransactionPlugin:
    metadata = PluginMetadata(
        id="gfs.core.persistence.transactions",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.persistence",
        provides=("match-transactions",),
        requires=("gfs.core.persistence.workspace",),
    )

    def __init__(self, workspace) -> None:
        self._workspace = workspace

    def activate(self, context: PluginContext) -> None:
        context.provide("match-transactions", MatchTransactionService(self._workspace))

    def deactivate(self) -> None:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
