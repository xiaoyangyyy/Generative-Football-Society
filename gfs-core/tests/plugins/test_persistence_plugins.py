import hashlib

import pytest

from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.persistence.lease import FileLease, LeaseUnavailable
from gfs_core.plugins.persistence.recovery.service import RecoveryPlugin, RecoveryService
from gfs_core.plugins.persistence.repositories.workspace import WorkspacePlugin, WorkspaceRepository
from gfs_core.plugins.persistence.transactions.match import TransactionPlugin


class Resource:
    def __init__(self, value=1):
        self.value = value

    def snapshot(self):
        return {"value": self.value}

    def restore(self, snapshot):
        self.value = snapshot["value"]

    def verify(self, result):
        assert result == self.value


def services(tmp_path):
    workspace = WorkspaceRepository(tmp_path)
    from gfs_core.plugins.persistence.transactions.match import MatchTransactionService

    return workspace, MatchTransactionService(workspace)


def test_successful_transaction_commits_checkpoint_and_receipt(tmp_path):
    workspace, transactions = services(tmp_path)
    resource = Resource()
    result = transactions.execute("match-1", resource, lambda: setattr(resource, "value", 2) or 2)
    assert result == 2
    assert workspace.read_checkpoint() == {"value": 2}
    assert workspace.read("transaction.json")["state"] == "committed"
    digest = hashlib.sha256(b"match-1").hexdigest()[:24]
    assert workspace.read(f"receipts/{digest}.json")["state"] == "committed"


def test_failed_operation_restores_pre_match_state(tmp_path):
    workspace, transactions = services(tmp_path)
    resource = Resource()

    def fail():
        resource.value = 99
        raise RuntimeError("simulated match failure")

    with pytest.raises(RuntimeError, match="simulated match failure"):
        transactions.execute("match-failure", resource, fail)
    assert resource.value == 1
    assert workspace.read_checkpoint() == {"value": 1}
    assert workspace.read("transaction.json")["state"] == "rolled_back"


def test_process_crash_recovery_restores_prepared_checkpoint(tmp_path):
    workspace = WorkspaceRepository(tmp_path)
    resource = Resource()
    checkpoint_hash = workspace.write_checkpoint(resource.snapshot())
    workspace.write("transaction.json", {
        "schema_version": 1,
        "state": "prepared",
        "identity": "match-crash",
        "checkpoint_sha256": checkpoint_hash,
    })
    resource.value = 99
    assert RecoveryService(workspace).recover(resource) == "rolled_back"
    assert resource.value == 1
    assert workspace.read("transaction.json")["state"] == "rolled_back"
    assert RecoveryService(workspace).recover(resource) == "already_rolled_back"


def test_recovery_rejects_checkpoint_identity_drift(tmp_path):
    workspace = WorkspaceRepository(tmp_path)
    workspace.write_checkpoint({"value": 1})
    workspace.write("transaction.json", {
        "state": "prepared",
        "identity": "match-drift",
        "checkpoint_sha256": "0" * 64,
    })
    with pytest.raises(ValueError, match="identity mismatch"):
        RecoveryService(workspace).recover(Resource())


def test_workspace_lease_excludes_second_owner(tmp_path):
    path = tmp_path / "data" / "persistence" / "match.lock"
    with FileLease(path):
        with pytest.raises(LeaseUnavailable):
            FileLease(path).acquire()


def test_plugins_compose_complete_persistence_stack(tmp_path):
    workspace = WorkspacePlugin(tmp_path)
    kernel = PluginKernel(PluginRegistry([
        workspace,
        TransactionPlugin(workspace._repository),
        RecoveryPlugin(workspace._repository),
    ])).start(["gfs.core.persistence.recovery"])
    assert kernel.context.require("workspace-repository") is workspace._repository
    assert kernel.context.require("match-transactions") is not None
    assert kernel.context.require("recovery") is not None
    kernel.stop()
