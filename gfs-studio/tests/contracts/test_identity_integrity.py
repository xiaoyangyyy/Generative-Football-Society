import json
from pathlib import Path

import pytest

from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_studio.plugins.evidence.receipts.store import ReceiptsPlugin
from gfs_studio.plugins.evidence.registry.identity import EvidenceRegistryPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_identity_mismatch_and_persistent_binding(tmp_path):
    name = "m2_mirrored_policy_protocol_v1.json"
    progress = json.loads((PROJECT_ROOT / "evidence" / "frozen" / "m2_mirrored_policy_progress_v6.json").read_text())
    expected = progress["execution_identity"]["protocol_sha256"]
    frozen = tmp_path / "evidence" / "frozen"
    frozen.mkdir(parents=True)
    (frozen / name).write_bytes(b"{}")
    registry = PluginRegistry([ReceiptsPlugin(tmp_path), EvidenceRegistryPlugin()])
    kernel = PluginKernel(registry).start(["gfs.studio.evidence.registry"])
    identities = kernel.context.require("identity-registry")
    assert identities.register(name, expected)["current"] is False
    kernel.stop()
    kernel = PluginKernel(registry).start(["gfs.studio.evidence.registry"])
    assert kernel.context.require("identity-registry").verify(name)["status"] == "identity_mismatch"
    with pytest.raises(ValueError, match="cannot be rebound"):
        kernel.context.require("identity-registry").register(name, "0" * 64)
    kernel.stop()
