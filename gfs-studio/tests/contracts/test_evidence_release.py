import json
from pathlib import Path

import pytest

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import build_evidence_registry
from gfs_studio.plugins.evidence.release.current import current_verification

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_evidence_bundle_verifies_bound_identity_and_blocks_release():
    kernel = PluginKernel(build_evidence_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.evidence"])
        bundle = kernel.context.require("evidence-bundle")
        assert set(bundle) == {"receipts", "registry", "release"}
        status = bundle["release"].evaluate()
        assert status["release_ready"] is False
        assert status["code_ready"] is current_verification(PROJECT_ROOT)["code_ready"]
        assert status["checks"]["identity_integrity"] is True
        assert status["checks"]["m2_completed"] is False
        assert status["checks"]["participant_evidence"] is False
        assert "reproduction_passed" in status["open_gates"]
        assert status["scope"] == "current_build_and_historical_research_evidence"
    finally:
        kernel.stop()


def test_identity_registry_detects_mismatch_and_rejects_unbound_digests():
    progress = json.loads((PROJECT_ROOT / "evidence" / "frozen" / "m2_mirrored_policy_progress_v6.json").read_text(encoding="utf-8"))
    kernel = PluginKernel(build_evidence_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.evidence.registry"])
        registry = kernel.context.require("identity-registry")
        name = "m2_mirrored_policy_protocol_v1.json"
        assert registry.register(name, progress["execution_identity"]["protocol_sha256"])["current"] is True
        with pytest.raises(ValueError, match="cannot be rebound"):
            registry.register(name, "0" * 64)
        with pytest.raises(ValueError, match="SHA-256"):
            registry.register(name, "not-a-digest")
    finally:
        kernel.stop()
