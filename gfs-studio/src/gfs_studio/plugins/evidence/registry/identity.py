import re

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.workspace import WorkspaceRepository


class IdentityRegistry:
    def __init__(self, evidence) -> None:
        self._evidence = evidence
        self._workspace = WorkspaceRepository(evidence.root)

    def register(self, relative: str, expected_sha256: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("evidence identity must be a lowercase SHA-256 digest")
        with self._workspace.lease("evidence-identities.lock", timeout=30.0):
            bindings = self._bindings()
            existing = bindings.get(relative)
            if existing is not None and existing != expected_sha256:
                raise ValueError("immutable evidence identity cannot be rebound")
            if existing is None:
                bindings[relative] = expected_sha256
                self._workspace.write("evidence_identities.json", {
                    "schema_version": 1,
                    "bindings": bindings,
                })
            return self.verify(relative)

    def verify(self, relative: str) -> dict:
        bindings = self._bindings()
        if relative not in bindings:
            raise KeyError(f"unregistered evidence identity: {relative}")
        expected = bindings[relative]
        current = self._evidence.verify_current_identity(relative, expected)
        return {
            "artifact": relative,
            "expected_sha256": expected,
            "current": current,
            "status": "current" if current else "identity_mismatch",
        }

    def list(self) -> list[dict]:
        return [self.verify(relative) for relative in sorted(self._bindings())]

    def _bindings(self) -> dict:
        try:
            document = self._workspace.read("evidence_identities.json")
        except FileNotFoundError:
            return {}
        if document.get("schema_version") != 1 or not isinstance(document.get("bindings"), dict):
            raise ValueError("invalid evidence identity registry")
        return document["bindings"]


class EvidenceRegistryPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.evidence.registry",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.evidence",
        provides=("identity-registry",),
        requires=("gfs.studio.evidence.receipts",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("identity-registry", IdentityRegistry(context.require("receipts")))

    def deactivate(self) -> None:
        pass
