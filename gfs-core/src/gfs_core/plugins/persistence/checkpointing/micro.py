import re

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.identity import content_identity


class MicroCheckpointService:
    def __init__(self, workspace) -> None:
        self.workspace = workspace

    def save(self, name: str, document: dict) -> dict:
        name = _name(name)
        required = {"schema_version", "request", "request_sha256", "next_tick", "state_sha256", "status"}
        if set(document) != required or document["schema_version"] != 1:
            raise ValueError("invalid micro checkpoint schema")
        if document["request_sha256"] != content_identity(document["request"]):
            raise ValueError("micro checkpoint request hash mismatch")
        if document["next_tick"] < 0 or len(document["state_sha256"]) != 64:
            raise ValueError("invalid micro checkpoint state")
        if document["status"] not in {"running", "interrupted", "completed"}:
            raise ValueError("invalid micro checkpoint status")
        self.workspace.write(f"micro-checkpoints/{name}.json", document)
        return document

    def load(self, name: str) -> dict:
        name = _name(name)
        document = self.workspace.read(f"micro-checkpoints/{name}.json")
        if document.get("schema_version") != 1:
            raise ValueError("unsupported micro checkpoint")
        if document["request_sha256"] != content_identity(document["request"]):
            raise ValueError("micro checkpoint identity mismatch")
        return document


class MicroCheckpointPlugin:
    metadata = PluginMetadata(
        id="gfs.core.persistence.micro-checkpointing",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.persistence",
        provides=("micro-checkpoints",),
        requires=("gfs.core.persistence.workspace",),
    )

    def __init__(self, workspace) -> None:
        self._workspace = workspace

    def activate(self, context: PluginContext) -> None:
        context.provide("micro-checkpoints", MicroCheckpointService(self._workspace))

    def deactivate(self) -> None:
        pass


def _name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise ValueError("invalid micro checkpoint name")
    return value
