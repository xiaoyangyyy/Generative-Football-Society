from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata


class CheckpointRegistry:
    def __init__(self, project_root: str | Path, evidence) -> None:
        root = Path(project_root)
        self.run_root = root / "data" / "training" / "runs"
        self.artifact_root = root / "data" / "training" / "artifacts"
        self.evidence = evidence

    def register(self, run_name: str) -> dict:
        run_dir = self.run_root / run_name
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") not in {"completed", "running", "failed", "interrupted"}:
            raise ValueError("invalid training checkpoint state")
        run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        checkpoint = run_dir / "latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {run_name}")
        digest = _hash(checkpoint)
        artifact = self.artifact_root / f"{digest}.pt"
        _publish(checkpoint, artifact, digest)
        document = {
            "run_name": run_name,
            "state": status["state"],
            "epoch": int(status["epoch"]),
            "history": list(status["history"]),
            "checkpoint_sha256": digest,
            "run": run_manifest,
        }
        identity = f"checkpoint-{digest}"
        try:
            receipt = self.evidence.issue_receipt(identity, document)
        except FileExistsError:
            receipt = self.evidence.read_receipt(identity, document)
        return {
            **document,
            "checkpoint": str(artifact),
            "receipt": asdict(receipt),
            "immutable": True,
        }

    def get(self, digest: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid checkpoint identity")
        artifact = self.artifact_root / f"{digest}.pt"
        if not artifact.is_file() or _hash(artifact) != digest:
            raise ValueError("checkpoint artifact identity mismatch")
        self.evidence.read_receipt(f"checkpoint-{digest}")
        return {
            "checkpoint": str(artifact),
            "checkpoint_sha256": digest,
            "immutable": True,
        }

    def list(self) -> list[str]:
        if not self.run_root.is_dir():
            return []
        return sorted(path.name for path in self.run_root.iterdir() if path.is_dir())

    def artifacts(self) -> list[str]:
        if not self.artifact_root.is_dir():
            return []
        return sorted(path.stem for path in self.artifact_root.glob("*.pt"))


class CheckpointPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.training.checkpoints",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.training",
        provides=("checkpoint-registry",),
        requires=("gfs.studio.evidence.receipts",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = project_root

    def activate(self, context: PluginContext) -> None:
        context.provide("checkpoint-registry", CheckpointRegistry(
            self._project_root, context.require("receipts"),
        ))

    def deactivate(self) -> None:
        pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if _hash(destination) != expected_sha256:
            raise ValueError("checkpoint artifact identity mismatch")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    temporary = Path(temporary)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if _hash(temporary) != expected_sha256:
            raise ValueError("checkpoint copy changed its identity")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
