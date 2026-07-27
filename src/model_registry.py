"""Atomic model lifecycle registry: candidate -> evaluated -> sealed -> deployed."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

TRANSITIONS = {
    "candidate": {"evaluated"},
    "evaluated": {"sealed"},
    "sealed": {"deployed"},
    "deployed": set(),
}


def artifact_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"schema_version": 1, "models": {}}

    def register(self, name: str, version: str, artifact: str | Path) -> dict:
        key = f"{name}:{version}"
        if key in self.data["models"]:
            raise ValueError(f"model already registered: {key}")
        entry = {"name": name, "version": version, "status": "candidate", "artifact": str(artifact), "sha256": artifact_sha256(artifact), "evidence": {}}
        self.data["models"][key] = entry
        self._save()
        return entry

    def promote(self, name: str, version: str, status: str, evidence: dict | None = None) -> dict:
        entry = self.data["models"][f"{name}:{version}"]
        if status not in TRANSITIONS.get(entry["status"], set()):
            raise ValueError(f"invalid promotion {entry['status']} -> {status}")
        if artifact_sha256(entry["artifact"]) != entry["sha256"]:
            raise RuntimeError("artifact changed after registration")
        if status in {"sealed", "deployed"} and not evidence:
            raise ValueError("sealed and deployed promotions require evidence")
        entry["status"] = status
        entry["evidence"].update(evidence or {})
        self._save()
        return entry

    def attest(self, name: str, version: str, evidence: dict) -> dict:
        """Refresh release evidence for an immutable sealed/deployed artifact."""
        entry = self.data["models"][f"{name}:{version}"]
        if entry["status"] not in {"sealed", "deployed"} or not evidence:
            raise ValueError("attestation requires sealed/deployed status and evidence")
        if artifact_sha256(entry["artifact"]) != entry["sha256"]:
            raise RuntimeError("artifact changed after registration")
        entry["evidence"] = dict(evidence)
        self._save()
        return entry

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}-")
        os.close(fd)
        try:
            Path(temporary).write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
