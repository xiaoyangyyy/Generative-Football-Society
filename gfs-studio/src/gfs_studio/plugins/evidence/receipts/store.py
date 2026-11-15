from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_studio.contracts.evidence import EvidenceReceipt


class EvidenceStore:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.frozen_root = self.root / "evidence" / "frozen"
        self.receipts_root = self.root / "evidence" / "receipts"
        self.current_root = self.root / "evidence" / "current"

    @staticmethod
    def _confined(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("evidence path escapes its root") from error
        return path

    def read_frozen(self, relative: str) -> dict:
        path = self._confined(self.frozen_root, relative)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"frozen evidence must be an object: {relative}")
        return payload

    def verify_current_identity(self, relative: str, expected_sha256: str) -> bool:
        path = self._confined(self.frozen_root, relative)
        return _file_hash(path) == expected_sha256

    def read_current(self, relative: str) -> dict:
        path = self._confined(self.current_root, relative)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("current evidence must be an object")
        return payload

    def write_current(self, relative: str, document: dict) -> str:
        if not isinstance(document, dict):
            raise ValueError("current evidence must be an object")
        content = json.dumps(
            document, ensure_ascii=False, indent=2, allow_nan=False,
        ).encode("utf-8")
        path = self._confined(self.current_root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return hashlib.sha256(content).hexdigest()

    def issue_receipt(self, identity: str, document: dict) -> EvidenceReceipt:
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 180:
            raise ValueError("receipt identity is invalid")
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in identity):
            raise ValueError("receipt identity contains unsupported characters")
        content = json.dumps(
            document, ensure_ascii=False, indent=2, allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        path = self._confined(self.receipts_root, f"{identity}.json")
        if path.exists():
            raise FileExistsError(f"receipt already exists: {identity}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return EvidenceReceipt(identity, digest, str(path))

    def read_receipt(self, identity: str, expected_document: dict | None = None) -> EvidenceReceipt:
        path = self._confined(self.receipts_root, f"{identity}.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("evidence receipt must contain an object")
        if expected_document is not None and payload != expected_document:
            raise ValueError("evidence receipt content mismatch")
        return EvidenceReceipt(identity, _file_hash(path), str(path))


class ReceiptsPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.evidence.receipts",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.evidence",
        provides=("receipts",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._store = EvidenceStore(project_root)

    def activate(self, context: PluginContext) -> None:
        context.provide("receipts", self._store)

    def deactivate(self) -> None:
        pass


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
