from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .lease import FileLease

MAX_DOCUMENT_BYTES = 32 * 1024 * 1024


class WorkspaceRepository:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.persistence = self._confined("data", "persistence")

    def _confined(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts).resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("workspace path escapes project root") from error
        if not relative.parts:
            raise ValueError("workspace path cannot be project root")
        return path

    def lease(self, name: str = "workspace.lock", *, timeout: float = 0.0) -> FileLease:
        return FileLease(self._confined("data", "persistence", name), timeout=timeout)

    def read(self, name: str) -> dict[str, Any]:
        path = self._confined("data", "persistence", name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise ValueError(f"workspace document is oversized: {name}")
        deadline = time.monotonic() + 5.0
        while True:
            try:
                text = path.read_text(encoding="utf-8")
                break
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"workspace document must be an object: {name}")
        return payload

    def write(self, name: str, payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, indent=2, allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_DOCUMENT_BYTES:
            raise ValueError(f"workspace document is oversized: {name}")
        path = self._confined("data", "persistence", name)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if os.name != "nt" or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return hashlib.sha256(encoded).hexdigest()

    def write_checkpoint(self, snapshot: Mapping[str, Any]) -> str:
        return self.write("checkpoint.json", snapshot)

    def read_checkpoint(self, *, expected_sha256: str | None = None) -> dict[str, Any]:
        checkpoint = self.read("checkpoint.json")
        if expected_sha256 is not None:
            actual = self.content_hash(checkpoint)
            if actual != expected_sha256:
                raise ValueError("workspace checkpoint identity mismatch")
        return checkpoint

    def write_receipt(self, identity: str, payload: Mapping[str, Any]) -> None:
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        self.write(f"receipts/{digest}.json", payload)

    @staticmethod
    def content_hash(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, indent=2, allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
