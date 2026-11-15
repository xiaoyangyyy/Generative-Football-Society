from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    identity: str
    content_sha256: str
    path: str


@runtime_checkable
class EvidenceStore(Protocol):
    root: Path

    def read_frozen(self, relative: str) -> dict: ...

    def verify_current_identity(self, relative: str, expected_sha256: str) -> bool: ...

    def issue_receipt(self, identity: str, document: dict) -> EvidenceReceipt: ...

    def read_receipt(self, identity: str, expected_document: dict | None = None) -> EvidenceReceipt: ...


@runtime_checkable
class IdentityRegistry(Protocol):
    def register(self, relative: str, expected_sha256: str) -> dict: ...

    def verify(self, relative: str) -> dict: ...

    def list(self) -> list[dict]: ...


@runtime_checkable
class ReleaseGate(Protocol):
    def evaluate(self) -> dict: ...
