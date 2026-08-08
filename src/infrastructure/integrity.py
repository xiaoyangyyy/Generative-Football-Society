"""Artifact identity primitives shared across product and research layers."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_text_hash_matches(path: str | Path, expected: str) -> bool:
    """Match exact bytes or Git's reversible LF/CRLF text conversion only."""
    artifact = Path(path)
    if not artifact.is_file() or len(str(expected)) != 64:
        return False
    raw = artifact.read_bytes()
    candidates = [raw]
    if b"\0" not in raw:
        lf = raw.replace(b"\r\n", b"\n")
        candidates.extend((lf, lf.replace(b"\n", b"\r\n")))
    return any(
        hashlib.sha256(candidate).hexdigest() == expected
        for candidate in candidates
    )


def verify_artifact_manifest(
    root: str | Path, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a non-empty artifact set and reject paths outside its root."""
    root_path = Path(root).resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return {"ok": False, "artifacts": 0, "failures": [
            {"reason": "artifact_manifest_is_empty"},
        ]}
    failures: list[dict[str, Any]] = []
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            failures.append({"reason": "invalid_artifact_entry"})
            continue
        relative = str(entry.get("path") or "")
        expected = str(entry.get("sha256") or "")
        candidate = (root_path / relative).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError:
            failures.append({"path": relative, "reason": "path_outside_root"})
            continue
        if not portable_text_hash_matches(candidate, expected):
            failures.append({
                "path": relative,
                "reason": "missing_or_identity_mismatch",
            })
    return {
        "ok": not failures,
        "artifacts": len(artifacts),
        "failures": failures,
    }
