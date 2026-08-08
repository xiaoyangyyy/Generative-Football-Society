import pytest

import hashlib

from src.infrastructure import (
    FileLease, LeaseUnavailable, file_sha256, portable_text_hash_matches,
    verify_artifact_manifest,
)


def test_file_lease_is_exclusive_and_recoverable(tmp_path):
    path = tmp_path / "resource.lock"
    first = FileLease(path).acquire()
    assert FileLease.is_held(path)
    with pytest.raises(LeaseUnavailable):
        FileLease(path).acquire()
    first.release()
    assert not FileLease.is_held(path)
    with FileLease(path) as second:
        assert second.held
    assert not second.held


def test_portable_text_identity_accepts_only_reversible_newline_conversion(tmp_path):
    artifact = tmp_path / "manifest.json"
    lf = b'{\n  "release": 1\n}\n'
    artifact.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert portable_text_hash_matches(artifact, hashlib.sha256(lf).hexdigest())
    assert not portable_text_hash_matches(
        artifact, hashlib.sha256(b'{"release":2}').hexdigest(),
    )
    assert file_sha256(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_artifact_manifest_rejects_root_escape_and_tampering(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("sealed\n", encoding="utf-8")
    valid = {"artifacts": [{
        "path": "artifact.txt", "sha256": file_sha256(artifact),
    }]}
    assert verify_artifact_manifest(tmp_path, valid)["ok"]
    escaped = {"artifacts": [{"path": "../outside", "sha256": "0" * 64}]}
    assert not verify_artifact_manifest(tmp_path, escaped)["ok"]
    artifact.write_text("tampered\n", encoding="utf-8")
    assert not verify_artifact_manifest(tmp_path, valid)["ok"]
