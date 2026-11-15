import json

import pytest

from gfs_studio.plugins.evidence.receipts.store import EvidenceStore


def test_evidence_store_verifies_identity_and_issues_immutable_receipt(tmp_path):
    frozen = tmp_path / "evidence" / "frozen"
    frozen.mkdir(parents=True)
    document = {"schema_version": 1, "result": "negative"}
    encoded = json.dumps(document, indent=2).encode()
    (frozen / "result.json").write_bytes(encoded)
    import hashlib

    digest = hashlib.sha256(encoded).hexdigest()
    store = EvidenceStore(tmp_path)
    assert store.read_frozen("result.json") == document
    assert store.verify_current_identity("result.json", digest)
    assert not store.verify_current_identity("result.json", "0" * 64)
    with pytest.raises(ValueError, match="escapes"):
        store.read_frozen("../secret.json")

    receipt = store.issue_receipt("receipt-1", document)
    assert receipt.content_sha256 == hashlib.sha256(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(FileExistsError):
        store.issue_receipt("receipt-1", document)
