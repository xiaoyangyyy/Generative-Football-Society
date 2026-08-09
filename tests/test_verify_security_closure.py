import hashlib
import json

import pytest

from scripts.verify_security_closure import (
    ATTESTATION,
    PROTOCOL_PATH,
    protocol_report,
    verify_closure,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    docs = tmp_path / "docs/SECURITY_CREDENTIAL_CLOSURE.md"
    docs.parent.mkdir(parents=True)
    docs.write_text("closure guide\n", encoding="utf-8")
    evidence = tmp_path / "data/evaluation/security_closure_v1/receipt.json"
    _write_json(evidence, {"provider": "deepseek", "status": "revoked", "redacted": True})
    revision = "1" * 40
    attestation = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "provider": "deepseek",
        "credential_revoked": True,
        "revoked_at": "2026-08-10T12:00:00+08:00",
        "revocation_evidence": evidence.relative_to(tmp_path).as_posix(),
        "revocation_evidence_sha256": _sha(evidence),
        "evidence_contains_secret": False,
        "replacement_credential_generated": False,
        "replacement_storage": "not_generated",
        "repository_secret_scan": {
            "commit_sha": revision,
            "boundary_aware_secret_match_count": 0,
        },
        "signed_at": "2026-08-10T12:05:00+08:00",
        "attestation": ATTESTATION,
    }
    attestation_path = tmp_path / "data/evaluation/security_closure_v1/attestation.json"
    _write_json(attestation_path, attestation)
    return protocol_path, attestation_path, attestation, revision


def test_security_closure_protocol_is_registered_without_secret_storage():
    report = protocol_report()
    assert report["passed"] is True
    assert report["closure_complete"] is False
    assert report["boundary_aware_secret_match_count"] == 0
    assert report["credential_value_stored"] is False
    assert report["provider_calls_made"] is False


def test_redacted_revocation_attestation_verifies(tmp_path):
    protocol_path, attestation_path, _, revision = _fixture(tmp_path)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is True
    assert report["status"] == "verified_closed"
    assert report["closure_complete"] is True
    assert len(report["artifact_sha256"]) == 2
    assert all(report["checks"].values())


def test_repository_secret_match_fails_without_exposing_value(tmp_path):
    protocol_path, attestation_path, _, revision = _fixture(tmp_path)
    suspect = tmp_path / "suspect.bin"
    suspect.write_bytes(b"sk-" + b"x" * 24)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["boundary_aware_secret_match_count"] == 1
    assert report["checks"]["current_revision_has_zero_boundary_aware_matches"] is False
    assert "x" * 24 not in json.dumps(report)


def test_unredacted_or_escaping_evidence_fails_closed(tmp_path):
    protocol_path, attestation_path, attestation, revision = _fixture(tmp_path)
    attestation["evidence_contains_secret"] = True
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["checks"][
        "redacted_receipt_is_confined_and_content_addressed"
    ] is False

    outside = tmp_path / "outside.json"
    _write_json(outside, {"redacted": True})
    attestation["evidence_contains_secret"] = False
    attestation["revocation_evidence"] = outside.relative_to(tmp_path).as_posix()
    attestation["revocation_evidence_sha256"] = _sha(outside)
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False


def test_scan_attestation_must_match_current_commit(tmp_path):
    protocol_path, attestation_path, attestation, revision = _fixture(tmp_path)
    attestation["repository_secret_scan"]["commit_sha"] = "2" * 40
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["checks"]["current_revision_has_zero_boundary_aware_matches"] is False


def test_attestation_path_cannot_escape_security_evidence_root(tmp_path):
    protocol_path, _, attestation, revision = _fixture(tmp_path)
    outside = tmp_path / "attestation.json"
    _write_json(outside, attestation)
    with pytest.raises(ValueError, match="security evidence root"):
        verify_closure(
            outside,
            root=tmp_path,
            protocol_path=protocol_path,
            revision=revision,
        )
