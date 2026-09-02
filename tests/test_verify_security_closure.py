import hashlib
import json
from types import SimpleNamespace

import pytest

import scripts.verify_security_closure as security_closure
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
    evidence_root = tmp_path / "data/evaluation/security_closure_v2"
    receipts = {}
    for provider in ("deepseek", "github"):
        receipt = evidence_root / f"{provider}_receipt.json"
        _write_json(
            receipt,
            {"provider": provider, "status": "revoked", "redacted": True},
        )
        receipts[provider] = receipt
    revision = "1" * 40
    attestation = {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "incidents": [
            {
                "incident_id": incident["incident_id"],
                "provider": incident["provider"],
                "credential_revoked": True,
                "revoked_at": "2026-08-10T12:00:00+08:00",
                "revocation_evidence": receipts[
                    incident["provider"]
                ].relative_to(tmp_path).as_posix(),
                "revocation_evidence_sha256": _sha(
                    receipts[incident["provider"]]
                ),
                "evidence_contains_secret": False,
                "replacement_credential_generated": False,
                "replacement_storage": "not_generated",
            }
            for incident in protocol["incidents"]
        ],
        "repository_secret_scan": {
            "commit_sha": revision,
            "secret_pattern_version": "gfs_known_credentials_v2",
            "boundary_aware_secret_match_count": 0,
            "boundary_aware_secret_matches_by_family": {
                "openai_compatible_api_key": 0,
                "github_classic_pat": 0,
                "github_fine_grained_pat": 0,
            },
        },
        "signed_at": "2026-08-10T12:05:00+08:00",
        "attestation": ATTESTATION,
    }
    attestation_path = evidence_root / "attestation.json"
    _write_json(attestation_path, attestation)
    return protocol_path, attestation_path, attestation, revision


def test_security_closure_protocol_is_registered_without_secret_storage():
    report = protocol_report()
    assert report["passed"] is True
    assert report["closure_complete"] is False
    assert report["known_incident_count"] == 2
    assert report["revoked_incident_count"] == 0
    assert report["known_incidents_complete"] is False
    assert report["incident_ids"] == [
        "deepseek_api_credential", "github_classic_pat",
    ]
    assert report["boundary_aware_secret_match_count"] == 0
    assert report["origin_has_embedded_credential"] is False
    assert report["checks"][
        "origin_remote_contains_no_embedded_credential"
    ] is True
    assert all(
        count == 0
        for count in report[
            "boundary_aware_secret_matches_by_family"
        ].values()
    )
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
    assert report["known_incidents_complete"] is True
    assert report["revoked_incident_count"] == 2
    assert report["providers"] == ["deepseek", "github"]
    assert len(report["artifact_sha256"]) == 3
    assert all(report["checks"].values())


@pytest.mark.parametrize("prefix,family", [
    (b"sk-", "openai_compatible_api_key"),
    (b"ghp_", "github_classic_pat"),
    (b"github_pat_", "github_fine_grained_pat"),
])
def test_repository_secret_match_fails_without_exposing_value(
    tmp_path, prefix, family,
):
    protocol_path, attestation_path, _, revision = _fixture(tmp_path)
    suspect = tmp_path / "suspect.bin"
    secret = prefix + b"x" * 32
    suspect.write_bytes(secret)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["boundary_aware_secret_match_count"] == 1
    assert report[
        "boundary_aware_secret_matches_by_family"
    ][family] == 1
    assert report["checks"]["current_revision_has_zero_boundary_aware_matches"] is False
    assert secret.decode() not in json.dumps(report)


def test_partial_incident_closure_fails_closed(tmp_path):
    protocol_path, attestation_path, attestation, revision = _fixture(tmp_path)
    attestation["incidents"].pop()
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["known_incidents_complete"] is False
    assert report["revoked_incident_count"] == 1
    assert report["checks"][
        "all_known_incidents_are_present_exactly_once"
    ] is False


def test_unredacted_or_escaping_evidence_fails_closed(tmp_path):
    protocol_path, attestation_path, attestation, revision = _fixture(tmp_path)
    incident = attestation["incidents"][0]
    incident["evidence_contains_secret"] = True
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["checks"][
        "every_redacted_receipt_is_distinct_confined_and_content_addressed"
    ] is False

    outside = tmp_path / "outside.json"
    _write_json(outside, {"redacted": True})
    incident["evidence_contains_secret"] = False
    incident["revocation_evidence"] = outside.relative_to(tmp_path).as_posix()
    incident["revocation_evidence_sha256"] = _sha(outside)
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False


def test_incidents_cannot_reuse_one_receipt(tmp_path):
    protocol_path, attestation_path, attestation, revision = _fixture(tmp_path)
    first = attestation["incidents"][0]
    second = attestation["incidents"][1]
    second["revocation_evidence"] = first["revocation_evidence"]
    second["revocation_evidence_sha256"] = first[
        "revocation_evidence_sha256"
    ]
    _write_json(attestation_path, attestation)
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["checks"][
        "every_redacted_receipt_is_distinct_confined_and_content_addressed"
    ] is False


def test_embedded_origin_credential_fails_without_exposing_url(
    tmp_path, monkeypatch,
):
    protocol_path, attestation_path, _, revision = _fixture(tmp_path)
    monkeypatch.setattr(
        security_closure,
        "_origin_has_embedded_credential",
        lambda root: True,
    )
    report = verify_closure(
        attestation_path,
        root=tmp_path,
        protocol_path=protocol_path,
        revision=revision,
    )
    assert report["passed"] is False
    assert report["origin_has_embedded_credential"] is True
    assert report["checks"][
        "origin_remote_contains_no_embedded_credential"
    ] is False


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/example/project.git", False),
    ("https://username@github.com/example/project.git", False),
    ("git@github.com:example/project.git", False),
    ("https://username:password@github.com/example/project.git", True),
    (
        "https://" + "ghp_" + "x" * 32
        + "@github.com/example/project.git",
        True,
    ),
])
def test_origin_credential_detection_is_bounded(
    tmp_path, monkeypatch, url, expected,
):
    monkeypatch.setattr(
        security_closure.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=url,
        ),
    )
    assert security_closure._origin_has_embedded_credential(tmp_path) is expected


def test_protocol_report_cannot_pass_with_embedded_origin(
    tmp_path, monkeypatch,
):
    protocol_path, _, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        security_closure,
        "_origin_has_embedded_credential",
        lambda root: True,
    )
    report = protocol_report(root=tmp_path, protocol_path=protocol_path)
    assert report["passed"] is False
    assert report["origin_has_embedded_credential"] is True
    assert report["checks"][
        "origin_remote_contains_no_embedded_credential"
    ] is False


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
