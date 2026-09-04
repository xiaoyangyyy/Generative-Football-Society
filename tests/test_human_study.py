import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.product_validation_study import PROTOCOL_PATH as TARGET_PROTOCOL
from scripts.product_value_study import (
    CASE_PACK_PATH,
    PROTOCOL_PATH as VALUE_PROTOCOL,
)
from src.product.human_study import (
    register_participant,
    validate_registry,
    verify_content_addressed_archive,
)


def _register_target(tmp_path, index):
    return register_participant(
        project_root=tmp_path,
        protocol_path=TARGET_PROTOCOL,
        registry_path=tmp_path / "registry.json",
        records_path=tmp_path / "records.jsonl",
        participant_id=f"P-{index:012d}",
        target_role=(
            "football_analyst",
            "research_engineer",
            "product_operator",
        )[index % 3],
        moderator_id="M-00000001",
        consent_recorded=True,
    )


def test_concurrent_registration_is_serialized_without_lost_entries(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda index: _register_target(tmp_path, index), range(24)))
    report = validate_registry(
        protocol_path=TARGET_PROTOCOL,
        registry=tmp_path / "registry.json",
    )
    assert all(result["created"] for result in results)
    assert report["registration_count"] == 24
    assert report["role_counts"] == {
        "football_analyst": 8,
        "product_operator": 8,
        "research_engineer": 8,
    }


def test_registration_refuses_after_measurement_and_conflicting_retry(tmp_path):
    original = _register_target(tmp_path, 1)["registration"]
    with pytest.raises(ValueError, match="conflicts with frozen data"):
        register_participant(
            project_root=tmp_path,
            protocol_path=TARGET_PROTOCOL,
            registry_path=tmp_path / "registry.json",
            records_path=tmp_path / "records.jsonl",
            participant_id=original["participant_id"],
            target_role="product_operator",
            moderator_id=original["moderator_id"],
            consent_recorded=True,
        )
    (tmp_path / "records.jsonl").write_text("{}\n", encoding="utf-8")
    retry = _register_target(tmp_path, 1)
    assert retry["created"] is False
    assert retry["registration"] == original
    with pytest.raises(ValueError, match="after measurement records exist"):
        _register_target(tmp_path, 2)


def test_registry_detects_protocol_case_pack_and_allocation_tampering(tmp_path):
    register_participant(
        project_root=tmp_path,
        protocol_path=VALUE_PROTOCOL,
        registry_path=tmp_path / "value-registry.json",
        records_path=tmp_path / "value-records.jsonl",
        participant_id="participant-00000001",
        target_role="football_analyst",
        moderator_id="moderator-12345678",
        consent_recorded=True,
        case_pack_manifest_path=CASE_PACK_PATH,
    )
    registry = json.loads((tmp_path / "value-registry.json").read_text(encoding="utf-8"))
    registry["registrations"][0]["sequence"] = (
        "BA" if registry["registrations"][0]["sequence"] == "AB" else "AB"
    )
    with pytest.raises(ValueError, match="frozen allocation"):
        validate_registry(
            protocol_path=VALUE_PROTOCOL,
            registry=registry,
            case_pack_manifest_path=CASE_PACK_PATH,
        )

    registry = json.loads((tmp_path / "value-registry.json").read_text(encoding="utf-8"))
    registry["case_pack_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="case-pack identity mismatch"):
        validate_registry(
            protocol_path=VALUE_PROTOCOL,
            registry=registry,
            case_pack_manifest_path=CASE_PACK_PATH,
        )

    registry = json.loads((tmp_path / "value-registry.json").read_text(encoding="utf-8"))
    registry["registrations"][0]["registered_at"] = "2020-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="registration_id is invalid"):
        validate_registry(
            protocol_path=VALUE_PROTOCOL,
            registry=registry,
            case_pack_manifest_path=CASE_PACK_PATH,
        )


def test_archive_rejects_duplicate_missing_and_mismatched_content(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    payload = b"evidence"
    digest = hashlib.sha256(payload).hexdigest()
    (archive / digest).write_bytes(payload)
    streamed = verify_content_addressed_archive(archive, [digest])
    assert streamed["verified_artifacts"] == 1
    assert streamed["verified_bytes"] == {}
    report = verify_content_addressed_archive(
        archive,
        [digest],
        retain_verified_bytes=True,
        max_bytes_per_artifact=len(payload),
    )
    assert report["verified_bytes"][digest] == payload
    with pytest.raises(ValueError, match="size limit"):
        verify_content_addressed_archive(
            archive,
            [digest],
            retain_verified_bytes=True,
            max_bytes_per_artifact=len(payload) - 1,
        )
    with pytest.raises(ValueError, match="unique"):
        verify_content_addressed_archive(archive, [digest, digest])
    with pytest.raises(ValueError, match="missing"):
        verify_content_addressed_archive(archive, ["0" * 64])
    (archive / digest).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_content_addressed_archive(archive, [digest])


def test_registry_json_contains_no_direct_identity_fields(tmp_path):
    _register_target(tmp_path, 7)
    registry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    serialized = json.dumps(registry)
    assert "name" not in serialized
    assert "email" not in serialized
    assert "organization" not in serialized
    assert "ip_address" not in serialized
