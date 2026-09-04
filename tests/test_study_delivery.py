import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.cli import build_parser
from src.product.study_delivery import (
    CASE_PACK_RELATIVE,
    FORBIDDEN_DELIVERY_FIELDS,
    PROTOCOL_RELATIVE,
    SCORING_SEAL_RELATIVE,
    ProductValueStudyDelivery,
)


ROOT = Path(__file__).resolve().parents[1]


def _copy_authority(target: Path) -> None:
    for relative in (
        PROTOCOL_RELATIVE,
        CASE_PACK_RELATIVE,
        SCORING_SEAL_RELATIVE,
    ):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def _register(service: ProductValueStudyDelivery, index: int = 1):
    return service.register(
        participant_id=f"participant-{index:08d}",
        target_role="football_analyst",
        moderator_id="moderator-12345678",
        consent_recorded=True,
    )["registration"]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_blinded_packet_is_deterministic_identity_bound_and_key_free(tmp_path):
    _copy_authority(tmp_path)
    service = ProductValueStudyDelivery(tmp_path)
    registration = _register(service)
    first = service.packet(registration["registration_id"])
    second = service.packet(registration["registration_id"])

    assert first == second
    assert service.packet_bytes(registration["registration_id"]) == service.packet_bytes(
        registration["registration_id"]
    )
    assert first["registration_id"] == registration["registration_id"]
    assert first["delivery_boundary"]["scoring_material_included"] is False
    assert not FORBIDDEN_DELIVERY_FIELDS.intersection(_all_keys(first))
    assert "moderator_id" not in set(_all_keys(first))
    expected = (
        [("gfs_studio", "case_pack_alpha"), ("manual_baseline", "case_pack_beta")]
        if registration["sequence"] == "AB"
        else [("manual_baseline", "case_pack_alpha"), ("gfs_studio", "case_pack_beta")]
    )
    assert [
        (row["condition"], row["case_pack"]) for row in first["conditions"]
    ] == expected


def test_status_is_aggregate_only_and_materialization_is_confined(tmp_path):
    _copy_authority(tmp_path)
    service = ProductValueStudyDelivery(tmp_path)
    registration = _register(service)
    status = service.status()
    serialized = json.dumps(status)

    assert status["registration_count"] == 1
    assert status["role_counts"]["football_analyst"] == 1
    assert status["participant_identifiers_exposed"] is False
    assert status["measurement_records_present"] is False
    assert status["participants_observed"] == 0
    assert status["registration_open"] is True
    assert registration["participant_id"] not in serialized
    assert registration["registration_id"] not in serialized
    target = service.materialize_packet(registration["registration_id"])
    assert target.is_file()
    assert target.parent == (tmp_path / "build/study-delivery/product-value-v1")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(
        service.packet_bytes(registration["registration_id"])
    ).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        service.materialize_packet(registration["registration_id"])
    with pytest.raises(ValueError, match="must stay"):
        service.materialize_packet(
            registration["registration_id"],
            tmp_path / "outside.json",
        )
    with pytest.raises(ValueError, match="must be a .json"):
        service.materialize_packet(
            registration["registration_id"],
            service.delivery_root / "packet.txt",
        )

    service.records_path.parent.mkdir(parents=True, exist_ok=True)
    service.records_path.write_text("{}\n", encoding="utf-8")
    measured_status = service.status()
    assert measured_status["measurement_records_present"] is True
    assert measured_status["participants_observed"] is None
    assert measured_status["registration_open"] is False


def test_materialization_rejects_linked_delivery_components(tmp_path):
    _copy_authority(tmp_path)
    service = ProductValueStudyDelivery(tmp_path)
    registration = _register(service)
    external = tmp_path / "external"
    external.mkdir()
    build = tmp_path / "build"
    try:
        build.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    with pytest.raises(ValueError, match="symbolic link"):
        service.materialize_packet(registration["registration_id"])
    assert list(external.iterdir()) == []


def test_authority_rejects_case_or_scoring_seal_tampering(tmp_path):
    _copy_authority(tmp_path)
    service = ProductValueStudyDelivery(tmp_path)
    cases = json.loads(service.case_pack_path.read_text(encoding="utf-8"))
    cases["packs"][0]["scoring_key"] = {"leak": True}
    service.case_pack_path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError, match="scoring material"):
        service.status()

    second = tmp_path / "second"
    _copy_authority(second)
    service = ProductValueStudyDelivery(second)
    seal = json.loads(service.scoring_seal_path.read_text(encoding="utf-8"))
    seal["scoring_keys"]["case_pack_alpha"][
        "next_evidence_action"
    ] = "automatic_production_promotion"
    service.scoring_seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="scoring seal"):
        service.status()


def test_value_study_cli_connects_status_registration_and_packet(tmp_path, capsys):
    _copy_authority(tmp_path)
    parser = build_parser()
    status = parser.parse_args([
        "--base-dir", str(tmp_path), "studio", "value-study", "status",
    ])
    assert status.func(status) == 0
    assert json.loads(capsys.readouterr().out)["registration_count"] == 0

    register = parser.parse_args([
        "--base-dir", str(tmp_path), "studio", "value-study", "register",
        "--participant-id", "participant-00000001",
        "--target-role", "football_analyst",
        "--moderator-id", "moderator-12345678",
        "--confirm-consent",
    ])
    assert register.func(register) == 0
    registration = json.loads(capsys.readouterr().out)["registration"]
    packet = parser.parse_args([
        "--base-dir", str(tmp_path), "studio", "value-study", "packet",
        "--registration-id", registration["registration_id"],
    ])
    assert packet.func(packet) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blinded_packet_materialized"
    assert result["scoring_material_exposed"] is False
    assert (tmp_path / result["path"]).is_file()
