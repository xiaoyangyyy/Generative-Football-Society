import json
from pathlib import Path

import pytest

from scripts.build_excellence_evidence_kit import (
    PROTOCOLS,
    build_archive,
    build_template_files,
    materialize_kit,
    verify_kit,
)
from scripts.product_validation_study import _validate_record as validate_product
from scripts.product_value_study import _validate_record as validate_value
from src.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]


def _copy_protocols(target: Path) -> None:
    for relative in PROTOCOLS.values():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())


def test_template_kit_is_deterministic_secret_free_and_zero_execution():
    first, files = build_archive(ROOT)
    second, _ = build_archive(ROOT)
    report = verify_kit(ROOT)
    assert first == second
    assert report["passed"] is True
    assert report["status"] == "passed_template_only_kit"
    assert report["template_count"] == 8
    assert "security/deepseek_api_credential_receipt.template.json" in files
    assert "security/github_classic_pat_receipt.template.json" in files
    security = json.loads(files["security/attestation.template.json"])
    assert security["schema_version"] == 2
    assert {
        row["incident_id"] for row in security["incidents"]
    } == {"deepseek_api_credential", "github_classic_pat"}
    assert all(report["checks"].values())
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert set(json.loads(files["manifest.json"])["files"]) == set(files) - {
        "manifest.json"
    }


def test_templates_fail_closed_in_real_participant_validators():
    files = build_template_files(ROOT)
    product = json.loads(
        files["product_validation/participant_record.template.json"]
    )
    value = json.loads(files["product_value/participant_record.template.json"])
    product_protocol = json.loads(
        (ROOT / PROTOCOLS["product"]).read_text(encoding="utf-8")
    )
    value_protocol = json.loads(
        (ROOT / PROTOCOLS["value"]).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="privacy allowlist"):
        validate_product(product, product_protocol)
    with pytest.raises(ValueError, match="privacy allowlist"):
        validate_value(value, value_protocol)


def test_materialization_is_confined_atomic_and_no_overwrite(tmp_path):
    _copy_protocols(tmp_path)
    relative = Path("build/evidence-kits/kit.zip")
    target = materialize_kit(relative, root=tmp_path)
    assert target.is_file()
    expected, _ = build_archive(tmp_path)
    assert target.read_bytes() == expected
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_kit(relative, root=tmp_path)
    assert materialize_kit(relative, root=tmp_path, overwrite=True) == target
    with pytest.raises(ValueError, match="must stay inside"):
        materialize_kit(Path("data/evaluation/fake-evidence.zip"), root=tmp_path)
    with pytest.raises(ValueError, match="must be a .zip"):
        materialize_kit(Path("build/evidence-kits/kit.json"), root=tmp_path)


def test_studio_evidence_kit_defaults_to_memory_only_audit(capsys):
    parser = build_parser()
    args = parser.parse_args([
        "--base-dir", str(ROOT), "studio", "evidence-kit",
    ])
    assert args.materialize is None
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert "materialized_path" not in report
