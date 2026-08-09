import hashlib
import json
import subprocess
import sys
import zipfile

import scripts.verify_data_release as release_verifier
from scripts.build_data_release_archive import (
    MANIFEST,
    approved_paths,
    archive_bytes,
    expected_manifest,
)


EXPECTED_ARCHIVE_SHA256 = (
    "df25a33dea93902bca965ba725ee72c67901015b5ed8808585a2e639d8dc7d79"
)


def test_licensed_data_manifest_is_exact_and_deterministic():
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = expected_manifest()
    assert stored == expected
    assert stored["file_count"] == 37
    assert stored["excluded_source_ids"] == [
        "metrica_sample_data",
        "skillcorner_open_data",
        "statsbomb_open_data",
        "transfermarkt_dataset_mirror",
    ]
    assert stored["deterministic_archive"]["sha256"] == EXPECTED_ARCHIVE_SHA256
    payload = archive_bytes(approved_paths())
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_ARCHIVE_SHA256


def test_data_release_verifier_rejects_manifest_tampering(tmp_path, monkeypatch):
    tampered = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tampered["entries"][0]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(release_verifier, "MANIFEST", path)
    report = release_verifier.verify_data_release()
    assert report["passed"] is False
    assert report["checks"]["manifest_is_current_and_content_addressed"] is False


def test_materialized_archive_contains_only_manifested_files(tmp_path):
    output = tmp_path / "gfs-idsse-data-supplement-v1.zip"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_data_release_archive.py",
            "--materialize",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert result["sha256"] == EXPECTED_ARCHIVE_SHA256
    assert hashlib.sha256(output.read_bytes()).hexdigest() == EXPECTED_ARCHIVE_SHA256
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [row["path"] for row in stored["entries"]]
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0)
            and info.compress_type == zipfile.ZIP_STORED
            for info in archive.infolist()
        )


def test_data_release_verifier_is_zero_execution_and_source_bounded():
    report = release_verifier.verify_data_release()
    assert report["passed"] is True
    assert report["approved_source_count"] == 1
    assert report["excluded_source_count"] == 4
    assert all(report["checks"].values())
    assert "scripts/verify_data_release.py" in report["artifact_sha256"]
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["formal_experiment_executed"] is False
