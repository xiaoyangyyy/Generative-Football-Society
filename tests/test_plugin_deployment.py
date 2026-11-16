import hashlib
import json
import shutil

from scripts.verify_deployment import ROOT, RUNTIME_REPORT, verify
from scripts.verify_plugin_containers import ARTIFACTS


def test_plugin_deployment_runs_only_studio_with_embedded_core():
    report = verify(ROOT)
    assert report["static_verified"], report["errors"]
    assert report["runtime_verified"] is False


def test_runtime_evidence_is_executed_and_deployment_hash_bound(tmp_path):
    for relative in ARTIFACTS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, destination)
    runtime = {
        "schema_version": 1,
        "verification": "gfs_plugin_container_runtime_v1",
        "source_revision": "1" * 40,
        "passed": True,
        "checks": {"real_container_passed": True},
        "artifact_sha256": {
            relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
            for relative in ARTIFACTS
        },
    }
    report_path = tmp_path / RUNTIME_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(runtime), encoding="utf-8")
    assert verify(tmp_path)["runtime_verified"] is True
    with (tmp_path / "deployment" / "compose.yaml").open("a", encoding="utf-8") as file:
        file.write("\n# changed deployment\n")
    assert verify(tmp_path)["runtime_verified"] is False
