import json
import subprocess
import sys

from scripts.verify_container_images import DIGEST, verify_container_images
from scripts.verify_container_runtime import verify_container_runtime
from scripts.verify_local_runtime import verify_local_runtime
from scripts.verify_reproducibility_matrix import verify_reproducibility_matrix


def test_reference_matrix_has_two_current_hashed_profiles():
    report = verify_reproducibility_matrix()
    assert report["passed"]
    assert report["profile_count"] == 2
    assert {row["id"] for row in report["profiles"]} == {
        "linux-py312-x86_64-cpu",
        "windows-py313-x86_64-cpu",
    }
    assert all(row["package_count"] == 64 for row in report["profiles"])
    assert all(all(row["checks"].values()) for row in report["profiles"])
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False


def test_local_runtime_imports_all_exact_direct_dependencies():
    report = verify_local_runtime()
    assert report["passed"]
    assert len(report["packages"]) == 8
    assert all(row["version_matches"] for row in report["packages"].values())
    assert all(row["imported"] for row in report["packages"].values())
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0


def test_container_image_contract_uses_two_immutable_digests():
    report = verify_container_images()
    assert report["passed"]
    assert len(report["images"]) == 2
    assert all(
        DIGEST.fullmatch(reference.rsplit("@", 1)[1])
        for reference in report["images"].values()
    )
    assert report["verification_external_calls_made"] is False


def test_container_runtime_default_is_non_executing_preflight():
    report = verify_container_runtime(execute=False)
    assert report["passed"] is False
    assert report["image_built"] is False
    assert report["deployment_started"] is False
    assert report["provider_calls_made"] is False
    assert report["status"] in {"docker_unavailable", "awaiting_explicit_execute"}


def test_matrix_cli_is_standalone_and_read_only():
    completed = subprocess.run(
        [sys.executable, "scripts/verify_reproducibility_matrix.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["passed"] is True
    assert report["external_calls_made"] is False
