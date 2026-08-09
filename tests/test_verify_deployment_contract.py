import hashlib
import json

import scripts.verify_deployment_contract as deployment_contract
from scripts.verify_deployment_contract import verify_deployment_contract


def test_deployment_contract_is_secure_and_honest_about_unbuilt_state():
    report = verify_deployment_contract()
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["image_built"] is False
    assert report["target_python_dependencies_hash_locked"] is True
    assert report["container_images_digest_pinned"] is True
    assert report["deployment_started"] is False
    assert report["external_calls_made"] is False
    if not report["docker_available"]:
        assert report["status"] == "passed_static_contract_unbuilt"
        assert report["docker_compose_validation"] == "docker_unavailable"
    assert any("not been built" in item for item in report["limitations"])


def test_runtime_evidence_requires_current_artifacts_and_zero_compute(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "Dockerfile"
    artifact.write_text("FROM scratch\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report = {
        "schema_version": 2,
        "status": "passed",
        "passed": True,
        "image_built": True,
        "deployment_started": True,
        "source_revision": "1" * 40,
        "image_id": "sha256:" + "2" * 64,
        "provider_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "checks": {
            check: True for check in deployment_contract.EXPECTED_RUNTIME_CHECKS
        },
        "artifact_sha256": {"Dockerfile": digest},
        "registry_access_attempted": True,
    }
    evidence = tmp_path / "data/evaluation/container_runtime_verification_v1.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(deployment_contract, "ROOT", tmp_path)
    verified, loaded = deployment_contract._verified_container_runtime()
    assert verified is True
    assert loaded["source_revision"] == "1" * 40

    artifact.write_text("FROM changed\n", encoding="utf-8")
    verified, _ = deployment_contract._verified_container_runtime()
    assert verified is False

    report["artifact_sha256"] = {"../../outside": "0" * 64}
    evidence.write_text(json.dumps(report), encoding="utf-8")
    verified, _ = deployment_contract._verified_container_runtime()
    assert verified is False
