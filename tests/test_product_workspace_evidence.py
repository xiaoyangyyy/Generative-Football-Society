import src.product.workspace as workspace_module
from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.workspace_evidence import (
    build_workspace_evidence,
    build_workspace_readiness,
)
from scripts.verify_product_recovery import CODE_IDENTITY_FILES


def test_workspace_evidence_repository_fails_closed_without_authority(tmp_path):
    evidence = build_workspace_evidence(
        root=tmp_path, artifact_path=workspace_module._artifact_path,
    )
    readiness = build_workspace_readiness(
        root=tmp_path,
        mode="stable",
        evidence=evidence,
        artifact_path=workspace_module._artifact_path,
    )

    assert evidence["stable_release"] is None
    assert evidence["world_model_checkpoint"] is None
    assert evidence["production_promotion_ready"] is False
    assert readiness["ready"] is False
    assert readiness["blockers"] == [
        "stable_release_available",
        "stable_release_identity_verified",
        "stable_release_artifacts_verified",
    ]
    assert "src/product/workspace_evidence.py" in CODE_IDENTITY_FILES


def test_product_workspace_evidence_facade_forwards_exact_dependencies(
    monkeypatch, tmp_path,
):
    workspace = ProductWorkspace(
        tmp_path, StudioConfig(name="Evidence facade"),
    )
    observed = {}

    def fake_builder(**kwargs):
        observed.update(kwargs)
        return {"evidence": True}

    monkeypatch.setattr(
        workspace_module, "build_workspace_evidence", fake_builder,
    )

    assert workspace.evidence() == {"evidence": True}
    assert observed == {
        "root": tmp_path.resolve(),
        "artifact_path": workspace_module._artifact_path,
    }


def test_product_workspace_readiness_facade_uses_current_evidence_once(
    monkeypatch, tmp_path,
):
    workspace = ProductWorkspace(
        tmp_path, StudioConfig(name="Readiness facade", mode="research"),
    )
    evidence = {"identity": "current"}
    calls = {"evidence": 0}
    observed = {}

    def current_evidence():
        calls["evidence"] += 1
        return evidence

    def fake_builder(**kwargs):
        observed.update(kwargs)
        return {"ready": True}

    monkeypatch.setattr(workspace, "evidence", current_evidence)
    monkeypatch.setattr(
        workspace_module, "build_workspace_readiness", fake_builder,
    )

    assert workspace.readiness() == {"ready": True}
    assert calls["evidence"] == 1
    assert observed == {
        "root": tmp_path.resolve(),
        "mode": "research",
        "evidence": evidence,
        "artifact_path": workspace_module._artifact_path,
    }
