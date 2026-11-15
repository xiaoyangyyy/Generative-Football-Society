import hashlib
import json

from gfs_studio.plugins.world_model.frame.service import FrameModelService


def test_frame_manifest_verifies_real_artifact_content(tmp_path):
    artifact = tmp_path / "data" / "frames" / "match.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frame-data")
    manifest = {
        "candidate": "test",
        "matches": [{
            "file": "data/frames/match.npz",
            "match_id": "match-1",
            "provider": "test",
            "sha256": hashlib.sha256(b"frame-data").hexdigest(),
        }],
    }
    manifest_path = tmp_path / "data" / "frame-world" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    service = FrameModelService(tmp_path)
    assert service.manifest()["artifact_verified"] is True
    artifact.write_bytes(b"tampered")
    status = service.manifest()
    assert status["artifact_verified"] is False
    assert status["invalid_artifacts"] == ["match-1"]
    assert status["scope"] == "verified_artifact_catalog_not_frame_inference"
