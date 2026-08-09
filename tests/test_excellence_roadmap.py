import json
from pathlib import Path

from src.product.control_plane import ProductControlPlane


ROOT = Path(__file__).resolve().parents[1]


def test_excellence_tracks_are_honest_bounded_and_evidence_backed():
    roadmap = json.loads(
        (ROOT / "data/evaluation/excellence_roadmap_v1.json").read_text(
            encoding="utf-8",
        )
    )
    assert roadmap["security"]["exposed_key_status"] == "revocation_required_by_user"
    assert not roadmap["security"]["repository_storage_authorized"]
    for name, gates in roadmap["tracks"].items():
        assert sum(gate["weight"] for gate in gates) == 100, name
        current_score = sum(gate["current_points"] for gate in gates)
        assert current_score == roadmap["current_assessment"][name]["score"]
        assert current_score >= roadmap["baseline_assessment"][name]["score"]
        for gate in gates:
            assert 0 <= gate["current_points"] <= gate["weight"]
            if gate["current_points"]:
                assert gate["evidence"], gate["id"]
                assert all((ROOT / path).exists() for path in gate["evidence"])
            if gate["status"] == "missing":
                assert gate["current_points"] == 0
            if gate["status"] == "verified":
                assert gate["evidence"]
                assert all((ROOT / path).exists() for path in gate["evidence"])


def test_excellence_phase_graph_is_ordered_and_not_falsely_complete():
    roadmap = json.loads(
        (ROOT / "data/evaluation/excellence_roadmap_v1.json").read_text(
            encoding="utf-8",
        )
    )
    phases = {phase["id"]: phase for phase in roadmap["phases"]}
    assert roadmap["current_phase"] == "S0"
    assert phases["S0"]["status"] == "in_progress"
    assert all(
        dependency in phases
        for phase in phases.values()
        for dependency in phase["depends_on"]
    )
    assert not any(phase["status"] == "completed" for phase in phases.values())


def test_control_plane_exposes_scores_and_security_action():
    snapshot = ProductControlPlane(ROOT).snapshot()["excellence"]
    assert snapshot["tracks"]["product"]["score"] == 82
    assert snapshot["tracks"]["product"]["baseline_score"] == 58
    assert snapshot["tracks"]["academic"]["score"] == 69
    assert snapshot["security_action"] == "revocation_required_by_user"


def test_control_plane_unifies_product_and_paper_release_gates():
    release = ProductControlPlane(ROOT).snapshot()["release"]
    assert release["code_ready"] is True
    assert release["release_ready"] is False
    assert release["passed_gate_count"] == 4
    assert release["open_gate_count"] == 7
    assert release["next_action"] == "local_runtime"
    assert release["scores"] == {"product": 82, "academic": 69}
    gates = {gate["id"]: gate for gate in release["gates"]}
    assert gates["paper_package"]["passed"] is True
    assert gates["target_hash_lock"]["passed"] is True
    assert gates["cyclonedx_sbom"]["passed"] is True
    assert gates["container_build"]["passed"] is False
    assert gates["confirmatory_results"]["passed"] is False
    assert release["external_calls_made"] is False


def test_control_plane_rejects_stale_or_escaping_report_hashes(tmp_path):
    control = ProductControlPlane(tmp_path)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("current", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert control._artifact_hashes_current({"artifact.txt": digest})
    assert not control._artifact_hashes_current({"artifact.txt": "0" * 64})
    assert not control._artifact_hashes_current({"../outside.txt": digest})
