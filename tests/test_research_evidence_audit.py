import hashlib

from scripts.audit_research_evidence import build_report
from scripts.verify_release_manifest import verify_artifacts
from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.simulation.runtime import SimulationConfig


def test_frozen_external_evidence_and_release_boundary_are_reproducible():
    report = build_report()
    assert report["external_benchmark"]["team_matches"] == 128
    assert report["external_benchmark"]["frozen_metric_count"] == 15
    assert report["gates"]["external_benchmark_frozen"]
    assert report["gates"]["failed_pass_labels_cross_provider"]
    assert report["gates"]["shot_and_goal_holdout"]
    assert report["gates"]["continuous_state_cross_provider"]
    assert report["release_boundary"]["active"] == "7.0.0"
    assert report["release_boundary"]["valid"]
    assert report["release_boundary"]["normalized_text_artifact_count"] == 16


def test_advanced_layers_cannot_claim_production_readiness():
    report = build_report()
    assert report["research_data_ready"]
    assert not report["production_promotion_ready"]
    assert report["production_blockers"]
    assert report["module_policy"]["advanced_world_model"]["status"] == (
        "research_only_default_off"
    )
    assert report["module_policy"]["llm_cognition"]["status"] == (
        "research_only_default_off"
    )
    assert report["ablation"]["evidence_grade"] == "diagnostic_only"


def test_advanced_layers_are_default_off_in_all_public_config_constructors():
    direct = SimulationConfig()
    mapped = SimulationConfig.from_mapping({})
    cognitive = CognitiveMatchConfig.from_mapping({})
    assert not direct.world_model_enabled
    assert not direct.cognitive_enabled
    assert not mapped.world_model_enabled
    assert not mapped.cognitive_enabled
    assert not cognitive.enabled


def test_release_verifier_accepts_only_reversible_text_newline_changes(tmp_path):
    expected = b'{\r\n  "value": 1\r\n}\r\n'
    path = tmp_path / "artifact.json"
    path.write_bytes(expected.replace(b"\r\n", b"\n"))
    manifest = {"artifacts": [{
        "path": "artifact.json",
        "sha256": hashlib.sha256(expected).hexdigest(),
    }]}
    failures, normalized = verify_artifacts(manifest, root=tmp_path)
    assert not failures
    assert normalized == [{
        "path": "artifact.json", "normalization": "lf_to_crlf",
    }]

    path.write_text('{"value": 2}\n', encoding="utf-8")
    failures, normalized = verify_artifacts(manifest, root=tmp_path)
    assert failures
    assert not normalized
