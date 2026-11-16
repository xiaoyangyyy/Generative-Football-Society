import json
import shutil
from pathlib import Path

from gfs_core.kernel import PluginKernel
from gfs_core.kernel.project import data_identity, source_identities
from gfs_studio.bootstrap.builtin_plugins import build_evidence_registry
from gfs_studio.plugins.evidence.release.current import (
    current_simulation_verification,
    current_verification,
    source_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare_project(tmp_path):
    for name in (
        "m2_mirrored_policy_protocol_v1.json",
        "m2_mirrored_policy_progress_v6.json",
        "reproduction_release_verification_v1.json",
        "product_value_validation_protocol_v1.json",
        "product_value_scoring_seal_v1.json",
    ):
        destination = tmp_path / "evidence" / "frozen" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PROJECT_ROOT / "evidence" / "frozen" / name, destination)
    report = {
        "code_ready": True,
        "source_identity": source_identity(tmp_path),
        "source_identities": source_identities(tmp_path),
        "checks": {
            "core_tests": True,
            "studio_tests": True,
            "core_ruff": True,
            "studio_ruff": True,
            "core_wheel": True,
            "studio_wheel": True,
            "two_wheels_built": True,
            "clean_venv_created": True,
            "clean_wheel_install": True,
            "installed_core_micro_match": True,
            "installed_studio_release_route": True,
            "plugin_supply_chain": True,
            "static_deployment": True,
        },
        "deployment": {"runtime_verified": False},
    }
    destination = tmp_path / "evidence" / "current" / "build_verification.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(report))


def test_source_identity_accepts_workspace_or_project_root():
    assert source_identity(PROJECT_ROOT) == source_identity(PROJECT_ROOT.parent)


def test_current_build_proof_is_source_bound_and_not_release_proof(tmp_path):
    prepare_project(tmp_path)
    kernel = PluginKernel(build_evidence_registry(tmp_path))
    try:
        kernel.start(["gfs.studio.evidence.release"])
        status = kernel.context.require("release-gates").evaluate()
        assert status["code_ready"] is True
        assert status["release_ready"] is False
        assert status["checks"]["deployment_runtime"] is False
        assert status["participants"]["valid_participants"] == 0
        assert status["scope"] == "current_build_and_historical_research_evidence"
        source = tmp_path / "gfs-studio" / "src" / "gfs_studio" / "changed.py"
        source.parent.mkdir(parents=True)
        source.write_text("VALUE = 1\n")
        assert current_verification(tmp_path)["code_ready"] is False
    finally:
        kernel.stop()


def test_complete_simulation_report_is_accepted_when_identity_matches(tmp_path):
    identities = source_identities(tmp_path)
    report = {
        "state": "completed",
        "protocol_complete": True,
        "source_identity": identities["simulation"],
        "source_identities": identities,
        "data_identity": data_identity(tmp_path),
        "matches_executed": 2,
        "protocol": {"runs_total": 2},
        "metrics": {"pass_completion": {"observed": 0.8}},
    }
    path = tmp_path / "evidence" / "current" / "simulation_evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    assert current_simulation_verification(tmp_path)["available"] is True


def test_simulation_identity_excludes_unrelated_verifier_but_includes_plugins(tmp_path):
    identities = source_identities(tmp_path)
    report = {
        "state": "completed", "protocol_complete": True,
        "source_identities": identities,
        "data_identity": data_identity(tmp_path),
        "matches_executed": 1, "protocol": {"runs_total": 1},
        "metrics": {"pass_completion": {"observed": 0.8}},
    }
    path = tmp_path / "evidence" / "current" / "simulation_evaluation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    unrelated = tmp_path / "scripts" / "run_formal_experiment.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("VALUE = 1\n")
    assert current_simulation_verification(tmp_path)["available"] is True
    manifest = tmp_path / "gfs-core" / "src" / "gfs_core" / "plugins" / "match" / "plugin.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[plugin]\nversion = 'changed'\n")
    assert current_simulation_verification(tmp_path)["available"] is False


def test_data_identity_changes_with_real_input_content(tmp_path):
    path = tmp_path / "data" / "raw" / "results.csv"
    path.parent.mkdir(parents=True)
    path.write_text("home_team,away_team\nBrazil,Argentina\n")
    first = data_identity(tmp_path)
    path.write_text("home_team,away_team\nBrazil,France\n")
    assert data_identity(tmp_path) != first
