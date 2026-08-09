from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

import pytest

from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.pilot import ProspectivePilot


def _evidence(root):
    (root / "data/releases").mkdir(parents=True)
    (root / "data/evaluation").mkdir(parents=True)
    (root / "data/world_model").mkdir(parents=True)
    stable_artifact = root / "data/releases/stable.txt"
    stable_artifact.write_bytes(b"sealed\n")
    release_bytes = (json.dumps({
        "release": "7.0.0",
        "artifacts": [{
            "path": "data/releases/stable.txt",
            "sha256": hashlib.sha256(b"sealed\n").hexdigest(),
        }],
    }, indent=2) + "\n").encode("utf-8")
    (root / "data/releases/v7.0.0.json").write_bytes(release_bytes)
    (root / "data/releases/current.json").write_text(json.dumps({
        "active": "7.0.0",
        "manifest": "data/releases/v7.0.0.json",
        "manifest_sha256": hashlib.sha256(release_bytes).hexdigest(),
        "rollback": {"release": "6.0.0"},
    }), encoding="utf-8")
    (root / "data/evaluation/staged_completion_v1.json").write_text(json.dumps({
        "all_stages_complete": True, "production_promotion_ready": False,
    }), encoding="utf-8")
    (root / "data/evaluation/phase5_research_layer_validation_v1.json").write_text(json.dumps({
        "world_model_candidate": {
            "accepted": True,
            "checkpoint": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
            "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
            "shot_planner_active": False, "shot_fallback": "physics_xg_prior",
        },
        "live_llm_evidence": {"evaluable": False},
    }), encoding="utf-8")
    (root / "data/world_model/latent_wm_rollout_calibrated_candidate.pt").write_bytes(b"model")


@dataclass
class _Summary:
    goals_micro_home: int = 2
    goals_micro_away: int = 1
    micro_xg_home: float = 1.4
    micro_xg_away: float = 0.8
    possession_home: float = 0.55
    passes_home: int = 40
    passes_away: int = 35
    shots_home: int = 8
    shots_away: int = 5
    final_psi: float = 0.1
    home_coach_stress: float = 0.2
    away_coach_stress: float = 0.3
    tactical_drift_home: float = 0.02
    tactical_drift_away: float = 0.03
    world_model_online_calibration: dict = field(default_factory=dict)
    world_model_decision_adoption: dict = field(default_factory=dict)
    world_model_runtime: dict = field(default_factory=lambda: {
        "loaded": True,
        "checkpoint_signature": "sha256:" + hashlib.sha256(b"model").hexdigest(),
        "shot_probability_source": "physics_xg_prior",
    })
    cognitive_triggers: list = field(default_factory=list)
    cognitive_plans: list = field(default_factory=list)
    cognitive_tier_usage: dict = field(default_factory=dict)
    timeline_snippet: list = field(default_factory=lambda: ["12' goal"])
    ball_log_path: str = "ball.jsonl"


def test_studio_status_composes_release_research_and_llm_readiness(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Research Lab", mode="research", seed=7),
    )
    status = workspace.status()
    assert status["readiness"]["ready"]
    assert status["evidence"]["stable_release"] == "7.0.0"
    assert status["evidence"]["shot_fallback"] == "physics_xg_prior"
    assert status["readiness"]["checks"]["research_checkpoint_identity_verified"]
    assert status["workflow"]["state"] == "ready_to_run"
    assert status["workflow"]["next_action"]["id"] == "run_first_match"


def test_stable_mode_rejects_tampered_release_manifest(tmp_path):
    _evidence(tmp_path)
    (tmp_path / "data/releases/v7.0.0.json").write_text(
        '{"release":"tampered"}\n', encoding="utf-8",
    )
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="stable"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert "stable_release_identity_verified" in readiness["blockers"]


def test_stable_mode_rejects_tampered_release_artifact(tmp_path):
    _evidence(tmp_path)
    (tmp_path / "data/releases/stable.txt").write_text(
        "tampered\n", encoding="utf-8",
    )
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="stable"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert "stable_release_artifacts_verified" in readiness["blockers"]


def test_evidence_artifact_paths_cannot_escape_workspace(tmp_path):
    _evidence(tmp_path)
    phase5 = tmp_path / "data/evaluation/phase5_research_layer_validation_v1.json"
    payload = json.loads(phase5.read_text(encoding="utf-8"))
    payload["world_model_candidate"]["checkpoint"] = "../../outside.pt"
    phase5.write_text(json.dumps(payload), encoding="utf-8")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="research"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert not readiness["checks"]["research_checkpoint_available"]


def test_research_mode_rejects_checkpoint_that_does_not_match_evidence(tmp_path):
    _evidence(tmp_path)
    checkpoint = tmp_path / "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
    checkpoint.write_bytes(b"tampered")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="research"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert "research_checkpoint_identity_verified" in readiness["blockers"]


def test_report_rejects_runtime_checkpoint_identity_mismatch(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="research"))
    from src import app

    summary = _Summary(world_model_runtime={
        "loaded": True,
        "checkpoint_signature": "sha256:" + "0" * 64,
        "shot_probability_source": "joint_world_model_head",
    })
    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: summary)
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert not report["integrity"]["accepted"]
    assert "world_model_runtime_identity_mismatch" in report["integrity"]["blockers"]


def test_accepted_shot_head_requires_exact_promoted_artifact_identity(tmp_path):
    _evidence(tmp_path)
    promoted = tmp_path / "data/world_model/frozen_shot_head_v1.json"
    promoted.write_text("{}", encoding="utf-8")
    decision = tmp_path / "data/evaluation/frozen_shot_head_decision.json"
    decision.write_text(json.dumps({
        "accepted": True,
        "promotion_artifact": "data/world_model/frozen_shot_head_v1.json",
        "promotion_artifact_sha256": "0" * 64,
    }), encoding="utf-8")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="research"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert "shot_head_identity_verified" in readiness["blockers"]


def test_cognitive_mode_requires_real_credentials(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(mode="cognitive"),
    )
    assert not workspace.readiness()["ready"]
    assert "llm_credentials_available" in workspace.readiness()["blockers"]


def test_cognitive_readiness_accepts_standard_openai_key_name(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="cognitive"))
    assert workspace.readiness()["ready"]


def test_cognitive_readiness_accepts_deepseek_key_and_reports_safe_provider(
    tmp_path, monkeypatch,
):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="cognitive"))
    readiness = workspace.readiness()
    assert readiness["ready"]
    assert readiness["llm_provider"]["provider"] == "deepseek"
    assert readiness["llm_provider"]["credential_source"] == "DEEPSEEK_API_KEY"
    assert "test-deepseek-key" not in json.dumps(readiness)


def test_cognitive_readiness_rejects_retired_deepseek_model(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    readiness = ProductWorkspace.create(
        tmp_path, StudioConfig(mode="cognitive"),
    ).readiness()
    assert not readiness["ready"]
    assert "llm_config_valid" in readiness["blockers"]
    assert "retired DeepSeek" in readiness["llm_config_error"]


def test_studio_match_writes_one_composed_product_report(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Demo", mode="research", seed=11),
    )
    from src import app
    absolute_ball_log = tmp_path / "outputs/ball_log/demo.jsonl"
    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: _Summary(
        ball_log_path=str(absolute_ball_log),
    ))
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert report["result"]["score"] == {"home": 2, "away": 1}
    assert report["layers"]["world_model"]["enabled"]
    assert report["layers"]["world_model"]["shot_probability_source"] == "physics_xg_prior"
    assert not report["layers"]["cognition"]["enabled"]
    assert report["evidence_snapshot"]["world_model_candidate_accepted"]
    assert report["artifacts"]["ball_log"] == "outputs/ball_log/demo.jsonl"
    assert str(tmp_path) not in json.dumps(report["artifacts"])
    assert report["integrity"] == {
        "accepted": True, "state": "accepted", "blockers": [],
    }
    assert json.loads(open(report["report_path"], encoding="utf-8").read())["match_id"] == report["match_id"]
    dashboard = open(report["dashboard_path"], encoding="utf-8").read()
    assert "Brazil" in dashboard and "GFS Studio" in dashboard
    assert ProductWorkspace.load(tmp_path).status()["matches_played"] == 1
    status = ProductWorkspace.load(tmp_path).status()
    assert status["runs_total"] == 1
    assert status["last_run"]["state"] == "completed"
    assert status["workflow"]["state"] == "review"
    assert status["workflow"]["progress"]["report_available"]
    assert status["workflow"]["next_action"]["id"] == "open_dashboard"
    assert status["workflow"]["next_action"]["target"].endswith(".html")


def test_pilot_preflight_is_blocked_without_credentials_and_makes_no_calls(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="cognitive"))
    result = ProspectivePilot(workspace).preflight()
    assert result["state"] == "blocked"
    assert not result["external_calls_made"]
    assert "llm_credentials_available" in result["blockers"]


def test_pilot_requires_a_clean_session_to_preserve_frozen_seeds(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.setenv("API_KEY", "test-key")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="cognitive"))
    session = workspace._session()
    session["matches"] = [{"match_id": "existing"}]
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")
    result = ProspectivePilot(workspace).preflight()
    assert "pilot_requires_empty_session" in result["blockers"]


def test_pilot_preflight_reports_all_cognitive_requirements_from_stable_mode(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="stable"))
    blockers = ProspectivePilot(workspace).preflight()["blockers"]
    assert "studio_mode_must_be_cognitive" in blockers
    assert "llm_credentials_available" in blockers


def test_cognitive_report_persists_real_provider_provenance(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.setenv("API_KEY", "test-key")
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="cognitive"))
    from src import app
    from src.simulation import llm_gateway

    class _Config:
        model = "test-model"
        base_url = "https://provider.invalid/v1"

    class _Gateway:
        config = _Config()
        call_count = 3

    gateway = _Gateway()
    monkeypatch.setattr(llm_gateway, "get_shared_llm_gateway", lambda: gateway)

    def run(*args, **kwargs):
        gateway.call_count += 1
        return _Summary(cognitive_plans=[{"action": "press"}])

    monkeypatch.setattr(app, "run_micro_match", run)
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert report["layers"]["cognition"]["provider"]["successful_calls"] == 1
    assert report["layers"]["cognition"]["provider"]["real_provider_evidence"]
    persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert persisted["artifacts"]["cognitive_log"]
    assert (tmp_path / persisted["artifacts"]["cognitive_log"]).is_file()
    assert report["integrity"]["accepted"]


def test_invalid_product_mode_is_rejected():
    with pytest.raises(ValueError):
        StudioConfig(mode="unsafe")


def test_studio_creation_requires_explicit_replace(tmp_path):
    _evidence(tmp_path)
    ProductWorkspace.create(tmp_path, StudioConfig(name="First"))
    with pytest.raises(FileExistsError, match="--replace"):
        ProductWorkspace.create(tmp_path, StudioConfig(name="Second"))
    replaced = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Second"), replace=True,
    )
    assert replaced._session()["name"] == "Second"


def test_empty_studio_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        StudioConfig(name="   ")


def test_concurrent_matches_commit_unique_session_indices(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Concurrent", mode="research", seed=11),
    )
    from src import app

    def run(*args, **kwargs):
        time.sleep(0.05)
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(
            lambda fixture: workspace.run_match(*fixture, fast=True),
            [("Brazil", "Argentina"), ("France", "Germany")],
        ))
    assert {report["match_id"].split("-", 1)[0] for report in reports} == {"0001", "0002"}
    assert ProductWorkspace.load(tmp_path).status()["matches_played"] == 2


def test_failed_match_is_audited_and_its_seed_is_not_reused(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Failure Audit", mode="research", seed=20),
    )
    from src import app

    calls = 0

    def run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated failure")
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", run)
    with pytest.raises(RuntimeError, match="simulated failure"):
        workspace.run_match("Brazil", "Argentina", fast=True)
    report = workspace.run_match("France", "Germany", fast=True)
    assert report["match_id"].startswith("0002-")
    assert report["fixture"]["seed"] == 21
    status = workspace.status()
    assert status["runs_total"] == 2
    assert status["failed_runs"] == 1
    assert status["matches_played"] == 1


def test_post_simulation_reporting_failure_preserves_run_provenance(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Report Failure", mode="research", seed=30),
    )
    from src import app
    from src.product import reporting

    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: _Summary())
    monkeypatch.setattr(
        reporting, "write_match_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("html failure")),
    )
    with pytest.raises(RuntimeError, match="html failure"):
        workspace.run_match("Brazil", "Argentina", fast=True)
    run = workspace._session()["runs"][0]
    assert run["state"] == "failed"
    assert run["simulation_completed_at"]
    assert run["successful_provider_calls"] == 0
