from dataclasses import dataclass, field
import json

import pytest

from src.product.workspace import ProductWorkspace, StudioConfig


def _evidence(root):
    (root / "data/releases").mkdir(parents=True)
    (root / "data/evaluation").mkdir(parents=True)
    (root / "data/world_model").mkdir(parents=True)
    (root / "data/releases/current.json").write_text(json.dumps({
        "active": "7.0.0", "rollback": {"release": "6.0.0"},
    }), encoding="utf-8")
    (root / "data/evaluation/staged_completion_v1.json").write_text(json.dumps({
        "all_stages_complete": True, "production_promotion_ready": False,
    }), encoding="utf-8")
    (root / "data/evaluation/phase5_research_layer_validation_v1.json").write_text(json.dumps({
        "world_model_candidate": {
            "accepted": True,
            "checkpoint": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
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


def test_cognitive_mode_requires_real_credentials(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(mode="cognitive"),
    )
    assert not workspace.readiness()["ready"]
    assert "llm_credentials_available" in workspace.readiness()["blockers"]


def test_studio_match_writes_one_composed_product_report(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Demo", mode="research", seed=11),
    )
    from src import app
    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: _Summary())
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert report["result"]["score"] == {"home": 2, "away": 1}
    assert report["layers"]["world_model"]["enabled"]
    assert not report["layers"]["cognition"]["enabled"]
    assert report["evidence_snapshot"]["world_model_candidate_accepted"]
    assert json.loads(open(report["report_path"], encoding="utf-8").read())["match_id"] == report["match_id"]
    dashboard = open(report["dashboard_path"], encoding="utf-8").read()
    assert "Brazil" in dashboard and "GFS Studio" in dashboard
    assert ProductWorkspace.load(tmp_path).status()["matches_played"] == 1


def test_invalid_product_mode_is_rejected():
    with pytest.raises(ValueError):
        StudioConfig(mode="unsafe")
