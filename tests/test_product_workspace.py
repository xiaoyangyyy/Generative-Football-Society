from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time

import pytest

from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.match_plan import MatchPlan, PLAYABLE_TACTICS, WorldModelForkPlan
from src.product.club_strategy import (
    resolve_fixture_club_strategy,
    strategy_identity,
)
from src.product.club_finance import evidence_identity
from src.product.season import ClubResourcePlan, ManagerDecision, SeasonPlan
from src.product.sporting_director import SportingDirective
from src.product.club_timeline import ClubEventChoice
from src.product.player_promises import PlayerPromisePlan, PlayerRolePromise
from src.product.decision_ledger import (
    build_manager_decision_ledger,
    validate_manager_decision_ledger,
)
from src.match_engine.tactical_catalog import TACTICAL_KEYS
from src.product.decision_advice import (
    build_manager_advice_adoption,
    build_manager_decision_advice,
)
from src.product.pilot import ProspectivePilot
from src.match_engine.manager_plan import InMatchInstruction, InMatchPlan
from src.simulation.squad_registry import (
    RecruitmentMove,
    RecruitmentPlan,
    load_effective_roster,
)
from src.simulation.player_lifecycle import RetentionPlan
from src.simulation.player_market import FreeAgentPlan


def _evidence(root):
    (root / "data/releases").mkdir(parents=True)
    (root / "data/evaluation").mkdir(parents=True)
    (root / "data/world_model").mkdir(parents=True)
    stable_artifact = root / "data/releases/stable.txt"
    stable_artifact.write_bytes(b"sealed\n")
    release_bytes = (
        json.dumps(
            {
                "release": "7.0.0",
                "artifacts": [
                    {
                        "path": "data/releases/stable.txt",
                        "sha256": hashlib.sha256(b"sealed\n").hexdigest(),
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    (root / "data/releases/v7.0.0.json").write_bytes(release_bytes)
    (root / "data/releases/current.json").write_text(
        json.dumps(
            {
                "active": "7.0.0",
                "manifest": "data/releases/v7.0.0.json",
                "manifest_sha256": hashlib.sha256(release_bytes).hexdigest(),
                "rollback": {"release": "6.0.0"},
            }
        ),
        encoding="utf-8",
    )
    (root / "data/evaluation/staged_completion_v1.json").write_text(
        json.dumps(
            {
                "all_stages_complete": True,
                "production_promotion_ready": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "data/evaluation/phase5_research_layer_validation_v1.json").write_text(
        json.dumps(
            {
                "world_model_candidate": {
                    "accepted": True,
                    "checkpoint": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
                    "checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
                    "shot_planner_active": False,
                    "shot_fallback": "physics_xg_prior",
                },
                "live_llm_evidence": {"evaluable": False},
            }
        ),
        encoding="utf-8",
    )
    (root / "data/world_model/latent_wm_rollout_calibrated_candidate.pt").write_bytes(
        b"model"
    )


def _manager_roster(root, team="Brazil"):
    roles = (
        "GK",
        "RB",
        "CB",
        "CB",
        "LB",
        "CM",
        "CM",
        "CM",
        "RW",
        "ST",
        "LW",
        "GK",
        "CB",
        "DM",
        "AM",
        "ST",
    )
    players = []
    for index, role in enumerate(roles):
        quality = 0.52 + 0.015 * index
        players.append(
            {
                "player_id": f"{team.lower()}-{index:02d}",
                "name": f"{team} Player {index}",
                "role": role,
                "age": 20 + index,
                "squad_role": "starter" if index < 11 else "bench",
                "availability": 1.0,
                "condition": {"composure": quality},
                "abilities": {
                    key: quality
                    for key in (
                        "tech",
                        "pass_skill",
                        "vision",
                        "spatial",
                        "pace",
                        "press",
                        "shot",
                        "power",
                        "aerial",
                        "mental",
                        "gk_reflex",
                        "gk_aerial",
                    )
                },
            }
        )
    path = root / f"data/rosters/{team}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "team_id": team,
                "formation": "4-3-3",
                "players": players,
            }
        ),
        encoding="utf-8",
    )


def _manager_advice_packet():
    return {
        "available": True,
        "evaluation_scope": "short_horizon_tactical_policy_proxy",
        "horizon_s": 20.0,
        "observation_coverage": 0.94,
        "checkpoint_signature": "sha256:" + hashlib.sha256(b"model").hexdigest(),
        "candidates": [
            {
                "tactical_preset": tactic,
                "risk_adjusted_value": 0.8 - index * 0.05,
                "effective_confidence": 0.75 - index * 0.01,
                "uncertainty": 0.15 + index * 0.01,
                "fatigue_cost_proxy": 0.2 + index * 0.02,
                "structural_risk_proxy": 0.25 + index * 0.02,
                "event_probabilities": {
                    "retain": 0.45,
                    "turnover": 0.22,
                    "shot": 0.18,
                    "foul": 0.10,
                    "out": 0.05,
                },
            }
            for index, tactic in enumerate(PLAYABLE_TACTICS)
        ],
        "fusion_reliability": {
            "evidence_tier": "insufficient_history",
            "guidance": "low_authority",
            "adjusted_recommendation_trust": 0.51,
            "matched_seed_samples": 0,
            "historical_match_records": 0,
            "causal_scope": "none",
        },
    }


@dataclass
class _Summary:
    goals_micro_home: int = 2
    goals_micro_away: int = 1
    goals_physics_home: int = 2
    goals_physics_away: int = 1
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
    world_model_action_adoption: dict = field(
        default_factory=lambda: {
            "available": True,
            "opportunities": 12,
            "influenced_opportunities": 10,
            "attribution_eligible_opportunities": 10,
            "counterfactual_action_changes": 2,
            "expected_counterfactual_action_changes": 1.8,
            "counterfactual_change_rate": 0.2,
            "expected_counterfactual_change_rate": 0.18,
            "mean_recommended_probability_shift": 0.015,
            "records": [
                {
                    "opportunity_id": "direct:Brazil:12.000:0",
                    "team_id": "Brazil",
                    "t_sec": 120.0,
                    "recommended_action": "pass",
                    "counterfactual_baseline_action": "hold",
                    "actual_action": "pass",
                    "policy_changed_action": True,
                    "attribution_eligible": True,
                    "base_probability": {"pass": 0.8},
                    "adjusted_probability": {"pass": 0.6},
                    "total_variation_distance": 0.2,
                    "quality_gates": {
                        "pass": {"reason": "validated_pass_vs_hold_advantage"}
                    },
                }
            ],
        }
    )
    world_model_branch_anchor: dict = field(default_factory=lambda: {
        "available": True,
        "requested_sec": 2700.0,
        "actual_sec": 2700.0,
        "state_identity": "a" * 64,
        "resume_capability": "deterministic_replay_only",
    })
    world_model_runtime: dict = field(
        default_factory=lambda: {
            "loaded": True,
            "checkpoint_signature": "sha256:" + hashlib.sha256(b"model").hexdigest(),
            "shot_probability_source": "physics_xg_prior",
        }
    )
    simulation_clock: dict = field(default_factory=lambda: {
        "schema_version": 1,
        "contract": "authoritative_tick_v2",
        "authoritative_tick_clock": True,
        "final_logical_sec": 5400.0,
        "final_state_clock_sec": 5400.0,
    })
    cognitive_triggers: list = field(default_factory=list)
    cognitive_plans: list = field(default_factory=list)
    cognitive_tier_usage: dict = field(default_factory=dict)
    timeline_snippet: list = field(default_factory=lambda: ["12' goal"])
    ball_log_path: str = "ball.jsonl"
    manager_effects: dict = field(default_factory=dict)
    in_match_management: dict = field(default_factory=dict)
    tactical_execution: dict = field(default_factory=dict)


def _runtime_tactical_execution(home, away, home_tactic, away_tactic):
    vector = {key: 0.5 for key in TACTICAL_KEYS}
    return {
        side: {
            "schema_version": 1,
            "team": team,
            "applied_tactic": tactic or "team_identity",
            "native_archetype": tactic or "balanced",
            "binding_kind": ("locked_preset" if tactic else "native_team_vector"),
            "source": ("studio_user_intervention" if tactic else "agent_team_identity"),
            "preset_locked": bool(tactic),
            "initial_vector": dict(vector),
            "final_vector": dict(vector),
            "changed_controls": [],
            "final_delta_l1": 0.0,
        }
        for side, team, tactic in (
            ("home", home, home_tactic),
            ("away", away, away_tactic),
        )
    }


def test_studio_status_composes_release_research_and_llm_readiness(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Research Lab", mode="research", seed=7),
    )
    status = workspace.status()
    assert status["readiness"]["ready"]
    assert status["evidence"]["stable_release"] == "7.0.0"
    assert status["evidence"]["shot_fallback"] == "physics_xg_prior"
    assert status["readiness"]["checks"]["research_checkpoint_identity_verified"]
    assert status["workflow"]["state"] == "ready_to_start_season"
    assert status["workflow"]["next_action"]["id"] == "start_season"
    assert status["workflow"]["next_action"]["target"] == "#season-form"
    assert status["workflow"]["alternative_actions"][0]["id"] == (
        "run_standalone_match"
    )


def test_workflow_unifies_season_setup_decision_and_matchday_actions(tmp_path):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Unified season", mode="research", seed=7),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            manager_team="Brazil",
            manager_objective="top_half",
        )
    )

    setup = workspace.status()["workflow"]
    assert setup["schema_version"] == 2
    assert setup["state"] == "season_setup"
    assert setup["next_action"]["id"] == "freeze_player_promises"
    assert setup["next_action"]["target"] == "#player-promise-form"
    assert setup["journey"][0] == {
        "id": "season_setup",
        "status": "action_required",
    }

    workspace.set_player_role_promises(
        PlayerPromisePlan((PlayerRolePromise("brazil-00", "core"),))
    )
    decision = workspace.workflow()
    assert decision["state"] == "season_decision"
    assert decision["next_action"]["id"] == "submit_manager_decision"
    assert decision["next_action"]["opponent"]
    assert decision["progress"]["season_setup_complete"] is True

    workspace.set_manager_decision(ManagerDecision(team="Brazil"))
    ready = workspace.status()["workflow"]
    assert ready["state"] == "season_matchday_ready"
    assert ready["next_action"]["id"] == "advance_season_matchday"
    assert ready["next_action"]["target"] == "#play-matchday"
    assert ready["progress"]["season_matchday_ready"] is True
    assert ready["journey"][2] == {
        "id": "matchday_execution",
        "status": "ready",
    }

    unsupported = json.loads(json.dumps(workspace.status()))
    unsupported["season"]["matchday_command_center"]["phase"] = "unknown"
    failed_closed = workspace.workflow(status=unsupported)
    assert failed_closed["state"] == "blocked"
    assert failed_closed["next_action"]["id"] == "inspect_season_state"

    readiness_blocked = json.loads(json.dumps(workspace.status()))
    readiness_blocked["readiness"] = {
        "ready": False,
        "blockers": ["test_gate"],
    }
    blocked = workspace.workflow(status=readiness_blocked)
    assert blocked["next_action"]["id"] == "resolve_readiness"
    assert [step["status"] for step in blocked["journey"][:3]] == [
        "blocked",
        "blocked",
        "blocked",
    ]


def test_workflow_prioritizes_active_spectator_season_over_single_match_loop(
    tmp_path,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Spectator season", mode="stable", seed=8),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team=None,
        )
    )

    workflow = workspace.status()["workflow"]

    assert workflow["state"] == "season_matchday_ready"
    assert workflow["next_action"]["id"] == "advance_season_matchday"
    assert workflow["next_action"]["matchday"] == 1

    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": {"home": 0, "away": 0},
            }
        )
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    completed = workspace.status()["workflow"]
    assert completed["state"] == "season_review"
    assert completed["next_action"]["id"] == "review_season"
    assert completed["progress"]["season_complete"] is True


def test_stable_mode_rejects_tampered_release_manifest(tmp_path):
    _evidence(tmp_path)
    (tmp_path / "data/releases/v7.0.0.json").write_text(
        '{"release":"tampered"}\n',
        encoding="utf-8",
    )
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="stable"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert "stable_release_identity_verified" in readiness["blockers"]


def test_stable_mode_rejects_tampered_release_artifact(tmp_path):
    _evidence(tmp_path)
    (tmp_path / "data/releases/stable.txt").write_text(
        "tampered\n",
        encoding="utf-8",
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

    summary = _Summary(
        world_model_runtime={
            "loaded": True,
            "checkpoint_signature": "sha256:" + "0" * 64,
            "shot_probability_source": "joint_world_model_head",
        }
    )
    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: summary)
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert not report["integrity"]["accepted"]
    assert "world_model_runtime_identity_mismatch" in report["integrity"]["blockers"]


def test_report_rejects_missing_authoritative_product_clock(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="research"))
    from src import app

    monkeypatch.setattr(
        app, "run_micro_match",
        lambda *args, **kwargs: _Summary(simulation_clock={}),
    )
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert not report["integrity"]["accepted"]
    assert "authoritative_product_clock_not_observed" in (
        report["integrity"]["blockers"]
    )


def test_accepted_shot_head_requires_exact_promoted_artifact_identity(tmp_path):
    _evidence(tmp_path)
    promoted = tmp_path / "data/world_model/frozen_shot_head_v1.json"
    promoted.write_text("{}", encoding="utf-8")
    decision = tmp_path / "data/evaluation/frozen_shot_head_decision.json"
    decision.write_text(
        json.dumps(
            {
                "accepted": True,
                "promotion_artifact": "data/world_model/frozen_shot_head_v1.json",
                "promotion_artifact_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(mode="research"))
    readiness = workspace.readiness()
    assert not readiness["ready"]
    assert "shot_head_identity_verified" in readiness["blockers"]


def test_cognitive_mode_requires_real_credentials(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.delenv("API_KEY", raising=False)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(mode="cognitive"),
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
    tmp_path,
    monkeypatch,
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


def test_evidence_exposes_confirmed_mechanism_and_inconclusive_outcome(tmp_path):
    _evidence(tmp_path)
    evaluation = tmp_path / "data/evaluation"
    (evaluation / "action_adoption_v1").mkdir(parents=True)
    (evaluation / "action_outcome_v1").mkdir(parents=True)
    (evaluation / "action_adoption_protocol_v1.json").write_text(
        json.dumps(
            {
                "protocol_id": "mechanism-v1",
                "claim_scope": "mechanism_only",
                "state": "preregistered_not_executed",
                "design": {"runs_total": 24},
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "action_adoption_v1/progress.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "arms": {
                    "M1_predict_only": {"rows": [{}] * 12},
                    "M1_action_policy": {"rows": [{}] * 12},
                },
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "action_adoption_v1/decision.json").write_text(
        json.dumps(
            {
                "status": "mechanism_confirmed",
                "passed": True,
                "mechanism": {
                    "action_opportunities": 1954,
                    "realized_counterfactual_action_changes": 25,
                },
                "promotion_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "action_outcome_protocol_v1.json").write_text(
        json.dumps(
            {
                "protocol_id": "outcome-v1",
                "design": {"runs_total": 60},
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "action_outcome_v1/progress.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "arms": {"M0": {"rows": [{}] * 30}, "M1": {"rows": [{}] * 30}},
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "action_outcome_v1/decision.json").write_text(
        json.dumps(
            {
                "decision": "inconclusive_keep_research_only",
                "promotion_supported": False,
                "pairs_total": 30,
                "primary": {
                    "point_delta": -0.03189252164278855,
                    "ci95_low": -26.061954645805613,
                    "ci95_high": 5.137429588662575,
                },
                "minimum_meaningful_delta_loss": 0.1,
                "behavior": {"changed_pairs": 30, "changed_pair_fraction": 1.0},
                "promotion_gates": {
                    "candidate_passes_all_external_validity_gates": False,
                    "upper_interval_below_negative_minimum_effect": False,
                    "minimum_behavior_change_met": True,
                    "execution_identity_verified": True,
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = ProductWorkspace.create(
        tmp_path,
        StudioConfig(mode="research"),
    ).evidence()
    mechanism = evidence["action_adoption_mechanism"]
    assert mechanism["execution_state"] == "mechanism_confirmed"
    assert mechanism["runs_executed"] == 24
    assert mechanism["mechanism"]["realized_counterfactual_action_changes"] == 25
    assert mechanism["promotion_authorized"] is False
    outcome = evidence["action_outcome_study"]
    assert outcome["execution_state"] == "inconclusive_keep_research_only"
    assert outcome["result_status"] == "inconclusive_keep_research_only"
    assert outcome["runs_executed"] == 60
    assert outcome["fixed_run_budget"] == 60
    assert outcome["promotion_supported"] is False
    assert outcome["behavior"]["changed_pairs"] == 30
    assert outcome["promotion_gates"]["minimum_behavior_change_met"] is True


def test_cognitive_readiness_rejects_retired_deepseek_model(tmp_path, monkeypatch):
    _evidence(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    readiness = ProductWorkspace.create(
        tmp_path,
        StudioConfig(mode="cognitive"),
    ).readiness()
    assert not readiness["ready"]
    assert "llm_config_valid" in readiness["blockers"]
    assert "retired DeepSeek" in readiness["llm_config_error"]


def test_studio_match_writes_one_composed_product_report(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Demo", mode="research", seed=11),
    )
    from src import app

    absolute_ball_log = tmp_path / "outputs/ball_log/demo.jsonl"
    absolute_ball_log.parent.mkdir(parents=True, exist_ok=True)
    absolute_ball_log.write_text(
        "\n".join(
            [
                json.dumps({"meta": {"home": "Brazil", "away": "Argentina"}}),
                json.dumps(
                    {
                        "type": "pass",
                        "t_sec": 120,
                        "team": "Brazil",
                        "from": "A",
                        "to": "B",
                        "kind": "ground",
                        "from_xy": [0.2, 0.3],
                        "land_xy": [0.7, 0.4],
                        "outcome": "COMPLETE",
                        "wm_action_opportunity_id": "direct:Brazil:12.000:0",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app,
        "run_micro_match",
        lambda *args, **kwargs: _Summary(
            ball_log_path=str(absolute_ball_log),
        ),
    )
    report = workspace.run_match("Brazil", "Argentina", fast=True)
    assert report["result"]["score"] == {"home": 2, "away": 1}
    assert report["layers"]["world_model"]["enabled"]
    assert (
        report["layers"]["world_model"]["shot_probability_source"] == "physics_xg_prior"
    )
    assert (
        report["layers"]["world_model"]["action_adoption"][
            "counterfactual_action_changes"
        ]
        == 2
    )
    assert not report["layers"]["cognition"]["enabled"]
    assert report["evidence_snapshot"]["world_model_candidate_accepted"]
    assert report["artifacts"]["ball_log"] == "outputs/ball_log/demo.jsonl"
    assert report["replay"]["available"]
    assert report["replay"]["retained_events"] == 1
    assert report["replay"]["world_model_action_links"]["directly_observed"] == 1
    assert str(tmp_path) not in json.dumps(report["artifacts"])
    assert report["integrity"] == {
        "accepted": True,
        "state": "accepted",
        "blockers": [],
    }
    assert (
        json.loads(open(report["report_path"], encoding="utf-8").read())["match_id"]
        == report["match_id"]
    )
    dashboard = open(report["dashboard_path"], encoding="utf-8").read()
    assert "Brazil" in dashboard and "GFS Studio" in dashboard
    assert "World-model action influence" in dashboard
    assert "Counterfactual changes" in dashboard
    assert "Ball-action trajectory replay" in dashboard
    assert '<svg viewBox="0 0 1000 640"' in dashboard
    assert ProductWorkspace.load(tmp_path).status()["matches_played"] == 1
    status = ProductWorkspace.load(tmp_path).status()
    latest_adoption = status["last_match"]["world_model_action_adoption"]
    assert latest_adoption["opportunities"] == 12
    assert latest_adoption["influenced_opportunities"] == 10
    assert latest_adoption["counterfactual_action_changes"] == 2
    assert latest_adoption["counterfactual_change_rate"] == 0.2
    assert "records" not in latest_adoption
    assert status["runs_total"] == 1
    assert status["last_run"]["state"] == "completed"
    assert status["workflow"]["state"] == "review"
    assert status["workflow"]["progress"]["report_available"]
    session = json.loads(workspace.session_path.read_text(encoding="utf-8"))
    session["matches"][-1].pop("world_model_action_adoption")
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")
    recovered = workspace.status()["last_match"]["world_model_action_adoption"]
    assert recovered["opportunities"] == 12
    assert "records" not in recovered
    assert status["workflow"]["next_action"]["id"] == "open_dashboard"
    assert status["workflow"]["next_action"]["target"].endswith(".html")


def test_match_history_is_bounded_latest_first_and_field_minimized(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="History", mode="stable", seed=7),
    )
    session = json.loads(workspace.session_path.read_text(encoding="utf-8"))
    session["matches"] = [
        {
            "match_id": f"{index:04d}-home-vs-away",
            "home": "Home",
            "away": "Away",
            "seed": index,
            "fast": True,
            "score": {"home": index % 3, "away": 0},
            "integrity": "accepted",
            "match_plan": {
                "experience": "observational",
                "home_tactic": "team_identity",
                "away_tactic": "team_identity",
            },
            "dashboard": f"outputs/studio/history/matches/{index:04d}.html",
            "comparison_dashboard": None,
            "private_note": "must-not-cross-history-boundary",
        }
        for index in range(55)
    ]
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    status = workspace.status()
    history = status["match_history"]
    assert status["matches_played"] == 55
    assert status["match_history_limit"] == 50
    assert status["match_history_truncated"] is True
    assert len(history) == 50
    assert history[0]["match_id"] == "0054-home-vs-away"
    assert history[-1]["match_id"] == "0005-home-vs-away"
    assert "private_note" not in json.dumps(history)


def test_tactical_lab_uses_locked_presets_physics_score_and_explicit_seed(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Lab", mode="research", seed=11),
    )
    observed = {}
    from src import app

    def fake_match(*args, **kwargs):
        observed.update(kwargs)
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", fake_match)
    plan = MatchPlan(
        experience="tactical_lab",
        home_tactic="gegenpress",
        away_tactic="low_block_counter",
        reuse_last_seed=False,
    )
    report = workspace.run_match(
        "Brazil",
        "Argentina",
        fast=True,
        plan=plan,
        seed_override=77,
    )
    assert observed["seed"] == 77
    assert observed["config"].use_micro_goals is True
    assert observed["home_tactic"] == "gegenpress"
    assert observed["away_tactic"] == "low_block_counter"
    assert observed["stage_name"] == report["match_id"]
    assert report["match_plan"]["score_path"] == "physics_official"
    assert report["match_plan"]["claim_boundary"].startswith(
        "single_run_descriptive_only"
    )
    assert report["integrity"]["accepted"]
    latest = workspace.status()["last_match"]
    assert latest["seed"] == 77
    assert latest["fast"] is True
    assert latest["match_plan"]["home_tactic"] == "gegenpress"
    assert workspace.replay_seed("Brazil", "Argentina") == 77
    with pytest.raises(ValueError, match="same ordered fixture"):
        workspace.replay_seed("Argentina", "Brazil")
    second_plan = MatchPlan(
        experience="tactical_lab",
        home_tactic="counter_attack",
        away_tactic="low_block_counter",
        reuse_last_seed=True,
    )
    second = workspace.run_match(
        "Brazil",
        "Argentina",
        fast=True,
        plan=second_plan,
    )
    assert second["fixture"]["seed"] == 77
    assert second["match_plan"]["paired_baseline_match_id"] == report["match_id"]
    assert Path(second["comparison_path"]).is_file()
    assert Path(second["comparison_dashboard_path"]).is_file()
    treatment_dashboard = Path(second["dashboard_path"]).read_text(encoding="utf-8")
    assert "打开同种子战术配对比较" in treatment_dashboard
    comparison = json.loads(Path(second["comparison_path"]).read_text(encoding="utf-8"))
    assert comparison["eligibility"]["eligible_for_tactical_attribution"]
    assert comparison["intervention"]["scope"] == ("single_side_tactical_intervention")
    latest_pair = workspace.status()["last_match"]
    assert latest_pair["comparison_dashboard"].endswith(".comparison.html")
    with pytest.raises(ValueError, match="same fast configuration"):
        workspace.run_match(
            "Brazil",
            "Argentina",
            fast=False,
            plan=second_plan,
        )
    Path(second["report_path"]).unlink()
    with pytest.raises(ValueError, match="baseline report is unavailable"):
        workspace.run_match(
            "Brazil",
            "Argentina",
            fast=True,
            plan=second_plan,
        )
    failed_status = workspace.status()
    assert failed_status["matches_played"] == 2
    assert failed_status["last_run"]["state"] == "failed"


def test_workspace_paired_execution_enforces_single_side_physics_contract(
    tmp_path,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Pair Guard", mode="research", seed=11),
    )
    unchanged_baseline = MatchPlan(
        experience="tactical_lab",
        home_tactic="balanced",
        away_tactic="low_block_counter",
    )
    unchanged_treatment = MatchPlan(
        experience="tactical_lab",
        home_tactic="balanced",
        away_tactic="low_block_counter",
        reuse_last_seed=True,
    )
    with pytest.raises(ValueError, match="exactly one changed"):
        workspace.run_paired_matches(
            "Brazil",
            "Argentina",
            fast=True,
            baseline_plan=unchanged_baseline,
            treatment_plan=unchanged_treatment,
            seed=7,
        )
    joint_treatment = MatchPlan(
        experience="tactical_lab",
        home_tactic="gegenpress",
        away_tactic="direct_vertical",
        reuse_last_seed=True,
    )
    with pytest.raises(ValueError, match="exactly one changed"):
        workspace.run_paired_matches(
            "Brazil",
            "Argentina",
            fast=True,
            baseline_plan=unchanged_baseline,
            treatment_plan=joint_treatment,
            seed=7,
        )
    with pytest.raises(ValueError, match="matching tactical_lab"):
        workspace.run_paired_matches(
            "Brazil",
            "Argentina",
            fast=True,
            baseline_plan=MatchPlan(),
            treatment_plan=joint_treatment,
            seed=7,
        )


def test_workspace_world_model_fork_is_atomic_and_changes_only_plan_authority(
    tmp_path, monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="World Fork", mode="research", seed=11),
    )
    fork = WorldModelForkPlan(
        home_tactic="balanced", away_tactic="low_block_counter", seed=77,
    )
    observed = []
    from src import app
    from src.simulation.runtime import environment_snapshot

    def fake_match(*args, **kwargs):
        environment = environment_snapshot()
        observed.append({
            "plan": environment.get("MATCH_WM_PLAN"),
            "checkpoint": environment.get("MATCH_WM_CHECKPOINT"),
            "branch_at": environment.get("MATCH_WM_BRANCH_AT_SEC"),
            "plan_start": environment.get("MATCH_WM_PLAN_START_SEC"),
            "home_tactic": kwargs.get("home_tactic"),
            "away_tactic": kwargs.get("away_tactic"),
            "seed": kwargs.get("seed"),
        })
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", fake_match)
    baseline, treatment = workspace.run_paired_matches(
        "Brazil", "Argentina", fast=True,
        baseline_plan=fork.baseline_plan(),
        treatment_plan=fork.treatment_plan(), seed=fork.seed,
        transaction_id="wm-fork-transaction-001",
    )
    assert [row["plan"] for row in observed] == ["0", "1"]
    assert observed[0]["checkpoint"] == observed[1]["checkpoint"]
    assert [row["branch_at"] for row in observed] == ["2700.0", "2700.0"]
    assert [row["plan_start"] for row in observed] == ["2700.0", "2700.0"]
    assert observed[0]["home_tactic"] == observed[1]["home_tactic"] == "balanced"
    assert observed[0]["away_tactic"] == observed[1]["away_tactic"] == (
        "low_block_counter"
    )
    assert observed[0]["seed"] == observed[1]["seed"] == 77
    comparison = json.loads(
        Path(treatment["comparison_path"]).read_text(encoding="utf-8")
    )
    assert comparison["intervention"]["scope"] == "world_model_action_policy"
    assert comparison["eligibility"][
        "eligible_for_world_model_policy_attribution"
    ]
    assert comparison["intervention"]["branch_anchor"]["verified"]
    assert comparison["policy_propagation"]["status"] == (
        "changed_actions_not_directly_observed"
    )
    assert comparison["policy_propagation"]["summary"] == {
        "valid_changed_decisions": 1,
        "retained_changed_decisions": 1,
        "directly_observed_changes": 0,
        "locally_attributable_changes": 0,
        "invalid_changed_records": 0,
        "duplicate_opportunity_ids": 0,
        "replay_windows_available": False,
        "decisions_truncated": False,
    }
    comparison_dashboard = Path(
        treatment["comparison_dashboard_path"]
    ).read_text(encoding="utf-8")
    assert 'data-testid="policy-propagation-panel"' in comparison_dashboard
    assert "直接轨迹未绑定" in comparison_dashboard
    assert "不声称下游因果" in comparison_dashboard
    assert baseline["match_plan"]["world_model_policy"] == "predict_only"
    assert treatment["match_plan"]["world_model_policy"] == "action_policy"


def test_paired_transaction_resumes_after_baseline_without_duplicate(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Pair Resume", mode="research", seed=11),
    )
    baseline_plan = MatchPlan(
        experience="tactical_lab",
        home_tactic="balanced",
        away_tactic="low_block_counter",
    )
    treatment_plan = MatchPlan(
        experience="tactical_lab",
        home_tactic="gegenpress",
        away_tactic="low_block_counter",
        reuse_last_seed=True,
    )
    calls = []
    from src import app

    def flaky_match(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("simulated treatment crash")
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", flaky_match)
    transaction_id = "pairtx-resume-001"
    with pytest.raises(RuntimeError, match="simulated treatment crash"):
        workspace.run_paired_matches(
            "Brazil",
            "Argentina",
            fast=True,
            baseline_plan=baseline_plan,
            treatment_plan=treatment_plan,
            seed=77,
            transaction_id=transaction_id,
        )
    session = json.loads(workspace.session_path.read_text(encoding="utf-8"))
    assert len(session["matches"]) == 1
    baseline_id = session["matches"][0]["match_id"]
    assert session["matches"][0]["match_plan"]["pair_transaction_id"] == (
        transaction_id
    )
    assert session["matches"][0]["match_plan"]["pair_role"] == "baseline"

    workspace.run_match("France", "Spain", fast=True)
    resumed_baseline, resumed_treatment = workspace.run_paired_matches(
        "Brazil",
        "Argentina",
        fast=True,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        seed=77,
        transaction_id=transaction_id,
    )
    assert len(calls) == 4
    assert resumed_baseline["match_id"] == baseline_id
    assert resumed_treatment["match_plan"]["paired_baseline_match_id"] == baseline_id
    assert resumed_treatment["match_plan"]["pair_transaction_id"] == transaction_id
    assert resumed_treatment["match_plan"]["pair_role"] == "treatment"
    assert Path(resumed_treatment["comparison_dashboard_path"]).is_file()

    replayed_baseline, replayed_treatment = workspace.run_paired_matches(
        "Brazil",
        "Argentina",
        fast=True,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        seed=77,
        transaction_id=transaction_id,
    )
    assert len(calls) == 4
    assert replayed_baseline["match_id"] == resumed_baseline["match_id"]
    assert replayed_treatment["match_id"] == resumed_treatment["match_id"]
    with pytest.raises(ValueError, match="identity conflict"):
        workspace.run_paired_matches(
            "Brazil",
            "Argentina",
            fast=True,
            baseline_plan=baseline_plan,
            treatment_plan=treatment_plan,
            seed=78,
            transaction_id=transaction_id,
        )
    assert len(calls) == 4


def test_stable_workspace_rejects_tactical_lab_before_reserving_run(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Stable", mode="stable", seed=11),
    )
    with pytest.raises(ValueError, match="requires research"):
        workspace.run_match(
            "Brazil",
            "Argentina",
            plan=MatchPlan(experience="tactical_lab", home_tactic="gegenpress"),
        )
    assert workspace.status()["runs_total"] == 0


def test_pilot_preflight_is_blocked_without_credentials_and_makes_no_calls(
    tmp_path, monkeypatch
):
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


def test_pilot_preflight_reports_all_cognitive_requirements_from_stable_mode(
    tmp_path, monkeypatch
):
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
        tmp_path,
        StudioConfig(name="Second"),
        replace=True,
    )
    assert replaced._session()["name"] == "Second"


def test_empty_studio_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        StudioConfig(name="   ")


def test_concurrent_matches_commit_unique_session_indices(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Concurrent", mode="research", seed=11),
    )
    from src import app

    def run(*args, **kwargs):
        time.sleep(0.05)
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(
            pool.map(
                lambda fixture: workspace.run_match(*fixture, fast=True),
                [("Brazil", "Argentina"), ("France", "Germany")],
            )
        )
    assert {report["match_id"].split("-", 1)[0] for report in reports} == {
        "0001",
        "0002",
    }
    assert ProductWorkspace.load(tmp_path).status()["matches_played"] == 2


def test_failed_match_is_audited_and_its_seed_is_not_reused(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Failure Audit", mode="research", seed=20),
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


def test_post_simulation_reporting_failure_preserves_run_provenance(
    tmp_path, monkeypatch
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Report Failure", mode="research", seed=30),
    )
    from src import app
    from src.product import reporting

    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: _Summary())
    monkeypatch.setattr(
        reporting,
        "write_match_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("html failure")),
    )
    with pytest.raises(RuntimeError, match="html failure"):
        workspace.run_match("Brazil", "Argentina", fast=True)
    run = workspace._session()["runs"][0]
    assert run["state"] == "failed"
    assert run["simulation_completed_at"]
    assert run["successful_provider_calls"] == 0


def test_paired_workspace_persists_synchronized_dual_replay(tmp_path, monkeypatch):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Pair Replay", mode="research", seed=11),
    )
    from src import app

    calls = 0

    def fake_match(home, away, **kwargs):
        nonlocal calls
        calls += 1
        path = tmp_path / "outputs/ball_log" / f"{kwargs['stage_name']}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        t_sec = 10 * calls
        path.write_text(
            "\n".join(
                [
                    json.dumps({"meta": {"home": home, "away": away}}),
                    json.dumps(
                        {
                            "type": "pass",
                            "t_sec": t_sec,
                            "team": home,
                            "from": f"A{calls}",
                            "to": f"B{calls}",
                            "kind": "ground",
                            "from_xy": [0.2, 0.3],
                            "land_xy": [0.7, 0.4],
                            "outcome": "COMPLETE",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return _Summary(ball_log_path=str(path))

    monkeypatch.setattr(app, "run_micro_match", fake_match)
    baseline_plan = MatchPlan(
        experience="tactical_lab",
        home_tactic="balanced",
        away_tactic="low_block_counter",
    )
    treatment_plan = MatchPlan(
        experience="tactical_lab",
        home_tactic="gegenpress",
        away_tactic="low_block_counter",
        reuse_last_seed=True,
    )
    _, treatment = workspace.run_paired_matches(
        "Brazil",
        "Argentina",
        fast=True,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        seed=77,
        transaction_id="pair-replay-001",
    )

    comparison = json.loads(
        Path(treatment["comparison_path"]).read_text(encoding="utf-8")
    )
    assert comparison["paired_replay"]["available"]
    assert [frame["t_sec"] for frame in comparison["paired_replay"]["frames"]] == [
        10,
        20,
    ]
    assert comparison["paired_replay"]["event_correspondence_authorized"] is False
    dashboard = Path(treatment["comparison_dashboard_path"]).read_text(encoding="utf-8")
    assert 'data-testid="paired-replay-panel"' in dashboard
    assert dashboard.count('<svg viewBox="0 0 1000 640"') == 2
    assert "只按比赛时钟同步，不进行动作一一配对" in dashboard


def test_manager_decision_preview_is_zero_write_and_matches_submission(
    tmp_path,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Preview", mode="research", seed=11),
    )
    created = workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    promised_id = created["manager_squad"]["players"][0]["player_id"]
    workspace.set_player_role_promises(
        PlayerPromisePlan((PlayerRolePromise(promised_id, "core"),))
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    before = workspace.session_path.read_bytes()
    decision = ManagerDecision(
        team="Brazil",
        tactic="gegenpress",
        rotation="balanced",
    )

    preview = workspace.preview_manager_decision(
        decision,
        fixture_id=fixture_id,
    )

    assert workspace.session_path.read_bytes() == before
    unchanged = workspace.season_status()
    assert unchanged["revision"] == 1
    assert unchanged["next_manager_fixture"]["manager_decision"] is None
    assert preview["base_revision"] == 1
    assert preview["lineup"] == {
        "available": True,
        "source": "automatic",
        "starters": 11,
        "bench": 5,
    }
    assert preview["engine_tradeoff"] == {
        "status_penalty": 1.25,
        "rotation_level": 0.5,
        "boundary": "fixed engine mechanics, not a match-outcome estimate",
    }
    assert preview["season_commitment_projection"]["recorded_matches"] == 1
    assert preview["player_promise_projection"]["completed_matches"] == 1
    assert "no score" in preview["claim_boundary"]

    with pytest.raises(ValueError, match="preview is stale"):
        workspace.set_manager_decision(
            decision,
            fixture_id=fixture_id,
            expected_revision=preview["base_revision"] + 1,
        )
    assert workspace.session_path.read_bytes() == before
    submitted = workspace.set_manager_decision(
        decision,
        fixture_id=fixture_id,
        expected_revision=preview["base_revision"],
    )
    assert (
        submitted["next_manager_fixture"]["manager_decision"]
        == preview["normalized_decision"]
    )
    assert (
        submitted["next_manager_fixture"]["opponent_preparation"]
        == (preview["opponent_preparation"])
    )
    pending_ledger = submitted["manager_decision_ledger"]
    assert pending_ledger["summary"]["decisions"] == 1
    assert pending_ledger["summary"]["world_model_advisor"]["advised_decisions"] == 0
    assert (
        pending_ledger["summary"]["world_model_advisor"]["causal_effect_authorized"]
        is False
    )
    assert pending_ledger["entries"][0]["lifecycle_state"] == (
        "frozen_awaiting_execution"
    )
    assert (
        pending_ledger["entries"][0]["long_term_accounting"]["evidence_state"]
        == "hypothetical_if_counted"
    )


def test_manager_execution_evidence_loading_is_bounded_to_recent_eight(
    tmp_path,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(tmp_path, StudioConfig(name="Ledger Window"))
    season = {
        "fixtures": [
            {
                "fixture_id": f"fixture-{index:02d}",
                "state": "completed",
                "home": "Brazil",
                "away": f"Opponent {index}",
                "manager_decision": {},
                "report": None,
            }
            for index in range(10)
        ]
    }

    evidence = workspace._manager_execution_evidence(
        season,
        manager_team="Brazil",
    )

    assert list(evidence) == [f"fixture-{index:02d}" for index in range(2, 10)]
    assert all(
        row
        == {
            "schema_version": 1,
            "available": False,
            "reason": "report_reference_unavailable",
        }
        for row in evidence.values()
    )


def test_world_model_advice_is_persisted_previewed_and_explicitly_adopted(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    (tmp_path / "data/evaluation/manager_advisor_protocol_v1.json").write_bytes(
        (
            Path(__file__).resolve().parents[1]
            / "data/evaluation/manager_advisor_protocol_v1.json"
        ).read_bytes()
    )
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Manager advisor", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        lambda **kwargs: _manager_advice_packet(),
    )

    advice = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert advice["available"]
    assert advice["issued_revision"] == 1
    assert advice["match_seed"] == 112
    assert advice["recommended_tactic"] == "team_identity"
    assert len(advice["candidates"]) == len(PLAYABLE_TACTICS)
    persisted = ProductWorkspace.load(tmp_path).season_status()
    assert persisted["revision"] == 1
    assert persisted["next_manager_fixture"]["manager_decision_advice"] == {
        key: value for key, value in advice.items() if key != "available"
    }

    decision = ManagerDecision(team="Brazil", tactic="team_identity")
    preview = workspace.preview_manager_decision(decision, fixture_id=fixture_id)
    assert preview["world_model_advice"]["current"] is True
    assert preview["world_model_advice"]["selected_alignment"] is True
    aligned_comparison = preview["world_model_advice"]["comparison"]
    assert aligned_comparison["aligned_with_recommendation"] is True
    assert aligned_comparison["authority"]["level"] == "exploratory_only"
    assert aligned_comparison["authority"]["automatic_adoption_authorized"] is False
    assert (
        aligned_comparison["recommended_minus_selected"]["risk_adjusted_value"] == 0.0
    )
    submitted = workspace.set_manager_decision(
        decision,
        fixture_id=fixture_id,
        expected_revision=preview["base_revision"],
        advice_adoption={
            "schema_version": 1,
            "advice_identity": advice["advice_identity"],
            "intent": "adopt_recommendation",
        },
    )

    adopted = submitted["next_manager_fixture"]["manager_advice_adoption"]
    assert adopted["aligned_with_recommendation"] is True
    assert adopted["intent"] == "adopt_recommendation"
    ledger_advice = submitted["manager_decision_ledger"]["entries"][0][
        "world_model_decision_support"
    ]
    assert ledger_advice["available"] is True
    assert (
        ledger_advice["adoption"]["adoption_identity"] == adopted["adoption_identity"]
    )
    pending_trace = submitted["manager_decision_ledger"]["entries"][0][
        "advisor_execution_trace"
    ]
    assert pending_trace["end_to_end_state"] == "awaiting_execution"
    assert pending_trace["runtime_binding"] == {
        "status": "awaiting_execution",
        "reason": "fixture_not_completed",
    }
    assert pending_trace["direct_recommendation_execution"] is False
    advisor_summary = submitted["manager_decision_ledger"]["summary"][
        "world_model_advisor"
    ]
    assert advisor_summary["advised_decisions"] == 1
    assert advisor_summary["adopted_recommendation"] == 1
    assert advisor_summary["reviewed_then_selected"] == 0
    assert advisor_summary["explicit_interaction_coverage"] == 1.0
    assert advisor_summary["outcome_effect_estimate"] is None
    assert len(advisor_summary["evidence_identity"]) == 64

    reviewed_advice = workspace.request_manager_decision_advice(
        fixture_id=fixture_id,
    )
    reviewed_preview = workspace.preview_manager_decision(
        ManagerDecision(team="Brazil", tactic="gegenpress"),
        fixture_id=fixture_id,
    )
    reviewed_comparison = reviewed_preview["world_model_advice"]["comparison"]
    assert reviewed_comparison["selected_tactic"] == "gegenpress"
    assert reviewed_comparison["recommended_tactic"] == "team_identity"
    assert reviewed_comparison["aligned_with_recommendation"] is False
    assert (
        reviewed_comparison["recommended_minus_selected"]["risk_adjusted_value"] > 0.0
    )
    reviewed = workspace.set_manager_decision(
        ManagerDecision(team="Brazil", tactic="gegenpress"),
        fixture_id=fixture_id,
        expected_revision=reviewed_preview["base_revision"],
        advice_adoption={
            "schema_version": 1,
            "advice_identity": reviewed_advice["advice_identity"],
            "intent": "reviewed_then_selected",
        },
    )
    manual_adoption = reviewed["next_manager_fixture"]["manager_advice_adoption"]
    assert manual_adoption["selected_tactic"] == "gegenpress"
    assert manual_adoption["recommended_tactic"] == "team_identity"
    assert manual_adoption["aligned_with_recommendation"] is False
    assert manual_adoption["intent"] == "reviewed_then_selected"
    reviewed_summary = reviewed["manager_decision_ledger"]["summary"][
        "world_model_advisor"
    ]
    assert reviewed_summary["adopted_recommendation"] == 0
    assert reviewed_summary["reviewed_then_selected"] == 1
    assert reviewed_summary["descriptive_alignment_rate"] == 0.0
    protocol_evidence = workspace.status()["evidence"]["manager_advisor_adoption"]
    assert protocol_evidence["available"] is True
    assert protocol_evidence["fixed_information_windows"] == [12, 24, 48]
    assert protocol_evidence["results_available"] is False
    assert protocol_evidence["causal_effect_authorized"] is False
    assert protocol_evidence["promotion_authorized"] is False


def test_world_model_advice_stale_link_and_stable_mode_fail_closed(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    stable = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Stable advisor boundary", mode="stable", seed=11),
    )
    stable.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    stable_fixture = stable.season_status()["next_manager_fixture"]["fixture_id"]
    before = stable.session_path.read_bytes()
    unavailable = stable.request_manager_decision_advice(fixture_id=stable_fixture)
    assert unavailable["available"] is False
    assert unavailable["reason"] == "stable_mode_has_no_world_model_advisor"
    assert stable.session_path.read_bytes() == before

    session = json.loads(stable.session_path.read_text(encoding="utf-8"))
    session["mode"] = "research"
    stable.session_path.write_text(json.dumps(session), encoding="utf-8")
    workspace = ProductWorkspace.load(tmp_path)
    before_quality_gate = workspace.session_path.read_bytes()
    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        lambda **kwargs: {
            "schema_version": 1,
            "available": False,
            "reason": "candidate_quality_gate_closed",
        },
    )
    quality_closed = workspace.request_manager_decision_advice(
        fixture_id=stable_fixture,
    )
    assert quality_closed["available"] is False
    assert quality_closed["reason"] == "candidate_quality_gate_closed"
    assert workspace.session_path.read_bytes() == before_quality_gate
    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        lambda **kwargs: _manager_advice_packet(),
    )
    advice = workspace.request_manager_decision_advice(fixture_id=stable_fixture)
    workspace.set_manager_decision(
        ManagerDecision(team="Brazil", tactic="gegenpress"),
        fixture_id=stable_fixture,
        expected_revision=1,
    )
    with pytest.raises(ValueError, match="advice is stale"):
        workspace.set_manager_decision(
            ManagerDecision(team="Brazil", tactic="team_identity"),
            fixture_id=stable_fixture,
            expected_revision=2,
            advice_adoption={
                "schema_version": 1,
                "advice_identity": advice["advice_identity"],
                "intent": "adopt_recommendation",
            },
        )


def test_adopted_world_model_advice_binds_to_direct_tactical_execution(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor execution chain", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        lambda **kwargs: _manager_advice_packet(),
    )
    advice = workspace.request_manager_decision_advice(fixture_id=fixture_id)
    workspace.set_manager_decision(
        ManagerDecision(team="Brazil", tactic=advice["recommended_tactic"]),
        fixture_id=fixture_id,
        expected_revision=1,
        advice_adoption={
            "schema_version": 1,
            "advice_identity": advice["advice_identity"],
            "intent": "adopt_recommendation",
        },
    )

    from src import app

    def run(home, away, **kwargs):
        summary = _Summary()
        summary.manager_effects = {
            side: {
                "base_status": 70.0,
                "effective_status": 70.0,
                "rotation_status_penalty": 0.0,
                "fatigue_load_multiplier": 1.0,
                "club_fatigue_load_factor": kwargs[f"{side}_fatigue_load_factor"],
            }
            for side in ("home", "away")
        }
        summary.tactical_execution = _runtime_tactical_execution(
            home,
            away,
            kwargs["home_tactic"],
            kwargs["away_tactic"],
        )
        return summary

    monkeypatch.setattr(app, "run_micro_match", run)
    monkeypatch.setattr(
        "src.simulation.cross_match_state.recover_persisted_teams",
        lambda *args, **kwargs: None,
    )
    completed = workspace.play_next_matchday(expected_revision=2)

    entry = completed["manager_decision_ledger"]["entries"][0]
    trace = entry["advisor_execution_trace"]
    assert "tactical_binding" in entry["execution"], entry["execution"]
    binding = entry["execution"]["tactical_binding"]
    assert binding["available"] is True
    assert binding["applied_tactic"] == advice["recommended_tactic"]
    assert len(binding["initial_vector"]) == 22
    assert trace["end_to_end_state"] == "recommendation_executed"
    assert trace["direct_recommendation_execution"] is True
    assert trace["runtime_binding"]["status"] == "verified"
    assert trace["runtime_binding"]["binding_identity"] == binding["binding_identity"]
    advisor_summary = completed["manager_decision_ledger"]["summary"][
        "world_model_advisor"
    ]
    assert advisor_summary["verified_tactical_bindings"] == 1
    assert advisor_summary["direct_recommendation_executions"] == 1
    assert advisor_summary["outcome_effect_estimate"] is None

    tampered = json.loads(json.dumps(completed["manager_decision_ledger"]))
    tampered_entry = tampered["entries"][0]
    tampered_entry["advisor_execution_trace"]["direct_recommendation_execution"] = False
    entry_source = dict(tampered_entry)
    entry_source.pop("entry_identity")
    tampered_entry["entry_identity"] = hashlib.sha256(
        json.dumps(
            entry_source,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    ledger_source = dict(tampered)
    ledger_source.pop("ledger_identity")
    tampered["ledger_identity"] = hashlib.sha256(
        json.dumps(
            ledger_source,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(
        ValueError,
        match="manager advisor execution trace replay mismatch",
    ):
        validate_manager_decision_ledger(tampered)


def test_world_model_advice_inference_releases_session_lease(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor short lease", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    lease_states = []

    def generate(**kwargs):
        from src.infrastructure import FileLease

        lease_states.append(FileLease.is_held(workspace.session_lease_path))
        return _manager_advice_packet()

    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        generate,
    )

    advice = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert advice["available"] is True
    assert lease_states == [False]


def test_world_model_advice_reuses_identical_current_inputs_without_inference(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor retry reuse", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    calls = 0

    def generate(**kwargs):
        nonlocal calls
        calls += 1
        return _manager_advice_packet()

    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        generate,
    )
    created = workspace.request_manager_decision_advice(fixture_id=fixture_id)
    persisted_bytes = workspace.session_path.read_bytes()
    reused = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert calls == 1
    assert reused["available"] is True
    assert reused["reused"] is True
    assert reused["advice_identity"] == created["advice_identity"]
    assert reused["issued_revision"] == created["issued_revision"] == 1
    assert workspace.session_path.read_bytes() == persisted_bytes
    assert workspace.season_status()["revision"] == 1


def test_concurrent_identical_world_model_advice_requests_converge(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor duplicate merge", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    barrier = threading.Barrier(2)
    calls = 0
    calls_lock = threading.Lock()

    def generate(**kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        barrier.wait(timeout=10.0)
        return _manager_advice_packet()

    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        generate,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: workspace.request_manager_decision_advice(
                    fixture_id=fixture_id,
                ),
                range(2),
            )
        )

    assert calls == 2
    assert all(row["available"] is True for row in results)
    assert {row["advice_identity"] for row in results} == {
        results[0]["advice_identity"]
    }
    assert sum(row.get("reused") is True for row in results) == 1
    status = workspace.season_status()
    assert status["revision"] == 1
    assert (
        status["next_manager_fixture"]["manager_decision_advice"]["advice_identity"]
        == results[0]["advice_identity"]
    )


def test_world_state_change_recomputes_instead_of_reusing_advice(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor reuse invalidation", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    calls = 0

    def generate(**kwargs):
        nonlocal calls
        calls += 1
        return _manager_advice_packet()

    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        generate,
    )
    created = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    def changed_capture(_root, teams):
        return {
            team: {
                "source": "deterministic_roster_baseline",
                "source_identity": "b" * 64,
            }
            for team in teams
        }

    monkeypatch.setattr("src.product.workspace.capture_world_state", changed_capture)
    recomputed = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert calls == 2
    assert recomputed["available"] is True
    assert "reused" not in recomputed
    assert recomputed["advice_identity"] != created["advice_identity"]
    assert recomputed["base_revision"] == 1
    assert recomputed["issued_revision"] == 2
    assert workspace.season_status()["revision"] == 2


def test_world_model_advice_does_not_overwrite_concurrent_manager_decision(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor optimistic commit", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    committed_bytes = []

    def generate(**kwargs):
        workspace.set_manager_decision(
            ManagerDecision(team="Brazil", tactic="gegenpress"),
            fixture_id=fixture_id,
            expected_revision=0,
        )
        committed_bytes.append(workspace.session_path.read_bytes())
        return _manager_advice_packet()

    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        generate,
    )

    unavailable = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert unavailable["available"] is False
    assert unavailable["reason"] == "advice_inputs_changed_during_generation"
    assert unavailable["retryable"] is True
    assert unavailable["claim_boundary"].startswith("research advisor unavailable")
    assert workspace.session_path.read_bytes() == committed_bytes[0]
    status = workspace.season_status()
    assert status["revision"] == 1
    assert status["next_manager_fixture"]["manager_decision"]["tactic"] == (
        "gegenpress"
    )
    assert status["next_manager_fixture"].get("manager_decision_advice") is None


def test_world_model_advice_rejects_world_state_drift_without_writing(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor state drift", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    before = workspace.session_path.read_bytes()
    captures = 0

    def capture(_root, teams):
        nonlocal captures
        captures += 1
        suffix = "a" if captures == 1 else "b"
        return {
            team: {
                "source": "deterministic_roster_baseline",
                "source_identity": suffix * 64,
            }
            for team in teams
        }

    monkeypatch.setattr("src.product.workspace.capture_world_state", capture)
    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        lambda **kwargs: _manager_advice_packet(),
    )

    unavailable = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert unavailable["available"] is False
    assert unavailable["reason"] == "advice_inputs_changed_during_generation"
    assert captures == 2
    assert workspace.session_path.read_bytes() == before


def test_world_model_advice_rejects_checkpoint_drift_without_writing(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Advisor checkpoint drift", mode="research", seed=11),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        )
    )
    fixture_id = workspace.season_status()["next_manager_fixture"]["fixture_id"]
    before = workspace.session_path.read_bytes()

    def generate(**kwargs):
        (
            tmp_path / "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
        ).write_bytes(b"replaced-after-inference")
        return _manager_advice_packet()

    monkeypatch.setattr(
        workspace,
        "_generate_manager_decision_advice_packet",
        generate,
    )

    unavailable = workspace.request_manager_decision_advice(fixture_id=fixture_id)

    assert unavailable["available"] is False
    assert unavailable["reason"] == "advice_inputs_changed_during_generation"
    assert unavailable["retryable"] is True
    assert workspace.session_path.read_bytes() == before


def test_workspace_season_completes_matchday_with_stable_continuity_ids(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Season", mode="research", seed=11),
    )
    created = workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
            manager_resources=ClubResourcePlan(4, 1, 1),
        )
    )
    assert created["progress"] == {"completed": 0, "total": 6}
    assert created["matchday_command_center"]["phase"] == "decision_required"
    assert (
        created["matchday_command_center"]["primary_action"]["id"] == "submit_decision"
    )
    assert created["matchday_command_center"]["prematch_intelligence"]["available"]
    assert created["matchday_command_center"]["prematch_intelligence"][
        "claim_boundary"
    ].endswith("real-world recommendation")
    strategy_briefing = created["matchday_command_center"]["club_strategy"]
    assert strategy_briefing["available"]
    assert strategy_briefing["manager"]["team"] == "Brazil"
    assert strategy_briefing["opponent_identity_tactic"] == "balanced"
    from src import app

    observed = []
    recoveries = []

    def run(*args, **kwargs):
        observed.append(kwargs)
        summary = _Summary()
        summary.manager_effects = {
            side: {
                "base_status": 70.0,
                "effective_status": 70.0,
                "rotation_status_penalty": 0.0,
                "fatigue_load_multiplier": 1.0,
                "club_fatigue_load_factor": kwargs[f"{side}_fatigue_load_factor"],
            }
            for side in ("home", "away")
        }
        return summary

    monkeypatch.setattr(app, "run_micro_match", run)
    monkeypatch.setattr(
        "src.simulation.cross_match_state.recover_persisted_teams",
        lambda base_dir, team_ids, **kwargs: recoveries.append(
            {
                "base_dir": base_dir,
                "team_ids": list(team_ids),
                **kwargs,
            }
        ),
    )
    decided = workspace.set_manager_decision(
        ManagerDecision(
            team="Brazil",
            tactic="gegenpress",
            rotation="strongest",
            in_match_plan=InMatchPlan(
                "Brazil",
                (
                    InMatchInstruction(
                        "protect-lead-70",
                        70,
                        "leading",
                        tactic="low_block_counter",
                    ),
                ),
            ),
        )
    )
    assert decided["revision"] == 1
    assert decided["matchday_command_center"]["phase"] == "ready_to_advance"
    frozen = decided["next_manager_fixture"]["manager_decision"]["lineup"]
    assert frozen["source"] == "automatic"
    assert len(frozen["starters"]) == 11
    assert len(frozen["roster_fingerprint"]) == 64
    assert (
        decided["next_manager_fixture"]["manager_decision"]["in_match_plan"][
            "instructions"
        ][0]["rule_id"]
        == "protect-lead-70"
    )
    assert (
        decided["next_manager_fixture"]["opponent_preparation"]["state"]
        == "insufficient_evidence"
    )
    assert (
        decided["next_manager_fixture"]["opponent_preparation"]["selected_tactic"]
        == "team_identity"
    )
    frozen_strategy = decided["matchday_command_center"]["club_strategy"]
    assert frozen_strategy["applied_preview"]["manager_override_team"] == "Brazil"
    assert "gegenpress" in {
        frozen_strategy["applied_preview"]["home_tactic"],
        frozen_strategy["applied_preview"]["away_tactic"],
    }
    status = workspace.play_next_matchday(expected_revision=1)
    assert status["progress"]["completed"] == 2
    assert status["next_matchday"] == 2
    assert status["matchday_command_center"]["last_result"]["score"] == {
        "home": 2,
        "away": 1,
    }
    assert status["matchday_command_center"]["last_result"]["dashboard"].endswith(
        ".html"
    )
    managed_completed = next(
        row
        for row in status["fixtures"]
        if row.get("manager_decision") is not None and row["state"] == "completed"
    )
    assert (
        sum(
            row.get("world_state_transition") is not None
            for row in status["fixtures"]
            if row["matchday"] == 1
        )
        == 1
    )
    world_state = managed_completed["world_state_transition"]
    assert world_state["recovery_complete"] is True
    assert len(world_state["transition_identity"]) == 64
    assert world_state["phases"]["before_match"]["Brazil"]["source"] == (
        "deterministic_roster_baseline"
    )
    assert world_state["phases"]["after_match"]["Brazil"]["source"] == (
        "deterministic_roster_baseline"
    )
    debrief = status["matchday_command_center"]["postmatch_debrief"]
    assert debrief["available"]
    assert debrief["decision"]["tactic"] == "gegenpress"
    assert debrief["causal_outcome_attribution"] is False
    ledger = status["manager_decision_ledger"]
    assert ledger["summary"]["decisions"] == 1
    assert ledger["summary"]["direct_execution_coverage"] == 1.0
    closed = ledger["entries"][0]
    assert closed["lifecycle_state"] == "executed_with_direct_evidence"
    assert closed["execution"]["evidence_grade"] == "direct_runtime_execution"
    assert closed["observed_result"]["descriptive_only"] is True
    assert closed["long_term_accounting"]["evidence_state"] == ("completed_evidence")
    state_delta = closed["long_term_accounting"]["persistent_team_state_delta"]
    assert state_delta["available"] is True
    assert state_delta["recovery_complete"] is True
    assert state_delta["transition_identity"] == world_state["transition_identity"]
    assert state_delta["match_delta"]["metrics_delta"]["team_fatigue_ema"] == 0
    assert len(closed["decision_identity"]) == 64
    assert len(closed["entry_identity"]) == 64
    assert (
        ProductWorkspace.load(tmp_path).season_status()["manager_decision_ledger"][
            "ledger_identity"
        ]
        == ledger["ledger_identity"]
    )
    tampered_debrief = {**debrief, "match_id": "tampered-match"}
    tampered_ledger = build_manager_decision_ledger(
        workspace._session()["season"],
        execution_by_fixture={closed["fixture_id"]: tampered_debrief},
    )
    assert tampered_ledger["entries"][0]["lifecycle_state"] == (
        "executed_evidence_unavailable"
    )
    assert tampered_ledger["entries"][0]["execution"]["reason"] == (
        "execution_identity_or_boundary_mismatch"
    )
    assert sum(row["played"] for row in status["standings"]) == 4
    assert all(call["continuity"] is True for call in observed)
    assert all(
        call["home_tactic"] is not None and call["away_tactic"] is not None
        for call in observed
    )
    managed = [
        call
        for call in observed
        if call["home_tactic"] == "gegenpress" or call["away_tactic"] == "gegenpress"
    ]
    assert len(managed) == 1
    ai_only = [call for call in observed if call not in managed]
    assert len(ai_only) == 1
    assert ai_only[0]["home_tactic"] == "balanced"
    assert ai_only[0]["away_tactic"] == "balanced"
    assert "strongest" in {managed[0]["home_rotation"], managed[0]["away_rotation"]}
    if managed[0]["home_rotation"] == "strongest":
        assert managed[0]["home_fatigue_load_factor"] == 0.96
        assert managed[0]["away_fatigue_load_factor"] == 1.0
    else:
        assert managed[0]["away_fatigue_load_factor"] == 0.96
        assert managed[0]["home_fatigue_load_factor"] == 1.0
    active_lineup = managed[0]["home_lineup"] or managed[0]["away_lineup"]
    assert active_lineup == frozen
    active_plan = managed[0]["home_in_match_plan"] or managed[0]["away_in_match_plan"]
    assert active_plan["team"] == "Brazil"
    assert active_plan["instructions"][0]["condition"] == "leading"
    assert {call["continuity_id"] for call in observed} == {
        "season-0001:md01-fx01",
        "season-0001:md01-fx02",
    }
    assert all(call["stage_name"].startswith(("0001-", "0002-")) for call in observed)
    persisted = ProductWorkspace.load(tmp_path).season_status()
    assert persisted["progress"]["completed"] == 2
    assert recoveries[:2] == [
        {
            "base_dir": str(tmp_path),
            "team_ids": ["Argentina", "France", "Germany"],
            "rest_units": 1.0,
            "transaction_id": "season-0001:recovery:md01",
        },
        {
            "base_dir": str(tmp_path),
            "team_ids": ["Brazil"],
            "rest_units": 1.48,
            "medical_recovery_credit": 0.15,
            "transaction_id": "season-0001:recovery:md01",
        },
    ]

    second = workspace.set_manager_decision(
        ManagerDecision(
            team="Brazil",
            tactic="gegenpress",
            rotation="balanced",
        )
    )
    assert (
        second["next_manager_fixture"]["opponent_preparation"]["evidence"][
            "observation_count"
        ]
        == 1
    )
    workspace.play_next_matchday(expected_revision=2)
    third = workspace.set_manager_decision(
        ManagerDecision(
            team="Brazil",
            tactic="tiki_taka",
            rotation="balanced",
        )
    )
    preparation = third["next_manager_fixture"]["opponent_preparation"]
    assert preparation["state"] == "adapted"
    assert preparation["evidence"]["tactic_counts"] == {"gegenpress": 2}
    assert preparation["selected_tactic"] == "direct_vertical"
    managed_fixture = third["next_manager_fixture"]
    before_third_day = len(observed)
    completed = workspace.play_next_matchday(expected_revision=3)
    third_day_calls = observed[before_third_day:]
    applied = next(
        call
        for call in third_day_calls
        if call["home_tactic"] == "tiki_taka" or call["away_tactic"] == "tiki_taka"
    )
    if managed_fixture["home"] == "Brazil":
        assert applied["home_tactic"] == "tiki_taka"
        assert applied["away_tactic"] == "direct_vertical"
    else:
        assert applied["away_tactic"] == "tiki_taka"
        assert applied["home_tactic"] == "direct_vertical"
    assert completed["state"] == "complete"
    final_accounting = completed["manager_decision_ledger"]["entries"][0][
        "long_term_accounting"
    ]
    assert {
        row["status"]
        for row in final_accounting["season_commitments"]["after"]["entries"]
    } <= {"fulfilled", "missed", "excused"}
    session = workspace._session()
    recorded = next(
        match
        for match in session["matches"]
        if (match["match_plan"].get("competition") or {}).get("fixture_id")
        == managed_fixture["fixture_id"]
    )
    assert recorded["match_plan"]["opponent_preparation"] == preparation
    assert (
        recorded["match_plan"]["club_strategy"]["opponent_adaptation_applied"] is True
    )
    assert recorded["match_plan"]["club_strategy"]["snapshot_identity"] == (
        strategy_identity(session["season"]["club_strategies"])
    )
    persisted_strategy_session = workspace._session()
    tampered_strategy_session = json.loads(json.dumps(persisted_strategy_session))
    target_record = next(
        match
        for match in tampered_strategy_session["matches"]
        if (match["match_plan"].get("competition") or {}).get("fixture_id")
        == managed_fixture["fixture_id"]
    )
    target_record["match_plan"]["club_strategy"]["away_tactic"] = "balanced"
    workspace.session_path.write_text(
        json.dumps(tampered_strategy_session),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed match club strategy mismatch"):
        workspace.status()
    workspace.session_path.write_text(
        json.dumps(persisted_strategy_session),
        encoding="utf-8",
    )
    assert recorded["match_plan"]["club_support"]["effects"] == {
        "baseline_rest_units": 1.0,
        "manager_rest_units": 1.48,
        "medical_recovery_credit_per_matchday": 0.15,
        "fatigue_load_factor": 0.96,
        "current_match_status_bonus": 0.0,
    }
    workspace.create_season(
        SeasonPlan(
            teams=("Brazil", "Argentina", "France", "Germany"),
            fast=True,
            manager_team="Brazil",
        ),
        replace=True,
    )
    archived = workspace.status()["season_history"][0]
    adapted_entry = next(
        entry
        for entry in archived["manager_profile"]["journal"]
        if (entry.get("opponent_preparation") or {}).get("state") == "adapted"
    )
    assert adapted_entry["opponent_preparation"] == preparation
    archived_world_entry = next(
        entry
        for entry in archived["manager_profile"]["journal"]
        if entry.get("world_state_transition") is not None
    )
    assert archived_world_entry["world_state_transition"]["transition_identity"]
    clean_archive_session = workspace._session()
    advice_archive_session = json.loads(json.dumps(clean_archive_session))
    advice_archive = advice_archive_session["season_history"][0]
    advice_entry = advice_archive["manager_profile"]["journal"][0]
    advice_home = (
        advice_archive["plan"]["manager_team"]
        if advice_entry["venue"] == "home"
        else advice_entry["opponent"]
    )
    advice_away = (
        advice_entry["opponent"]
        if advice_entry["venue"] == "home"
        else advice_archive["plan"]["manager_team"]
    )
    archived_advice = build_manager_decision_advice(
        packet=_manager_advice_packet(),
        season_id=advice_archive["season_id"],
        fixture_id=advice_entry["fixture_id"],
        home=advice_home,
        away=advice_away,
        manager_team=advice_archive["plan"]["manager_team"],
        mode="research",
        match_seed=(
            advice_archive["seed"]
            + advice_entry["matchday"] * 100
            + advice_entry["order"]
        ),
        base_revision=0,
        checkpoint_artifact=(
            "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
        ),
        checkpoint_sha256=hashlib.sha256(b"model").hexdigest(),
        state_snapshots={
            advice_home: {
                "source": "deterministic_roster_baseline",
                "source_identity": "a" * 64,
            },
            advice_away: {
                "source": "deterministic_roster_baseline",
                "source_identity": "b" * 64,
            },
        },
    )
    archived_adoption = build_manager_advice_adoption(
        archived_advice,
        selected_tactic=advice_entry["tactic"],
        intent=(
            "adopt_recommendation"
            if advice_entry["tactic"] == archived_advice["recommended_tactic"]
            else "reviewed_then_selected"
        ),
    )
    advice_entry["manager_decision_advice"] = archived_advice
    advice_entry["manager_advice_adoption"] = archived_adoption
    assert (
        workspace._season_history_view(advice_archive_session)[0]["manager_profile"][
            "journal"
        ][0]["manager_advice_adoption"]
        == archived_adoption
    )
    tampered_advice_archive = json.loads(json.dumps(advice_archive_session))
    tampered_advice_archive["season_history"][0]["manager_profile"]["journal"][0][
        "manager_advice_adoption"
    ]["selected_rank"] = 99
    with pytest.raises(ValueError, match="adoption replay mismatch"):
        workspace._season_history_view(tampered_advice_archive)
    tampered_world_session = json.loads(json.dumps(clean_archive_session))
    transition = next(
        entry["world_state_transition"]
        for entry in tampered_world_session["season_history"][0]["manager_profile"][
            "journal"
        ]
        if entry.get("world_state_transition") is not None
    )
    manager_team = tampered_world_session["season_history"][0]["plan"]["manager_team"]
    transition["match_delta"][manager_team]["metrics_delta"]["team_fatigue_ema"] = 99.0
    transition_source = dict(transition)
    transition_source.pop("transition_identity")
    transition["transition_identity"] = hashlib.sha256(
        json.dumps(
            transition_source,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    workspace.session_path.write_text(
        json.dumps(tampered_world_session),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="world-state transition replay mismatch"):
        workspace.status()
    workspace.session_path.write_text(
        json.dumps(clean_archive_session),
        encoding="utf-8",
    )
    tampered = workspace._session()
    target_entry = next(
        entry
        for entry in tampered["season_history"][0]["manager_profile"]["journal"]
        if (entry.get("opponent_preparation") or {}).get("state") == "adapted"
    )
    target_entry["opponent_preparation"]["selected_tactic"] = "balanced"
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="archived opponent preparation"):
        workspace.status()


def test_workspace_manager_debrief_rejects_report_outside_workspace(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Debrief Boundary", mode="research", seed=13),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            fast=True,
            manager_team="A",
        )
    )
    session = workspace._session()
    fixture = next(
        item
        for item in session["season"]["fixtures"]
        if "A" in {item["home"], item["away"]}
    )
    fixture.update(
        {
            "state": "completed",
            "match_id": "escaped-report",
            "report": str(tmp_path.parent / "outside-report.json"),
            "score": {"home": 0, "away": 0},
            "manager_decision": ManagerDecision(team="A").as_dict(),
        }
    )
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    debrief = workspace.season_status()["matchday_command_center"]["postmatch_debrief"]

    assert debrief == {
        "schema_version": 1,
        "available": False,
        "reason": "report_reference_unavailable",
    }


def test_workspace_archives_only_completed_season_before_starting_next(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Career", mode="research", seed=23),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    with pytest.raises(ValueError, match="only a completed"):
        workspace.create_season(
            SeasonPlan(teams=("A", "B", "C", "D"), manager_team="A"),
            replace=True,
        )
    assert workspace.status()["season_history"] == []

    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": {"home": 1, "away": 0},
            }
        )
        if "A" in {fixture["home"], fixture["away"]}:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    second = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="points_target",
            manager_points_target=5,
        ),
        replace=True,
    )
    status = workspace.status()

    assert second["season_id"] == "season-0002"
    assert second["manager_profile"]["objective"]["target_points"] == 5
    assert len(status["season_history"]) == 1
    assert status["season_history_summary"] == {
        "total": 1,
        "retained": 1,
        "limit": 12,
        "truncated": False,
    }
    archived = status["season_history"][0]
    assert archived["season_id"] == "season-0001"
    assert archived["manager_profile"]["objective"]["kind"] == "champion"
    assert len(archived["manager_profile"]["journal"]) == 3

    corrupted = workspace._session()
    corrupted["season_history"][0]["manager_profile"]["journal"] = "invalid"
    workspace.session_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="archived manager profile"):
        workspace.status()


def test_workspace_recruitment_atomically_changes_the_real_next_season_squad(
    tmp_path,
):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
    base_roster_path = tmp_path / "data/rosters/A.json"
    frozen_base = base_roster_path.read_bytes()
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Squad career", mode="research", seed=31),
    )
    with pytest.raises(ValueError, match="later season"):
        workspace.create_season(
            SeasonPlan(
                teams=("A", "B", "C", "D"),
                manager_team="A",
                manager_recruitment=RecruitmentPlan(
                    "market-0002-deadbeefdead",
                    (RecruitmentMove("candidate", "outgoing"),),
                ),
            )
        )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        manager_home = fixture["home"] == "A"
        manager_away = fixture["away"] == "A"
        score = {"home": 0, "away": 0}
        if manager_home:
            score = {"home": 2, "away": 0}
        elif manager_away:
            score = {"home": 0, "away": 2}
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": score,
            }
        )
        if manager_home or manager_away:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    market = workspace.recruitment_market("A")
    assert market == workspace.recruitment_market("A")
    assert market["season_index"] == 2
    assert market["balance_before_settlement"] == 8
    assert market["settlement_preview"]["type"] == "season_settlement"
    assert market["available_budget"] == min(8, market["club_balance"])
    candidate = market["candidates"][0]
    outgoing = next(
        player for player in market["outgoing_players"] if player["role"] != "GK"
    )
    plan = RecruitmentPlan(
        market["market_id"],
        (RecruitmentMove(candidate["player_id"], outgoing["player_id"]),),
    )
    second = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
            manager_recruitment=plan,
        ),
        replace=True,
    )

    assert second["season_id"] == "season-0002"
    assert second["recruitment_transaction"]["spent"] == candidate["recruitment_cost"]
    assert second["recruitment_transaction"]["plan"] == plan.as_dict()
    player_ids = {player["player_id"] for player in second["manager_squad"]["players"]}
    assert candidate["player_id"] in player_ids
    assert outgoing["player_id"] not in player_ids
    signed = next(
        player
        for player in second["manager_squad"]["players"]
        if player["player_id"] == candidate["player_id"]
    )
    assert signed["source"] == "gfs_fictional_recruitment_v1"
    assert signed["recruitment_cost"] == candidate["recruitment_cost"]
    assert base_roster_path.read_bytes() == frozen_base
    persisted = workspace._session()
    transaction = persisted["squad_registry"]["clubs"]["A"]["transactions"][0]
    assert transaction["season_id"] == "season-0002"
    finance = workspace.status()["club_finance"]
    assert finance["balance"] == market["club_balance"] - candidate["recruitment_cost"]
    assert finance["ledger_entry_count"] == 2
    assert [entry["type"] for entry in finance["ledger_entries"]] == [
        "season_settlement",
        "recruitment_charge",
    ]

    persisted = workspace._session()
    tampered_finance = json.loads(json.dumps(persisted))
    tampered_finance["finance_registry"]["clubs"]["A"]["entries"][0][
        "balance_after"
    ] += 1
    workspace.session_path.write_text(
        json.dumps(tampered_finance),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finance ledger replay mismatch"):
        workspace.status()
    workspace.session_path.write_text(json.dumps(persisted), encoding="utf-8")

    tampered = workspace._session()
    tampered["squad_registry"]["clubs"]["A"]["transactions"][0]["spent"] = 0
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        workspace.season_status()


def test_workspace_develops_players_from_exact_minutes_and_replays_the_source(
    tmp_path,
):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
        roster_path = tmp_path / f"data/rosters/{team}.json"
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        for player in roster["players"]:
            player["age"] = 20
        roster_path.write_text(json.dumps(roster), encoding="utf-8")
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Development career", mode="research", seed=32),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    session = workspace._session()
    reports = tmp_path / "outputs"
    reports.mkdir()
    for fixture in session["season"]["fixtures"]:
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": {"home": 1, "away": 1},
            }
        )
        if "A" in {fixture["home"], fixture["away"]}:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
        payload = {
            "match_id": fixture["fixture_id"],
            "match_plan": {
                "competition": {
                    "season_id": session["season"]["season_id"],
                    "fixture_id": fixture["fixture_id"],
                }
            },
            "raw_summary": {
                "player_stats": {
                    fixture["home"]: {
                        f"{fixture['home'].lower()}-00": {"minutes": 90.0},
                    },
                    fixture["away"]: {
                        f"{fixture['away'].lower()}-00": {"minutes": 90.0},
                    },
                }
            },
        }
        (reports / f"{fixture['fixture_id']}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    second = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
        ),
        replace=True,
    )
    effective = load_effective_roster(tmp_path, "A")
    players = {player["player_id"]: player for player in effective["players"]}
    assert players["a-00"]["age"] == 21
    assert players["a-01"]["age"] == 21
    assert players["a-00"]["abilities"]["tech"] == pytest.approx(0.52765)
    assert players["a-01"]["abilities"]["tech"] == pytest.approx(0.53925)
    assert second["season_id"] == "season-0002"

    persisted = workspace._session()
    evidence = persisted["season_history"][0]["player_development_evidence"]["A"]
    assert evidence["coverage"] == "complete"
    assert evidence["player_totals"]["a-00"] == {
        "minutes": 270.0,
        "appearances": 3,
    }
    transaction = persisted["player_development"]["clubs"]["A"]["transactions"][0]
    assert transaction["summary"]["evidence_coverage"] == "complete"
    assert transaction["sports_science_points"] == 2
    assert transaction["players"][0]["season_delta"] == pytest.approx(0.009)
    assert workspace.status()["player_development"]["transaction_count"] == 4

    tampered = json.loads(json.dumps(persisted))
    tampered["player_development"]["clubs"]["A"]["transactions"][0]["players"][0][
        "season_delta"
    ] = 0.0
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="player development source replay mismatch"):
        workspace.status()


def test_workspace_retention_retirement_and_academy_change_the_real_squad(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
        path = tmp_path / f"data/rosters/{team}.json"
        roster = json.loads(path.read_text(encoding="utf-8"))
        for player in roster["players"]:
            player["age"] = 24
            player["career"] = {
                "schema_version": 1,
                "contract_years_remaining": 2,
                "contract_source": "renewed",
            }
        if team == "A":
            roster["players"][1]["career"]["contract_years_remaining"] = 1
            roster["players"][2]["age"] = 39
        path.write_text(json.dumps(roster), encoding="utf-8")
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Lifecycle career", mode="research", seed=33),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": {"home": 1, "away": 1},
            }
        )
        if "A" in {fixture["home"], fixture["away"]}:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    preview = workspace.lifecycle_preview("A")
    assert preview["expiring_player_ids"] == ["a-01"]
    assert preview["retiring_player_ids"] == ["a-02"]
    retention = RetentionPlan(preview["cycle_id"], ())
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
            manager_retention=retention,
        ),
        replace=True,
    )

    effective = load_effective_roster(tmp_path, "A")
    player_ids = {player["player_id"] for player in effective["players"]}
    assert len(player_ids) == 16
    assert "a-01" not in player_ids and "a-02" not in player_ids
    academy = [
        player
        for player in effective["players"]
        if player.get("source") == "gfs_fictional_academy_v1"
    ]
    assert len(academy) == 2
    assert sorted(player["role"] for player in academy) == ["CB", "RB"]
    transaction = workspace._session()["player_lifecycle"]["clubs"]["A"][
        "transactions"
    ][0]
    assert transaction["summary"] == {
        "renewed": 0,
        "continued": 14,
        "new_signings": 0,
        "contract_releases": 1,
        "retirements": 1,
        "academy_promotions": 2,
    }

    persisted = workspace._session()
    tampered = json.loads(json.dumps(persisted))
    tampered["season"]["plan"]["manager_retention"]["renew_player_ids"] = ["a-01"]
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="player lifecycle season plan mismatch"):
        workspace.status()


def test_global_market_preserves_player_identity_across_three_seasons(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
        path = tmp_path / f"data/rosters/{team}.json"
        roster = json.loads(path.read_text(encoding="utf-8"))
        for player in roster["players"]:
            player["age"] = 24
            player["career"] = {
                "schema_version": 1,
                "contract_years_remaining": 2,
                "contract_source": "renewed",
            }
        if team == "A":
            roster["players"][1]["career"]["contract_years_remaining"] = 1
        path.write_text(json.dumps(roster), encoding="utf-8")

    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Global market", mode="research", seed=34),
    )

    def complete_active(manager_team):
        session = workspace._session()
        for fixture in session["season"]["fixtures"]:
            fixture.update(
                {
                    "state": "completed",
                    "match_id": fixture["fixture_id"],
                    "report": f"outputs/{fixture['fixture_id']}.json",
                    "score": {"home": 1, "away": 1},
                }
            )
            if manager_team in {fixture["home"], fixture["away"]}:
                fixture["manager_decision"] = {
                    "team": manager_team,
                    "tactic": "balanced",
                    "rotation": "balanced",
                }
        session["season"]["state"] = "complete"
        workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    complete_active("A")
    lifecycle = workspace.lifecycle_preview("A")
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
            manager_retention=RetentionPlan(lifecycle["cycle_id"], ()),
        ),
        replace=True,
    )
    assert workspace.status()["player_market"]["pool_size"] == 1

    complete_active("A")
    market = workspace.free_agent_market("B")
    assert market["candidate_count"] == 1
    assert market["candidates"][0]["player_id"] == "a-01"
    assert market["candidates"][0]["origin_team"] == "A"
    assert market["candidates"][0]["age"] == 25
    assert "quality" not in market["candidates"][0]
    baseline_width = market["candidates"][0]["observation"]["interval_width"]
    scouted = workspace.scout_free_agent("B", "a-01")
    assert scouted["scouting_budget"] == {"limit": 2, "used": 1, "remaining": 1}
    assert scouted["candidates"][0]["scouted"] is True
    assert scouted["candidates"][0]["observation"]["interval_width"] < baseline_width
    assert workspace.scout_free_agent("B", "a-01")["scouting_budget"]["used"] == 1
    safe_pool = workspace.status()["player_market"]["pool"][0]
    assert "player" not in safe_pool and "abilities" not in safe_pool
    persisted_report = workspace._session()
    tampered_report = json.loads(json.dumps(persisted_report))
    tampered_report["scouting_registry"]["reports"][0]["observation"][
        "estimated_quality"
    ] += 0.001
    workspace.session_path.write_text(json.dumps(tampered_report), encoding="utf-8")
    with pytest.raises(ValueError, match="scouting report source replay mismatch"):
        workspace.status()
    workspace.session_path.write_text(json.dumps(persisted_report), encoding="utf-8")
    market = workspace.free_agent_market("B")
    outgoing = next(
        player
        for player in market["outgoing_players"]
        if player["role"] == market["candidates"][0]["role"]
    )
    signing = FreeAgentPlan(
        market["market_id"],
        "a-01",
        outgoing["player_id"],
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="B",
            manager_objective="top_half",
            manager_free_agent=signing,
        ),
        replace=True,
    )

    effective = load_effective_roster(tmp_path, "B")
    signed = next(
        player for player in effective["players"] if player["player_id"] == "a-01"
    )
    assert signed["team_id"] == "B"
    assert signed["free_agent_origin_team"] == "A"
    assert signed["source"] == "gfs_global_free_agent_v1"
    assert outgoing["player_id"] not in {
        player["player_id"] for player in effective["players"]
    }
    persisted = workspace._session()
    transition = persisted["player_market"]["transitions"][-1]
    assert transition["summary"]["signings"] == 1
    assert transition["entries"][0]["player_id"] == outgoing["player_id"]
    assert transition["entries"][0]["available_from_season_id"] == "season-0004"

    tampered = json.loads(json.dumps(persisted))
    tampered["season"]["plan"]["manager_free_agent"]["outgoing_player_id"] = "b-02"
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="global market season plan mismatch"):
        workspace.status()
    workspace.session_path.write_text(json.dumps(persisted), encoding="utf-8")

    complete_active("B")
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="B",
            manager_objective="top_half",
        ),
        replace=True,
    )
    outcomes = workspace.status()["scouting_outcomes"]
    assert outcomes["outcome_count"] == 1
    assert outcomes["manager_outcome_count"] == 1
    resolved = outcomes["manager_outcomes"][0]
    assert resolved["season_id"] == "season-0003"
    assert resolved["player_id"] == "a-01"
    assert resolved["observation_source"] == "manager_scouting_report"
    assert resolved["realized"]["interval_contained_truth"] is True
    assert resolved["realized"]["evidence_coverage"] == "unavailable"
    assert resolved["realized"]["usage_band"] == "evidence_unavailable"

    persisted_outcome = workspace._session()
    tampered_outcome = json.loads(json.dumps(persisted_outcome))
    tampered_outcome["scouting_outcomes"]["outcomes"][0]["realized"]["team_points"] += 1
    workspace.session_path.write_text(
        json.dumps(tampered_outcome),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scouting outcome source replay mismatch"):
        workspace.status()


def test_sporting_director_plan_freezes_one_cross_module_next_season_policy(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Sporting director", mode="research", seed=41),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
        )
    )
    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": {"home": 1, "away": 1},
            }
        )
        if "A" in {fixture["home"], fixture["away"]}:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    brief = workspace.sporting_plan("A")
    workflow = workspace.status()["workflow"]
    assert workflow["state"] == "season_planning"
    assert workflow["next_action"]["id"] == "review_sporting_plan"
    assert workflow["progress"]["sporting_plan_required"] is True
    assert brief["source_season_id"] == "season-0001"
    assert brief["target_season_id"] == "season-0002"
    assert brief["source_roster_identity"]
    assert len(brief["recommendation"]["priority_roles"]) <= 3
    candidate = brief["recruitment_candidates"][0]
    outside_role = next(
        row["role"]
        for row in brief["role_diagnostics"]
        if row["role"] != candidate["role"]
    )
    outgoing = next(row for row in brief["outgoing_players"] if row["role"] != "GK")
    conflicting_directive = SportingDirective(
        brief["planning_id"],
        philosophy="balanced",
        risk_level="high",
        priority_roles=(outside_role,),
    )
    conflicting_recruitment = RecruitmentPlan(
        brief["recruitment_market_id"],
        (RecruitmentMove(candidate["player_id"], outgoing["player_id"]),),
    )
    before_conflict = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match="outside sporting priority roles"):
        workspace.create_season(
            SeasonPlan(
                teams=("A", "B", "C", "D"),
                manager_team="A",
                manager_objective="top_half",
                manager_sporting_directive=conflicting_directive,
                manager_recruitment=conflicting_recruitment,
            ),
            replace=True,
        )
    assert workspace.session_path.read_bytes() == before_conflict

    directive = SportingDirective(
        brief["planning_id"],
        philosophy="balanced",
        risk_level="balanced",
        priority_roles=tuple(brief["recommendation"]["priority_roles"]),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
            manager_sporting_directive=directive,
        ),
        replace=True,
    )
    frozen = workspace.status()["season"]
    assert frozen["sporting_brief"] == brief
    assert frozen["sporting_evaluation"]["directive"] == directive.as_dict()
    assert frozen["sporting_evaluation"]["selected_recruitment_moves"] == 0

    persisted = workspace._session()
    tampered = json.loads(json.dumps(persisted))
    tampered_brief = tampered["season"]["sporting_brief"]
    tampered_brief["club_balance"] += 1
    tampered_brief.pop("planning_id")
    rewritten_id = evidence_identity(tampered_brief)
    tampered_brief["planning_id"] = rewritten_id
    tampered["season"]["plan"]["manager_sporting_directive"]["planning_id"] = (
        rewritten_id
    )
    tampered["season"]["sporting_evaluation"]["planning_id"] = rewritten_id
    tampered["season"]["sporting_evaluation"]["directive"]["planning_id"] = rewritten_id
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="sporting brief source replay mismatch"):
        workspace.status()


def test_completed_sporting_plan_produces_replayable_cross_season_review(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Strategy review", mode="research", seed=43),
    )

    def complete_current_season():
        session = workspace._session()
        for fixture in session["season"]["fixtures"]:
            fixture.update(
                {
                    "state": "completed",
                    "match_id": fixture["fixture_id"],
                    "report": f"outputs/{fixture['fixture_id']}.json",
                    "score": {"home": 1, "away": 1},
                }
            )
            if "A" in {fixture["home"], fixture["away"]}:
                fixture["manager_decision"] = {
                    "team": "A",
                    "tactic": "balanced",
                    "rotation": "balanced",
                }
        session["season"]["state"] = "complete"
        workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
        )
    )
    complete_current_season()
    brief = workspace.sporting_plan("A")
    directive = SportingDirective(
        brief["planning_id"],
        philosophy="balanced",
        risk_level="balanced",
        priority_roles=tuple(brief["recommendation"]["priority_roles"]),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
            manager_sporting_directive=directive,
        ),
        replace=True,
    )
    complete_current_season()
    next_brief = workspace.sporting_plan("A")
    feedback = next_brief["previous_strategy_review"]
    assert feedback["season_id"] == "season-0002"
    assert feedback["priority_unaddressed_roles"] == sorted(directive.priority_roles)
    assert all(
        row["continuity_signal"]
        for row in next_brief["role_diagnostics"]
        if row["role"] in directive.priority_roles
    )
    next_directive = SportingDirective(
        next_brief["planning_id"],
        philosophy="balanced",
        risk_level="balanced",
        priority_roles=tuple(next_brief["recommendation"]["priority_roles"]),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
            manager_sporting_directive=next_directive,
        ),
        replace=True,
    )

    status = workspace.status()
    reviews = status["sporting_reviews"]
    assert reviews["review_count"] == 1
    assert reviews["latest_review"]["season_id"] == "season-0002"
    assert reviews["latest_review"]["evidence_coverage"] == "unavailable"
    assert "post_squad" not in reviews["latest_review"]
    assert reviews["latest_review"]["directive"] == directive.as_dict()
    assert status["season"]["sporting_brief"] == next_brief

    tampered = workspace._session()
    review = tampered["sporting_reviews"]["reviews"][0]
    priority = set(review["directive"]["priority_roles"])
    row = next(
        item
        for item in review["post_squad"]
        if item["role"] not in priority
        and all(
            abs(item["quality"] - threshold) > 0.005 for threshold in (0.62, 0.70, 0.78)
        )
    )
    row["quality"] = round(row["quality"] + 0.001, 6)
    review.pop("review_id")
    review["review_id"] = evidence_identity(review)
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="sporting (brief|review) source replay mismatch",
    ):
        workspace.status()


def test_club_situation_is_an_atomic_manager_choice_with_source_replay(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Club timeline", mode="research", seed=47),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
        )
    )
    session = workspace._session()
    managed = [
        row
        for row in session["season"]["fixtures"]
        if "A" in {row["home"], row["away"]}
    ]
    for fixture in managed[:2]:
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": {"home": 1, "away": 0},
                "manager_decision": ManagerDecision(
                    "A",
                    tactic="balanced",
                    rotation="strongest",
                ).as_dict(),
            }
        )
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    season = workspace.season_status()
    situation = season["club_timeline_view"]["current_situation"]
    assert situation["kind"] == "squad_load"
    protect = ClubEventChoice(
        situation["event_id"],
        situation["event_identity"],
        "protect_squad",
    )
    before = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match="conflicts with the club situation"):
        workspace.set_manager_decision(
            ManagerDecision(
                "A",
                tactic="balanced",
                rotation="strongest",
                club_event_choice=protect,
            ),
            fixture_id=managed[2]["fixture_id"],
        )
    assert workspace.session_path.read_bytes() == before

    push = ClubEventChoice(
        situation["event_id"],
        situation["event_identity"],
        "push_starters",
    )
    updated = workspace.set_manager_decision(
        ManagerDecision(
            "A",
            tactic="balanced",
            rotation="strongest",
            club_event_choice=push,
        ),
        fixture_id=managed[2]["fixture_id"],
    )
    assert updated["club_timeline_view"]["resolved_count"] == 1
    assert updated["club_timeline"]["events"][0]["control"] == "manager"

    tampered = workspace._session()
    resolution = tampered["season"]["club_timeline"]["events"][0]
    event = resolution["event"]
    event["trigger"]["strongest_rotations"] = 1
    event.pop("event_identity")
    event["event_identity"] = hashlib.sha256(
        json.dumps(
            event,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolution["choice"]["event_identity"] = event["event_identity"]
    target = next(
        row
        for row in tampered["season"]["fixtures"]
        if row["fixture_id"] == managed[2]["fixture_id"]
    )
    target["manager_decision"]["club_event_choice"]["event_identity"] = event[
        "event_identity"
    ]
    resolution.pop("resolution_id")
    resolution["resolution_id"] = hashlib.sha256(
        json.dumps(
            resolution,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="club situation source replay mismatch"):
        workspace.status()


def test_workspace_finance_blocks_unaffordable_recruitment_atomically(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
    roster_path = tmp_path / "data/rosters/A.json"
    expensive = json.loads(roster_path.read_text(encoding="utf-8"))
    for player in expensive["players"]:
        player["abilities"] = {key: 0.82 for key in player["abilities"]}
    roster_path.write_text(json.dumps(expensive), encoding="utf-8")
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Finance gate", mode="research", seed=37),
    )
    active = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    assert workspace.status()["club_finance"]["balance"] == 8
    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        manager_home = fixture["home"] == "A"
        manager_away = fixture["away"] == "A"
        score = {"home": 0, "away": 0}
        if manager_home:
            score = {"home": 0, "away": 2}
        elif manager_away:
            score = {"home": 2, "away": 0}
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": score,
            }
        )
        if manager_home or manager_away:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    status = workspace.status()
    projection = status["club_finance"]["projected_after_current_season"]
    assert projection["settlement"]["delta"] == -5
    assert projection["balance_after"] == 3
    market = workspace.recruitment_market("A")
    assert market["club_balance"] == 3
    assert market["available_budget"] == 3
    candidate = market["candidates"][4]
    assert candidate["recruitment_cost"] == 6
    outgoing = next(
        player for player in market["outgoing_players"] if player["role"] != "GK"
    )
    recruitment = RecruitmentPlan(
        market["market_id"],
        (RecruitmentMove(candidate["player_id"], outgoing["player_id"]),),
    )
    before = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match=r"available club funds \(3\)"):
        workspace.create_season(
            SeasonPlan(
                teams=("A", "B", "C", "D"),
                manager_team="A",
                manager_objective="top_half",
                manager_recruitment=recruitment,
            ),
            replace=True,
        )
    assert workspace.session_path.read_bytes() == before
    assert workspace.season_status()["season_id"] == active["season_id"]


def test_workspace_replays_ai_club_evolution_into_next_season_rosters(tmp_path):
    _evidence(tmp_path)
    for team in ("A", "B", "C", "D"):
        _manager_roster(tmp_path, team)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Living league", mode="research", seed=43),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="champion",
        )
    )
    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        manager_home = fixture["home"] == "A"
        manager_away = fixture["away"] == "A"
        score = {"home": 1, "away": 1}
        if manager_home:
            score = {"home": 2, "away": 0}
        elif manager_away:
            score = {"home": 0, "away": 2}
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": score,
            }
        )
        if manager_home or manager_away:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    rival_market = workspace.recruitment_market("B")
    assert rival_market["settlement_preview"]["control"] == "ai_club"
    assert rival_market["club_balance"] >= rival_market["available_budget"]

    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
        ),
        replace=True,
    )
    persisted = workspace._session()
    transition = persisted["league_ecosystem"]["transitions"][-1]
    strategy_snapshot = persisted["season"]["club_strategies"]
    assert transition["from_season_id"] == "season-0001"
    assert transition["to_season_id"] == "season-0002"
    assert transition["manager_controlled_team"] == "A"
    assert strategy_snapshot["source_season_id"] == "season-0001"
    assert strategy_snapshot["ecosystem_transition_identity"] == (
        strategy_identity(transition)
    )
    strategy_profiles = {
        profile["team"]: profile for profile in strategy_snapshot["profiles"]
    }
    assert strategy_profiles["B"]["recruitment_move_count"] >= 1
    assert "post_window_roster:" in " ".join(strategy_profiles["B"]["reasons"])
    assert [club["team"] for club in transition["ai_clubs"]] == ["B", "C", "D"]
    assert all(club["decision"]["available"] for club in transition["ai_clubs"])
    assert all(club["decision"]["plan"] for club in transition["ai_clubs"])
    assert set(persisted["finance_registry"]["clubs"]) == {"A", "B", "C", "D"}
    assert all(
        len(persisted["squad_registry"]["clubs"][team]["transactions"]) == 1
        for team in ("B", "C", "D")
    )
    b_transaction = persisted["squad_registry"]["clubs"]["B"]["transactions"][0]
    effective_b = load_effective_roster(tmp_path, "B")
    assert effective_b is not None
    assert b_transaction["moves"][0]["incoming"]["player_id"] in {
        player["player_id"] for player in effective_b["players"]
    }
    status = workspace.status()
    assert status["league_ecosystem"]["transition_count"] == 1
    assert status["league_ecosystem"]["latest_transition"] == transition

    tampered_strategy = workspace._session()
    tampered_strategy["season"]["club_strategies"]["profiles"][0]["style_scores"][
        "balanced"
    ] += 0.01
    workspace.session_path.write_text(
        json.dumps(tampered_strategy),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="club strategy source replay mismatch"):
        workspace.status()
    workspace.session_path.write_text(json.dumps(persisted), encoding="utf-8")

    tampered = workspace._session()
    tampered["league_ecosystem"]["transitions"][0]["ai_clubs"][0]["decision"][
        "minimum_quality_improvement"
    ] = 0.5
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source replay mismatch"):
        workspace.status()


def test_workspace_enforces_board_contract_atomically_before_next_season(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Board contract", mode="research", seed=41),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="top_half",
        )
    )
    session = workspace._session()
    for fixture in session["season"]["fixtures"]:
        manager_home = fixture["home"] == "A"
        manager_away = fixture["away"] == "A"
        score = {"home": 0, "away": 0}
        if manager_home:
            score = {"home": 0, "away": 2}
        elif manager_away:
            score = {"home": 2, "away": 0}
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": f"outputs/{fixture['fixture_id']}.json",
                "score": score,
            }
        )
        if manager_home or manager_away:
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
            }
    session["season"]["state"] = "complete"
    workspace.session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 3"):
        workspace.create_season(
            SeasonPlan(
                teams=("A", "B", "C", "D"),
                manager_team="A",
                manager_objective="points_target",
                manager_points_target=2,
            ),
            replace=True,
        )
    unchanged = workspace._session()
    assert unchanged["season"]["season_id"] == "season-0001"
    assert unchanged["season_history"] == []
    assert unchanged["archived_seasons_total"] == 0

    created = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
            manager_objective="points_target",
            manager_points_target=3,
        ),
        replace=True,
    )
    status = workspace.status()
    assert created["season_id"] == "season-0002"
    assert status["manager_career"]["employment_status"] == "under_review"
    assert status["manager_career"]["board_confidence"] == 42
    assert status["manager_career"]["appointment_state"] == "continuing"
    assert status["season_history"][0]["board_review"]["grade"] == "negative"

    tampered = workspace._session()
    tampered["season_history"][0]["manager_profile"]["commitments"]["entries"][1][
        "status"
    ] = "missed"
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment source replay mismatch"):
        workspace.status()


def test_named_player_promises_freeze_before_decisions_and_replay_roster_source(
    tmp_path,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path, "A")
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Player promises", mode="research", seed=47),
    )
    created = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
        )
    )
    original = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match="age 23 or younger"):
        workspace.set_player_role_promises(
            PlayerPromisePlan((PlayerRolePromise("a-15", "development"),))
        )
    assert workspace.session_path.read_bytes() == original

    promised = workspace.set_player_role_promises(
        PlayerPromisePlan(
            (
                PlayerRolePromise("a-00", "core"),
                PlayerRolePromise("a-01", "development"),
            )
        )
    )
    contract = promised["player_role_promises"]
    assert contract["control"] == "manager"
    assert [
        (row["player_id"], row["promised_role"]) for row in contract["players"]
    ] == [
        ("a-00", "core"),
        ("a-01", "development"),
    ]
    frozen = workspace.session_path.read_bytes()
    with pytest.raises(ValueError, match="already frozen"):
        workspace.set_player_role_promises(
            PlayerPromisePlan((PlayerRolePromise("a-02", "rotation"),))
        )
    assert workspace.session_path.read_bytes() == frozen

    workspace.set_manager_decision(
        ManagerDecision(team="A", tactic="balanced", rotation="balanced"),
        fixture_id=created["next_manager_fixture"]["fixture_id"],
    )
    status = workspace.season_status()
    assert status["player_promise_view"]["control"] == "manager"
    assert status["player_promise_view"]["entries"][0]["status"] == "pending"

    tampered = workspace._session()
    contract = tampered["season"]["player_role_promises"]
    contract["players"][0]["name"] = "Invented Name"
    identity_source = dict(contract)
    identity_source.pop("contract_id")
    contract["contract_id"] = hashlib.sha256(
        json.dumps(
            identity_source,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    workspace.session_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="contract source replay mismatch"):
        workspace.status()


def test_player_promise_archive_joins_lineup_accountability_to_exact_minutes(
    tmp_path,
):
    _evidence(tmp_path)
    _manager_roster(tmp_path, "A")
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Promise outcomes", mode="research", seed=53),
    )
    created = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_team="A",
        )
    )
    workspace.set_player_role_promises(
        PlayerPromisePlan(
            (
                PlayerRolePromise("a-00", "core"),
                PlayerRolePromise("a-01", "development"),
            )
        )
    )
    session = workspace._session()
    season = session["season"]
    lineup = created["manager_squad"]["automatic"]["balanced"]
    for fixture in season["fixtures"]:
        report = tmp_path / f"outputs/{fixture['fixture_id']}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        player_stats = {fixture["home"]: {}, fixture["away"]: {}}
        if "A" in {fixture["home"], fixture["away"]}:
            player_stats["A"] = {
                "a-00": {"minutes": 90.0},
                "a-01": {"minutes": 90.0},
            }
            fixture["manager_decision"] = {
                "team": "A",
                "tactic": "balanced",
                "rotation": "balanced",
                "lineup": lineup,
            }
            fixture["player_promise_availability"] = {
                "a-00": True,
                "a-01": True,
            }
        report.write_text(
            json.dumps(
                {
                    "match_id": fixture["fixture_id"],
                    "match_plan": {
                        "competition": {
                            "season_id": season["season_id"],
                            "fixture_id": fixture["fixture_id"],
                        }
                    },
                    "raw_summary": {"player_stats": player_stats},
                }
            ),
            encoding="utf-8",
        )
        fixture.update(
            {
                "state": "completed",
                "match_id": fixture["fixture_id"],
                "report": report.relative_to(tmp_path).as_posix(),
                "score": {"home": 1, "away": 0},
            }
        )
    season["state"] = "complete"

    archived = workspace._archive_with_development_evidence(season)

    outcome = archived["player_promise_outcomes"]
    assert outcome["coverage"] == "complete"
    assert [row["observed_status"] for row in outcome["players"]] == [
        "fulfilled",
        "fulfilled",
    ]
    assert [row["selection_status"] for row in outcome["players"]] == [
        "fulfilled",
        "fulfilled",
    ]
    replay_session = workspace._session()
    replay_session["season_history"] = [archived]
    replay_session["season"] = None
    assert (
        workspace._season_history_view(replay_session)[0]["player_promise_outcomes"]
        == outcome
    )

    tampered = json.loads(json.dumps(replay_session))
    tampered["season_history"][0]["player_promise_outcomes"]["players"][0][
        "observed_status"
    ] = "missed"
    with pytest.raises(ValueError, match="outcome replay mismatch"):
        workspace._season_history_view(tampered)


def test_workspace_requires_next_manager_decision_before_any_fixture_starts(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Decision gate", mode="research", seed=17),
    )
    status = workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            fast=True,
            manager_team="A",
        )
    )
    from src import app

    calls = []
    monkeypatch.setattr(
        app,
        "run_micro_match",
        lambda *args, **kwargs: calls.append(kwargs) or _Summary(),
    )
    with pytest.raises(ValueError, match="manager decision is required"):
        workspace.play_next_matchday()
    assert calls == []
    assert all(
        fixture.get("attempts", 0) == 0
        for fixture in workspace.season_status()["fixtures"]
    )

    next_fixture = status["next_manager_fixture"]
    later_fixture = next(
        fixture
        for fixture in status["fixtures"]
        if fixture["fixture_id"] != next_fixture["fixture_id"]
        and "A" in {fixture["home"], fixture["away"]}
    )
    with pytest.raises(ValueError, match="only the next"):
        workspace.set_manager_decision(
            ManagerDecision(team="A", tactic="balanced", rotation="rotate"),
            fixture_id=later_fixture["fixture_id"],
        )


def test_workspace_rejects_stale_manager_decision_revision(tmp_path):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Revision gate", mode="research", seed=19),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            fast=True,
            manager_team="A",
        )
    )
    decided = workspace.set_manager_decision(
        ManagerDecision(team="A", tactic="counter_attack", rotation="balanced"),
    )
    assert decided["revision"] == 1
    with pytest.raises(ValueError, match="decision revision conflict"):
        workspace.play_next_matchday(expected_revision=0)


def test_workspace_season_retries_failed_fixture_with_same_settlement_identity(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Retry Season", mode="research", seed=3),
    )
    workspace.create_season(
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            fast=True,
            manager_team="A",
        )
    )
    workspace.set_manager_decision(ManagerDecision(team="A"))
    from src import app

    ids = []

    def fail_once(*args, **kwargs):
        ids.append(kwargs["continuity_id"])
        if len(ids) == 1:
            raise RuntimeError("season fixture crash")
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", fail_once)
    with pytest.raises(RuntimeError, match="fixture crash"):
        workspace.play_next_matchday()
    failed_fixture = workspace.season_status()["fixtures"][0]
    assert failed_fixture["state"] == "failed"
    before_identity = failed_fixture["world_state_before"][failed_fixture["home"]][
        "source_identity"
    ]
    assert failed_fixture.get("world_state_transition") is None
    status = workspace.play_next_matchday()
    assert status["progress"]["completed"] == 2
    assert ids[0] == ids[1] == "season-0001:md01-fx01"
    assert status["fixtures"][0]["attempts"] == 2
    retried_transition = status["fixtures"][0]["world_state_transition"]
    assert (
        retried_transition["phases"]["before_match"][status["fixtures"][0]["home"]][
            "source_identity"
        ]
        == before_identity
    )
    assert retried_transition["recovery_complete"] is True


def test_workspace_season_adopts_existing_completed_report_without_rerun(
    tmp_path,
    monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path,
        StudioConfig(name="Recover Season", mode="research", seed=5),
    )
    workspace.create_season(SeasonPlan(teams=("A", "B", "C", "D"), fast=True))
    from src import app

    calls = []
    monkeypatch.setattr(
        app,
        "run_micro_match",
        lambda *args, **kwargs: calls.append(kwargs) or _Summary(),
    )
    session = workspace._session()
    fixture = session["season"]["fixtures"][0]
    competition = {
        "season_id": "season-0001",
        "fixture_id": fixture["fixture_id"],
        "matchday": fixture["matchday"],
    }
    club_strategy = resolve_fixture_club_strategy(
        session["season"]["club_strategies"],
        home=fixture["home"],
        away=fixture["away"],
    )
    workspace._run_match_locked(
        fixture["home"],
        fixture["away"],
        fast=True,
        plan=MatchPlan(
            experience="season_manager",
            home_tactic=club_strategy["home_tactic"],
            away_tactic=club_strategy["away_tactic"],
        ),
        seed_override=106,
        continuity=True,
        continuity_id="season-0001:md01-fx01",
        competition=competition,
        club_strategy=club_strategy,
    )
    assert len(calls) == 1
    status = workspace.play_next_matchday()
    assert status["progress"]["completed"] == 2
    assert len(calls) == 2
    assert status["fixtures"][0]["match_id"].startswith("0001-")
