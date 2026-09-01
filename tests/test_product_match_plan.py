from types import MethodType, SimpleNamespace

import pytest

from src.match_engine.tactical_profile import apply_locked_tactical_preset
from src.product.match_plan import (
    MatchPlan, PairedMatchPlan, WorldModelForkPlan, playable_tactic_catalog,
)
from tests.test_affective_phase1b import _FakeAgent


def test_observational_plan_preserves_team_identity_and_macro_replay():
    plan = MatchPlan.from_payload({})
    assert plan.experience == "observational"
    assert plan.score_path == "macro_replay"
    assert plan.as_dict()["claim_boundary"].startswith("single_run_descriptive_only")
    with pytest.raises(ValueError, match="only available in tactical_lab"):
        MatchPlan(reuse_last_seed=True)


def test_tactical_lab_is_mode_gated_and_requires_real_intervention():
    with pytest.raises(ValueError, match="requires at least one"):
        MatchPlan(experience="tactical_lab")
    plan = MatchPlan(experience="tactical_lab", home_tactic="gegenpress")
    with pytest.raises(ValueError, match="requires research"):
        plan.validate_for_mode("stable")
    plan.validate_for_mode("research")
    assert plan.score_path == "physics_official"


def test_season_manager_plan_is_confined_to_season_gameplay():
    plan = MatchPlan(
        experience="season_manager", home_tactic="gegenpress",
        away_tactic="team_identity",
    )
    plan.validate_for_mode("research", context="season")
    assert plan.score_path == "physics_official"
    assert "gameplay" in plan.as_dict()["claim_boundary"]
    with pytest.raises(ValueError, match="only available inside a season"):
        plan.validate_for_mode("research")


@pytest.mark.parametrize("field", ["home_tactic", "away_tactic"])
def test_unknown_tactic_is_rejected_instead_of_silently_becoming_balanced(field):
    with pytest.raises(ValueError, match=f"unsupported {field.split('_')[0]} tactic"):
        MatchPlan.from_payload({"experience": "tactical_lab", field: "imaginary"})


def test_catalog_has_stable_ids_and_user_facing_explanations():
    catalog = playable_tactic_catalog()
    assert catalog[0]["id"] == "team_identity"
    assert {row["id"] for row in catalog} >= {
        "balanced", "tiki_taka", "gegenpress", "counter_attack",
        "low_block_counter", "direct_vertical",
    }
    assert all(row["label"] and row["description"] for row in catalog)


def test_paired_plan_freezes_seed_single_side_and_round_trips():
    plan = PairedMatchPlan(
        baseline_home_tactic="balanced",
        baseline_away_tactic="low_block_counter",
        treatment_home_tactic="gegenpress",
        treatment_away_tactic="low_block_counter",
        seed=77,
    )
    assert plan.focus_side == "home"
    assert not plan.baseline_plan().reuse_last_seed
    assert plan.treatment_plan().reuse_last_seed
    assert plan.baseline_plan().score_path == "physics_official"
    assert PairedMatchPlan.from_payload(plan.as_dict()).as_dict() == plan.as_dict()
    plan.validate_for_mode("research")
    with pytest.raises(ValueError, match="requires research"):
        plan.validate_for_mode("stable")


@pytest.mark.parametrize("values", [
    {
        "baseline_home_tactic": "balanced",
        "baseline_away_tactic": "low_block_counter",
        "treatment_home_tactic": "balanced",
        "treatment_away_tactic": "low_block_counter",
        "seed": 7,
    },
    {
        "baseline_home_tactic": "balanced",
        "baseline_away_tactic": "low_block_counter",
        "treatment_home_tactic": "gegenpress",
        "treatment_away_tactic": "direct_vertical",
        "seed": 7,
    },
    {
        "baseline_home_tactic": "balanced",
        "baseline_away_tactic": "low_block_counter",
        "treatment_home_tactic": "gegenpress",
        "treatment_away_tactic": "low_block_counter",
        "seed": True,
    },
])
def test_paired_plan_rejects_no_change_joint_change_or_invalid_seed(values):
    with pytest.raises(ValueError):
        PairedMatchPlan(**values)


def test_world_model_fork_is_a_fixed_tactic_predict_only_policy_pair():
    plan = WorldModelForkPlan(
        home_tactic="balanced", away_tactic="low_block_counter", seed=91,
    )
    baseline, treatment = plan.baseline_plan(), plan.treatment_plan()
    assert baseline.experience == treatment.experience == "world_model_lab"
    assert baseline.world_model_policy == "predict_only"
    assert treatment.world_model_policy == "action_policy"
    assert not baseline.reuse_last_seed and treatment.reuse_last_seed
    assert baseline.home_tactic == treatment.home_tactic == "balanced"
    assert baseline.away_tactic == treatment.away_tactic == "low_block_counter"
    assert baseline.world_model_branch_at_sec == 2700.0
    assert treatment.world_model_branch_at_sec == 2700.0
    assert WorldModelForkPlan.from_payload(plan.as_dict()) == plan
    plan.validate_for_mode("research")
    with pytest.raises(ValueError, match="deterministic research mode"):
        plan.validate_for_mode("cognitive")
    with pytest.raises(ValueError, match="confined to world_model_lab"):
        MatchPlan(world_model_policy="action_policy")


@pytest.mark.parametrize("branch", [-1, 5401, float("nan"), True])
def test_world_model_fork_rejects_invalid_branch_time(branch):
    with pytest.raises(ValueError, match="branch time"):
        WorldModelForkPlan(
            home_tactic="balanced", away_tactic="low_block_counter",
            seed=91, branch_at_sec=branch,
        )


def test_user_preset_changes_full_vector_and_records_non_llm_source():
    agent = _FakeAgent("Brazil")
    agent.coach_profile = SimpleNamespace(preferred_preset="balanced")
    agent.semantic_memory = {}
    vector = apply_locked_tactical_preset(
        agent, "gegenpress", source="studio_user_intervention",
    )
    assert agent.coach_profile.preferred_preset == "gegenpress"
    assert agent.style_archetype == "gegenpress"
    assert agent.tactical_vector == vector
    assert vector["pressing_intensity"] > vector["low_block"]
    assert len(vector) >= 20
    assert agent.semantic_memory["tactical_intervention"] == {
        "preset": "gegenpress", "source": "studio_user_intervention",
    }


def test_public_micro_match_applies_user_tactics_before_engine_execution(
    monkeypatch, tmp_path,
):
    from src import app

    home = _FakeAgent("Brazil")
    away = _FakeAgent("Argentina")
    for agent in (home, away):
        agent.coach_profile = SimpleNamespace(preferred_preset="balanced")
        agent.semantic_memory = {}
        agent.simulate_internal_game = lambda _pressure: {}
    world = SimpleNamespace(agents={"Brazil": home, "Argentina": away})
    monkeypatch.setattr(app, "build_simulation", lambda _root: (world, None, None))
    monkeypatch.setattr(
        "src.memory_engine.poisson_simulator.simulate_match_score",
        lambda *args, **kwargs: (0, 0, 0.8, 0.7),
    )
    observed = {}

    def fake_engine(home_agent, away_agent, **kwargs):
        observed["home"] = home_agent
        observed["away"] = away_agent
        observed["kwargs"] = kwargs
        return "summary"

    monkeypatch.setattr(
        "src.match_engine.match_micro_runner.run_match_micro_simulation",
        fake_engine,
    )
    result = app.run_micro_match(
        "Brazil", "Argentina", base_dir=tmp_path,
        home_tactic="gegenpress", away_tactic="low_block_counter",
        stage_name="studio-match-0001",
    )
    assert result == "summary"
    assert observed["home"].style_archetype == "gegenpress"
    assert observed["away"].style_archetype == "low_block_counter"
    assert observed["home"].tactical_vector["pressing_intensity"] > (
        observed["away"].tactical_vector["pressing_intensity"]
    )
    assert observed["home"].semantic_memory["tactical_intervention"][
        "source"
    ] == "studio_user_intervention"
    assert observed["kwargs"]["base_dir"] == tmp_path
    assert observed["kwargs"]["stage_name"] == "studio-match-0001"


def test_public_micro_match_continuity_is_explicit_and_reports_transition(
    monkeypatch, tmp_path,
):
    from src import app

    home = _FakeAgent("Brazil")
    away = _FakeAgent("Argentina")
    for agent in (home, away):
        agent.semantic_memory = {}
        agent.simulate_internal_game = lambda _pressure: {}
    world = SimpleNamespace(agents={"Brazil": home, "Argentina": away})
    monkeypatch.setattr(app, "build_simulation", lambda _root: (world, None, None))
    monkeypatch.setattr(
        "src.memory_engine.poisson_simulator.simulate_match_score",
        lambda *args, **kwargs: (1, 0, 1.2, 0.6),
    )
    summary = SimpleNamespace(
        goals_micro_home=2,
        goals_micro_away=1,
        micro_xg_home=1.7,
        micro_xg_away=0.9,
    )
    monkeypatch.setattr(
        "src.match_engine.match_micro_runner.run_match_micro_simulation",
        lambda *args, **kwargs: summary,
    )
    calls = {"prepare": 0, "finalize": []}

    def fake_prepare(*args, **kwargs):
        calls["prepare"] += 1
        return {"players": []}, None

    monkeypatch.setattr("src.simulation.match_pipeline.prepare_match_agents", fake_prepare)
    monkeypatch.setattr(
        "src.simulation.cross_match_state.carryover_snapshot",
        lambda agent: {"team_id": agent.team_name, "revision": calls["prepare"]},
    )
    monkeypatch.setattr(
        "src.simulation.match_pipeline.finalize_match_feedback",
        lambda *args, **kwargs: calls["finalize"].append(kwargs),
    )

    result = app.run_micro_match(
        "Brazil", "Argentina", base_dir=tmp_path,
        stage_name="season-1-md-1", continuity=True,
    )
    assert result is summary
    assert calls["prepare"] == 1
    assert calls["finalize"][0]["result_home"] == "win"
    assert calls["finalize"][0]["score_diff_home"] == 1
    assert calls["finalize"][0]["xg_home"] == 1.7
    assert summary.continuity_state["enabled"] is True
    assert summary.continuity_state["rosters_loaded"] == {"home": True, "away": False}


def test_manager_rotation_and_continuity_status_reach_both_score_paths(
    monkeypatch, tmp_path,
):
    from src import app

    home = _FakeAgent("Brazil")
    away = _FakeAgent("Argentina")
    for agent in (home, away):
        agent.semantic_memory = {}
        agent.simulate_internal_game = lambda _pressure: {}
    home.get_effective_status = lambda: 60.0
    away.get_effective_status = lambda: 70.0
    from src.simulation.agent import SocietyAgent

    home.tactical_effects = MethodType(SocietyAgent.tactical_effects, home)
    away.tactical_effects = MethodType(SocietyAgent.tactical_effects, away)
    world = SimpleNamespace(agents={"Brazil": home, "Argentina": away})
    monkeypatch.setattr(app, "build_simulation", lambda _root: (world, None, None))
    macro = {}

    def fake_score(home_status, away_status, **kwargs):
        macro.update(home=home_status, away=away_status)
        return 1, 0, 1.2, 0.6

    monkeypatch.setattr(
        "src.memory_engine.poisson_simulator.simulate_match_score", fake_score,
    )
    micro = {}
    summary = SimpleNamespace(
        goals_micro_home=1, goals_micro_away=0,
        micro_xg_home=1.2, micro_xg_away=0.6,
    )

    def fake_engine(*args, **kwargs):
        micro.update(kwargs)
        return summary

    monkeypatch.setattr(
        "src.match_engine.match_micro_runner.run_match_micro_simulation", fake_engine,
    )
    frozen_lineup = {
        "starters": [f"p{index:02d}" for index in range(11)],
        "bench": ["p11", "p12"],
        "source": "manual",
        "roster_fingerprint": "a" * 64,
    }
    prepared = {}

    def fake_lineup_prepare(*args, **kwargs):
        prepared.update(kwargs)
        roster = {
            "lineup_selection": frozen_lineup,
            "players": [
                {"player_id": f"p{index:02d}", "name": f"Player {index}"}
                for index in range(13)
            ],
        }
        return roster, None

    monkeypatch.setattr(
        "src.simulation.match_pipeline.prepare_match_agents",
        fake_lineup_prepare,
    )
    monkeypatch.setattr(
        "src.simulation.cross_match_state.carryover_snapshot",
        lambda agent: {"team_id": agent.team_name},
    )
    settled = {}
    monkeypatch.setattr(
        "src.simulation.match_pipeline.finalize_match_feedback",
        lambda *args, **kwargs: settled.update(kwargs),
    )

    result = app.run_micro_match(
        "Brazil", "Argentina", base_dir=tmp_path, continuity=True,
        continuity_id="season-1:fixture-1", home_rotation="rotate",
        away_rotation="strongest", home_lineup=frozen_lineup,
        home_fatigue_load_factor=0.88,
    )
    assert result is summary
    assert macro["home"] == pytest.approx(57.212)
    assert macro["away"] == pytest.approx(70.212)
    assert micro["eff_status_home"] == pytest.approx(57.212)
    assert micro["eff_status_away"] == pytest.approx(70.212)
    assert home.tactical_controls["rotation_aggressiveness"] == 1.0
    assert settled["fatigue_load_home"] == pytest.approx(1.048 * 0.88)
    assert settled["fatigue_load_away"] == pytest.approx(1.498)
    assert settled["fatigue_load_home"] < settled["fatigue_load_away"]
    assert summary.manager_effects["home"]["rotation_status_penalty"] == 3.0
    assert summary.manager_effects["home"]["club_fatigue_load_factor"] == 0.88
    assert prepared["home_lineup"] == frozen_lineup
    assert summary.manager_effects["home"]["lineup"]["starter_names"][0] == "Player 0"


def test_public_micro_match_rejects_lineup_without_continuity():
    from src import app

    with pytest.raises(ValueError, match="require continuity"):
        app.run_micro_match(
            home_lineup={
                "starters": [f"p{index}" for index in range(11)],
                "bench": [], "source": "manual",
                "roster_fingerprint": "a" * 64,
            },
        )
    with pytest.raises(ValueError, match="support requires continuity"):
        app.run_micro_match(home_fatigue_load_factor=0.88)
