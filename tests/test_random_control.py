import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.memory_engine.poisson_simulator import simulate_match_score, simulate_penalty_shootout
from src.memory_engine.tournament_simulator import simulate_journey
from src.simulation.agent import SocietyAgent
from src.simulation.engine import SocialMediaFeed, WorldEngine
from src.simulation.random_control import derive_seed, named_py_rng, named_rng
from src.simulation.referee_policy import RefereePolicy
from src.simulation.social_dialogue import SocialDialogueEngine
from src.simulation.tournament_2026 import TournamentManager
from src.simulation.tournament_match import TournamentMatchMixin
from src.simulation.tournament_scoring import TournamentScoringMixin
from src.simulation.tournament_checkpoint import (
    checkpoint_root_seed, load_checkpoint, save_checkpoint,
)


def test_named_streams_are_stable_and_isolated():
    assert derive_seed(42, "match", "A", "B") == derive_seed(42, "match", "A", "B")
    assert derive_seed(42, "match", "A", "B") != derive_seed(42, "match", "B", "A")
    assert named_rng(42, "score").random() == named_rng(42, "score").random()
    assert named_py_rng(42, "social").random() == named_py_rng(
        42, "social",
    ).random()
    assert 0 <= derive_seed(42, "fixture") < 2**32


def _social_agent(name: str, seed: int) -> SocietyAgent:
    return SocietyAgent(
        name,
        {
            "tier": "Semi-Core", "final_status_score": 55.0,
            "c1_win_rate": 52.0, "c3_major_exp": 45.0,
            "c5_pressure": 40.0,
        },
        initialization_rng=random.Random(seed),
        random_root_seed=derive_seed(91, "agent", name),
    )


def _social_dialogue_snapshot():
    first = _social_agent("Alpha", 1)
    second = _social_agent("Beta", 2)
    feed = SocialMediaFeed()
    dialogue = SocialDialogueEngine(feed)
    result = dialogue.run_post_match_dialogue(
        first, second, "Final",
        {
            "winner": "Alpha", "score": "2-1", "score_diff": 1,
            "key_event": "late goal", "drama_score": 0.8,
            "stage_pressure": 1.0, "ref_strictness": 0.6,
        },
        rng=named_rng(91, "match", "social_dialogue"),
    )
    return result, feed.posts


def test_referee_and_social_streams_are_replayable_and_global_rng_isolated():
    policy = RefereePolicy()
    first = SimpleNamespace(media_exposure=0.8)
    second = SimpleNamespace(media_exposure=0.2)
    referee_a = policy.sample(
        first, second, 0.7, rng=named_rng(91, "match", "referee"),
    )

    random.seed(311)
    np.random.seed(311)
    expected_global = (random.random(), float(np.random.random()))
    random.seed(311)
    np.random.seed(311)
    social_a = _social_dialogue_snapshot()
    referee_b = policy.sample(
        first, second, 0.7, rng=named_rng(91, "match", "referee"),
    )
    observed_global = (random.random(), float(np.random.random()))

    assert referee_a == referee_b
    assert social_a == _social_dialogue_snapshot()
    assert observed_global == expected_global


def test_world_day_headline_is_isolated_from_social_draw_count():
    import pandas as pd

    stats = pd.DataFrame(
        {
            "tier": ["Core Power"] * 10,
            "final_status_score": [60.0 + index for index in range(10)],
        },
        index=[f"Core {index}" for index in range(10)],
    )
    first = WorldEngine(stats, root_seed=91)
    second = WorldEngine(stats, root_seed=91)
    first.current_date = second.current_date = pd.Timestamp("2026-01-03")
    regular_post = second.handle_post

    def noisy_post(agent, *, rng):
        for _ in range(100):
            rng.random()
        regular_post(agent, rng=rng)

    second.handle_post = noisy_post
    first.run_day()
    second.run_day()
    first_headline = [
        post["content"] for post in first.feed.posts
        if post["author"] == "GLOBAL_FOOTBALL_NEWS"
    ]
    second_headline = [
        post["content"] for post in second.feed.posts
        if post["author"] == "GLOBAL_FOOTBALL_NEWS"
    ]
    assert first_headline == second_headline


def test_tournament_match_identity_is_forwarded_through_every_stage(monkeypatch):
    manager = object.__new__(TournamentManager)
    manager.root_seed = 91
    observed = {}
    agents = [SimpleNamespace(name=name) for name in ("a1", "a2", "ah", "aa")]

    def prepare(*args, **kwargs):
        observed["prepare"] = kwargs["match_seed"]
        return (*agents, "Alpha", "Beta", False, 0.5, {}, {}, {}, {}, {})

    def simulate(**kwargs):
        observed["simulate"] = kwargs["match_seed"]
        return {
            "s1": 1, "s2": 0, "xg1": 1.1, "xg2": 0.7,
            "winner_name": "Alpha", "drama_score": 0.4,
            "key_event": "goal", "verdict_json": "{}",
            "xg_context_line": "", "micro_summary": None,
            "reg_s1": 1, "reg_s2": 0, "went_to_extra_time": False,
            "pen1": None, "pen2": None, "tactical_1": {},
            "tactical_2": {}, "fused_1": {}, "fused_2": {},
            "eff_status_1": 55.0, "eff_status_2": 54.0,
        }

    def narrative(**kwargs):
        observed["narrative"] = kwargs["match_seed"]
        return 0.0, 0.0

    def finalize(**kwargs):
        observed["finalize"] = kwargs["match_seed"]
        return "Alpha"

    monkeypatch.setattr(manager, "_prepare_match", prepare)
    monkeypatch.setattr(manager, "_simulate_match", simulate)
    monkeypatch.setattr(manager, "_run_narrative_and_social", narrative)
    monkeypatch.setattr(manager, "_finalize_match_state", finalize)

    winner = TournamentMatchMixin.play_match(
        manager, "Alpha", "Beta", "Final", SimpleNamespace(),
        fixture_seed=17,
    )
    expected = derive_seed(
        91, "tournament_match", 17, "Final", "Alpha", "Beta",
    )
    assert winner == "Alpha"
    assert observed == {
        "prepare": expected, "simulate": expected,
        "narrative": expected, "finalize": expected,
    }


def test_tournament_scoring_uses_match_identity_not_environment_seed(
    monkeypatch,
):
    referee = {"strictness": 0.5}
    monkeypatch.setenv("GFS_SEED", "1")
    first = TournamentScoringMixin()._build_regulation_context(
        stage_name="Final", t1_name="Alpha", t2_name="Beta",
        referee=referee, match_seed=77,
    )
    monkeypatch.setenv("GFS_SEED", "999")
    second = TournamentScoringMixin()._build_regulation_context(
        stage_name="Final", t1_name="Alpha", t2_name="Beta",
        referee=referee, match_seed=77,
    )
    assert first["seed"] == second["seed"] == 77
    assert first["rng_score"].random() == second["rng_score"].random()


def test_legacy_journey_is_replayable_and_global_rng_isolated():
    import pandas as pd

    teams = [
        "Argentina", "Algeria", "Austria", "Jordan",
        *(f"Test Team {index}" for index in range(12)),
    ]
    stats = pd.DataFrame(
        {
            "final_status_score": [
                80.0, 50.0, 60.0, 40.0,
                *(79.0 - index for index in range(12)),
            ],
            "tier": ["Core"] * len(teams),
        },
        index=teams,
    )
    random.seed(601)
    expected_next = random.random()
    random.seed(601)
    first = simulate_journey("Argentina", stats, root_seed=91)
    observed_next = random.random()
    second = simulate_journey("Argentina", stats, root_seed=91)

    assert first == second
    assert observed_next == expected_next


def test_score_and_penalties_accept_reproducible_rngs():
    first = simulate_match_score(50, 45, rng=np.random.default_rng(7))
    second = simulate_match_score(50, 45, rng=np.random.default_rng(7))
    assert first == second
    assert simulate_penalty_shootout(np.random.default_rng(9)) == simulate_penalty_shootout(
        np.random.default_rng(9)
    )


def test_lightweight_monte_carlo_is_reproducible():
    from src.app import run_monte_carlo
    assert run_monte_carlo(2, seed=91) == run_monte_carlo(2, seed=91)


def test_checkpoint_round_trip_and_version_validation(tmp_path):
    kwargs = dict(
        standings={"A": {}}, qualified_teams=[], phase="group",
        group_schedule_progress={}, ko_round=None, ko_fixture_index=0,
        r32_fixtures=[], completed_matches=[], final_result={}, match_index=3,
        root_seed=91,
    )
    save_checkpoint(str(tmp_path), **kwargs)
    loaded = load_checkpoint(str(tmp_path))
    assert loaded["match_index"] == 3
    assert checkpoint_root_seed(loaded) == 91

    path = tmp_path / "data" / "persistence" / "tournament_checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported tournament checkpoint"):
        load_checkpoint(str(tmp_path))


def test_checkpoint_rejects_content_tampering_and_unsafe_v1(tmp_path):
    kwargs = dict(
        standings={"A": {}}, qualified_teams=[], phase="group",
        group_schedule_progress={}, ko_round=None, ko_fixture_index=0,
        r32_fixtures=[], completed_matches=[], final_result={}, match_index=3,
        root_seed=91,
    )
    save_checkpoint(str(tmp_path), **kwargs)
    path = tmp_path / "data" / "persistence" / "tournament_checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["match_index"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content integrity mismatch"):
        load_checkpoint(str(tmp_path))

    payload["version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks random-world identity"):
        load_checkpoint(str(tmp_path))


def test_resume_seed_comes_from_checkpoint_and_conflicts_fail(tmp_path, monkeypatch):
    from src import app

    kwargs = dict(
        standings={"A": {}}, qualified_teams=[], phase="group",
        group_schedule_progress={}, ko_round=None, ko_fixture_index=0,
        r32_fixtures=[], completed_matches=[], final_result={}, match_index=0,
        root_seed=91,
    )
    save_checkpoint(str(tmp_path), **kwargs)
    monkeypatch.setenv("GFS_SEED", "999")
    assert app._resolve_tournament_root_seed(
        tmp_path, resume=True, requested_seed=None,
    ) == 91
    assert app._resolve_tournament_root_seed(
        tmp_path, resume=True, requested_seed=91,
    ) == 91
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        app._resolve_tournament_root_seed(
            tmp_path, resume=True, requested_seed=92,
        )


def test_build_simulation_forwards_explicit_seed(tmp_path, monkeypatch):
    from src import app

    observed = {}

    def build(base_dir, **kwargs):
        observed.update(base_dir=base_dir, **kwargs)
        return "world", "tournament", "tactics"

    monkeypatch.setattr(app, "build_world_and_tournament", build)
    result = app.build_simulation(tmp_path, require_tactics=True, seed=91)
    assert result == ("world", "tournament", "tactics")
    assert observed == {
        "base_dir": str(tmp_path), "require_tactics": True,
        "initialization_seed": 91,
    }


def test_manager_rejects_checkpoint_for_another_random_world(tmp_path):
    kwargs = dict(
        standings={"A": {}}, qualified_teams=[], phase="group",
        group_schedule_progress={}, ko_round=None, ko_fixture_index=0,
        r32_fixtures=[], completed_matches=[], final_result={}, match_index=0,
        root_seed=91,
    )
    save_checkpoint(str(tmp_path), **kwargs)
    checkpoint = load_checkpoint(str(tmp_path))
    manager = object.__new__(TournamentManager)
    manager.root_seed = 92
    with pytest.raises(ValueError, match="does not match the current world"):
        manager._restore_from_checkpoint(checkpoint)


def test_tournament_manager_uses_callers_project_root(tmp_path):
    world = SimpleNamespace(root_seed=91, feed=SocialMediaFeed())
    manager = TournamentManager(world, base_dir=tmp_path)
    assert manager.base_dir == str(tmp_path.resolve())
