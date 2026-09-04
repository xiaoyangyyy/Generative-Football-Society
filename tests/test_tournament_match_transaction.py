import json
import random
from types import SimpleNamespace

import pandas as pd
import pytest

from src.infrastructure.locking import FileLease, LeaseUnavailable
from src.simulation.agent import SocietyAgent
from src.simulation.engine import SocialMediaFeed
from src.simulation.random_control import derive_seed
from src.simulation.tournament_2026 import TournamentManager
from src.simulation.tournament_checkpoint import load_checkpoint
from src.simulation.tournament_transaction import TournamentMatchRollbackError
from src.simulation.world_state import snapshot_world_state


RUN_IDENTITY = "d" * 64


def _agent(name: str, seed: int) -> SocietyAgent:
    return SocietyAgent(
        name,
        {
            "tier": "Semi-Core",
            "final_status_score": 55.0,
            "c1_win_rate": 52.0,
            "c3_major_exp": 45.0,
            "c5_pressure": 40.0,
        },
        initialization_rng=random.Random(seed),
        random_root_seed=derive_seed(91, "agent", name),
    )


def _manager(tmp_path) -> TournamentManager:
    agents = {
        name: _agent(name, index + 1)
        for index, name in enumerate(("Alpha", "Beta"))
    }
    world = SimpleNamespace(
        root_seed=91,
        agents=agents,
        feed=SocialMediaFeed(),
        current_date=pd.Timestamp("2026-01-07"),
    )
    return TournamentManager(
        world, base_dir=tmp_path, run_identity_sha256=RUN_IDENTITY,
    )


def _state_paths(tmp_path):
    return {
        "carryover": tmp_path / "data/persistence/squad_carryover.json",
        "cognitive": tmp_path / "data/persistence/cognitive_log/old.json",
        "ball": tmp_path / "outputs/ball_log/old.jsonl",
        "narrative": tmp_path / "outputs/narrative_debug.jsonl",
    }


def _seed_state_files(tmp_path):
    paths = _state_paths(tmp_path)
    for index, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"before-{index}", encoding="utf-8")
    return paths, {name: path.read_bytes() for name, path in paths.items()}


def test_failed_match_restores_memory_files_and_checkpoint(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    paths, original_bytes = _seed_state_files(tmp_path)
    world_before = snapshot_world_state(manager)
    standings_before = json.loads(json.dumps(manager.standings))
    new_cognitive = paths["cognitive"].parent / "new.json"
    new_ball = paths["ball"].parent / "new.jsonl"

    def fail_match(*args, **kwargs):
        manager.world.agents["Alpha"].momentum = 9.0
        manager.world.feed.publish("Alpha", "must roll back", tags=["test"])
        manager.standings = {"mutated": {}}
        for path in paths.values():
            path.write_text("after", encoding="utf-8")
        new_cognitive.write_text("new", encoding="utf-8")
        new_ball.write_text("new", encoding="utf-8")
        raise RuntimeError("private failure detail")

    monkeypatch.setattr(manager, "_play_match_once", fail_match)
    with pytest.raises(RuntimeError, match="private failure detail"):
        manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())

    assert snapshot_world_state(manager) == world_before
    assert manager.standings == standings_before
    assert manager.match_index == 0
    assert manager.completed_matches == []
    assert manager.match_results == {}
    for name, path in paths.items():
        assert path.read_bytes() == original_bytes[name]
    assert not new_cognitive.exists()
    assert not new_ball.exists()
    checkpoint = load_checkpoint(str(tmp_path))
    assert checkpoint["match_index"] == 0
    assert checkpoint["completed_matches"] == []
    assert checkpoint["match_results"] == {}
    assert checkpoint["world_state"] == world_before

    receipts = list((
        tmp_path / "data/persistence/tournament_match_rollbacks"
    ).glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["rollback_verified"] is True
    assert receipt["original_error_type"] == "RuntimeError"
    assert receipt["rollback_error_type"] is None
    assert receipt["contains_error_message"] is False
    assert "private failure detail" not in receipts[0].read_text(encoding="utf-8")


def test_successful_match_requires_and_verifies_one_durable_commit(
    tmp_path, monkeypatch,
):
    manager = _manager(tmp_path)

    def commit_match(*args, **kwargs):
        manager.world.agents["Alpha"].momentum = 1.5
        return manager._commit_match_result(
            stage_name="Final", t1_name="Alpha", t2_name="Beta",
            s1=2, s2=1, winner_name="Alpha",
        )

    monkeypatch.setattr(manager, "_play_match_once", commit_match)
    winner = manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())

    match_key = manager._match_key("Final", "Alpha", "Beta")
    assert winner == "Alpha"
    assert manager.match_index == 1
    assert manager.completed_matches == [match_key]
    assert manager.match_results == {match_key: "Alpha"}
    checkpoint = load_checkpoint(str(tmp_path))
    assert checkpoint["match_index"] == 1
    assert checkpoint["completed_matches"] == [match_key]
    assert checkpoint["world_state"]["agents"]["Alpha"]["state"]["momentum"] == 1.5


def test_return_without_commit_is_rolled_back(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    world_before = snapshot_world_state(manager)

    def forget_commit(*args, **kwargs):
        manager.world.agents["Alpha"].momentum = 7.0
        return "Alpha"

    monkeypatch.setattr(manager, "_play_match_once", forget_commit)
    with pytest.raises(RuntimeError, match="without one in-memory result commit"):
        manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())

    assert snapshot_world_state(manager) == world_before
    assert manager.match_index == 0
    assert manager.completed_matches == []


def test_commit_with_spurious_result_is_rolled_back(tmp_path, monkeypatch):
    manager = _manager(tmp_path)

    def corrupt_commit(*args, **kwargs):
        match_key = manager._match_key("Final", "Alpha", "Beta")
        manager.completed_matches.extend(["Other::A::B", match_key])
        manager.match_results.update({
            "Other::A::B": "A",
            match_key: "Alpha",
        })
        manager.match_index += 1
        manager._save_checkpoint()
        return "Alpha"

    monkeypatch.setattr(manager, "_play_match_once", corrupt_commit)
    with pytest.raises(RuntimeError, match="without one in-memory result commit"):
        manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())
    assert manager.match_index == 0
    assert manager.completed_matches == []
    assert manager.match_results == {}
    checkpoint = load_checkpoint(str(tmp_path))
    assert checkpoint["match_index"] == 0
    assert checkpoint["completed_matches"] == []
    assert checkpoint["match_results"] == {}


def test_rollback_failure_is_never_reported_as_original_failure(
    tmp_path, monkeypatch,
):
    manager = _manager(tmp_path)

    def fail_match(*args, **kwargs):
        raise RuntimeError("match failed")

    def fail_restore(*args, **kwargs):
        raise OSError("restore failed")

    monkeypatch.setattr(manager, "_play_match_once", fail_match)
    monkeypatch.setattr(
        "src.simulation.tournament_transaction.restore_state_artifacts",
        fail_restore,
    )
    with pytest.raises(TournamentMatchRollbackError) as error:
        manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())
    assert error.value.original_error_type == "RuntimeError"
    assert error.value.rollback_error_type == "OSError"


def test_unproven_in_memory_restore_fails_the_rollback_proof(
    tmp_path, monkeypatch,
):
    manager = _manager(tmp_path)

    def fail_match(*args, **kwargs):
        manager.world.agents["Alpha"].momentum = 8.0
        raise RuntimeError("match failed")

    monkeypatch.setattr(manager, "_play_match_once", fail_match)
    monkeypatch.setattr(manager, "_restore_from_checkpoint", lambda _value: None)
    with pytest.raises(TournamentMatchRollbackError) as error:
        manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())
    assert error.value.original_error_type == "RuntimeError"
    assert error.value.rollback_error_type == "RuntimeError"


def test_external_match_output_target_fails_before_execution(
    tmp_path, monkeypatch,
):
    manager = _manager(tmp_path)
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True

    external = tmp_path.parent / f"{tmp_path.name}-external-ball-log"
    monkeypatch.setenv("MATCH_BALL_LOG", "1")
    monkeypatch.setenv("MATCH_BALL_LOG_DIR", str(external))
    monkeypatch.setattr(manager, "_play_match_once", should_not_run)

    with pytest.raises(ValueError, match="project-owned state paths"):
        manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())
    assert called is False
    assert manager.match_index == 0
    assert not external.exists()


def test_workspace_match_lease_rejects_concurrent_writer(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    called = False

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(manager, "_play_match_once", should_not_run)
    lock_path = (
        tmp_path / "data/persistence/tournament_match_transaction.lock"
    )
    with FileLease(lock_path):
        with pytest.raises(LeaseUnavailable, match="already owned"):
            manager.play_match("Alpha", "Beta", "Final", SimpleNamespace())
    assert called is False
