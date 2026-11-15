import shutil
from pathlib import Path

import pytest

from gfs_core.bootstrap.application import CoreApplication
from gfs_core.plugins.data.squads.tactics import TacticsRepository
from gfs_core.plugins.match.scoring.service import MacroScoreService
from gfs_core.plugins.persistence.recovery.service import RecoveryService
from gfs_core.plugins.persistence.repositories.workspace import WorkspaceRepository
from gfs_core.plugins.persistence.transactions.match import MatchTransactionService
from gfs_core.plugins.society.memory.service import SocietyMemory
from gfs_core.plugins.tournament.checkpointing.carryover import PlayerCarryover, TeamCarryover, CrossMatchService
from gfs_core.plugins.tournament.checkpointing.service import CheckpointService
from gfs_core.plugins.tournament.engines import MacroMatchEngine
from gfs_core.plugins.tournament.runner import TournamentRunner
from gfs_core.plugins.tournament.scheduling.groups import WORLD_CUP_2026_GROUPS
from gfs_core.plugins.tournament.scheduling.official import OFFICIAL_GROUP_MATCHDAYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_runner(root, root_seed=42):
    workspace = WorkspaceRepository(root)
    return TournamentRunner(
        groups=WORLD_CUP_2026_GROUPS,
        engine=MacroMatchEngine(MacroScoreService()),
        transactions=MatchTransactionService(workspace),
        checkpointing=CheckpointService(workspace, RecoveryService(workspace)),
        status=CoreApplication(PROJECT_ROOT).load_status(),
        tactics=TacticsRepository(PROJECT_ROOT),
        cross_match=CrossMatchService(memory=SocietyMemory(root)),
        root_seed=root_seed,
    )


def test_official_group_schedule_has_three_matches_per_team():
    for group, teams in WORLD_CUP_2026_GROUPS.items():
        counts = {team: 0 for team in teams}
        for matchday in OFFICIAL_GROUP_MATCHDAYS[f"Group {group}"]:
            assert len(matchday) == 2
            for home, away in matchday:
                counts[home] += 1
                counts[away] += 1
        assert set(counts.values()) == {3}


def test_full_tournament_uses_real_status_and_settles_every_team(tmp_path):
    runner = make_runner(tmp_path)
    assert runner.profile_for("Brazil").status_score != runner.profile_for("Haiti").status_score
    result = runner.run()
    assert set(result) == {"champion", "runner_up"}
    assert result["champion"] != result["runner_up"]
    assert runner.match_index == 103
    assert len(runner.completed_matches) == 103
    assert runner.phase == "complete"
    assert runner.play_next() is None
    assert runner.run() == result
    assert all(team.settled_match_ids for team in runner.carryover.values())
    assert all(team.fatigue_ema > 0 for team in runner.carryover.values())
    assert any(team.morale_ema != 0.55 for team in runner.carryover.values())


def test_partial_tournament_resume_finishes_with_identical_result(tmp_path):
    runner = make_runner(tmp_path)
    for _ in range(7):
        runner.play_next()
    resumed_root = tmp_path / "resumed"
    resumed_root.mkdir()
    shutil.copytree(
        tmp_path / "data" / "persistence",
        resumed_root / "data" / "persistence",
        dirs_exist_ok=True,
    )
    resumed = make_runner(resumed_root)
    result = resumed.resume()
    expected = make_runner(tmp_path / "expected").run()
    assert result == expected
    assert resumed.match_index == 103
    assert resumed.phase == "complete"


def test_scenario_identity_separates_seeds_and_invalid_resume(tmp_path):
    first = make_runner(tmp_path / "first")
    second = make_runner(tmp_path / "second", root_seed=43)
    assert first.scenario_id != second.scenario_id
    first_result = first.play_next()
    second_result = second.play_next()
    assert first_result["key"] != second_result["key"]
    invalid = first.snapshot()
    invalid["scenario_id"] = second.scenario_id
    with pytest.raises(ValueError, match="scenario identity mismatch"):
        first._restore(invalid)


def test_failed_fixture_rolls_back_fixture_cursor_and_state(tmp_path):
    runner = make_runner(tmp_path)
    original = runner.engine.scoring.simulate

    def fail(*args, **kwargs):
        raise RuntimeError("injected scoring failure")

    runner.engine.scoring.simulate = fail
    with pytest.raises(RuntimeError, match="injected scoring failure"):
        runner.play_next()
    assert runner.match_index == 0
    assert runner.completed_matches == {}
    assert all(not team.settled_match_ids for team in runner.carryover.values())
    runner.engine.scoring.simulate = original
    result = runner.play_next()
    assert result["key"] in runner.completed_matches
    assert result["engine"] == "macro"
    assert runner.match_index == 1


def test_failed_settlement_rolls_back_memory_with_match(tmp_path):
    runner = make_runner(tmp_path)
    original = runner.cross_match.settle_outcome

    def fail_after_memory(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-memory failure")

    runner.cross_match.settle_outcome = fail_after_memory
    with pytest.raises(RuntimeError, match="post-memory"):
        runner.play_next()
    assert runner.memory.snapshot()["records"] == []
    assert runner.match_index == 0
    assert all(not team.settled_match_ids for team in runner.carryover.values())


def test_carryover_changes_effective_status_and_checkpoint_identity(tmp_path):
    runner = make_runner(tmp_path)
    before = runner.profile_for("Mexico")
    result = runner.play_next()
    after = runner.profile_for("Mexico")
    assert after.fatigue > before.fatigue
    assert after.status_score < before.status_score
    assert result["key"] in runner.carryover["Mexico"].settled_match_ids
    invalid = runner.snapshot()
    invalid["base_status"]["Brazil"] += 1
    with pytest.raises(ValueError, match="base team status"):
        runner._restore(invalid)


def test_cross_match_recovery_is_bounded_and_idempotent():
    service = CrossMatchService()
    carryover = TeamCarryover(
        "Brazil",
        players={
            "p1": PlayerCarryover(
                "p1",
                suspension_matches_left=1,
                injury_matches_left=2,
                injury_severity=0.5,
            ),
        },
        fatigue_ema=0.8,
        media_pressure=0.4,
    )
    service.recover(carryover, "rest-1", rest_units=1.0)
    service.recover(carryover, "rest-1", rest_units=1.0)
    assert carryover.players["p1"].suspension_matches_left == 0
    assert carryover.players["p1"].injury_matches_left == 1
    assert 0 < carryover.fatigue_ema < 0.8
    assert 0 < carryover.media_pressure < 0.4
    restored = TeamCarryover.from_dict(carryover.to_dict())
    assert restored.to_dict() == carryover.to_dict()
