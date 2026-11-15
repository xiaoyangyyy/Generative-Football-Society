from pathlib import Path

from gfs_core.bootstrap.application import CoreApplication
from gfs_core.contracts.micro_state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    PlayerAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from gfs_core.plugins.society.memory.service import SocietyMemory
from gfs_core.plugins.tournament.checkpointing.carryover import CrossMatchService, TeamCarryover

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_real_player_statistics_create_memory_and_bounded_next_state(tmp_path):
    summary = CoreApplication(
        PROJECT_ROOT, workspace_root=tmp_path / "match",
    ).run_micro_match(fast=True, seed=42)
    memory = SocietyMemory(tmp_path)
    carryover = TeamCarryover("Brazil")
    service = CrossMatchService(memory=memory)
    service.settle(carryover, summary, "match-1")
    player_id = next(iter(summary.player_stats["Brazil"]))
    performances = memory.recall(
        "Brazil", subject=f"player:{player_id}", kind="player_performance",
    )
    relations = memory.recall("Brazil", kind="coach_player_relation")
    assert len(performances) == 1
    assert performances[0]["context"]["player_id"] == player_id
    assert relations[0]["context"]["relation"] == "coach_player"
    player = PlayerAffectiveState(player_id, "Player", "CM", "Brazil")
    state = MatchAffectiveState(
        TeamAffectiveState("Brazil", players=[player], coach=CoachAffectiveState("Brazil")),
        TeamAffectiveState("Argentina"),
        RefereeAffectiveState(),
        CrowdState(),
    )
    applied = memory.apply_to_state(state)
    assert applied["players_affected"] == [player_id]
    assert player.morale_logit != 0.0
    assert 0.0 <= player.cognitive_load <= 1.0
    assert abs(state.home.coach.z_trust) <= 6.0
    assert applied["scope"] == "deterministic_memory_appraisal_only"
    snapshot = memory.snapshot()
    service.settle(carryover, summary, "match-1")
    assert memory.snapshot() == snapshot
    service.settle(carryover, summary, "match-2")
    assert memory.snapshot() != snapshot
    memory.restore(snapshot)
    assert memory.snapshot() == snapshot
