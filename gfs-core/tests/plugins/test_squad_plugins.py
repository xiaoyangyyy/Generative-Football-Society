from pathlib import Path

import numpy as np

from gfs_core.contracts.society import TeamProfile
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.data.squads.repository import RosterRepository
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def profile(team="Brazil"):
    return TeamProfile(
        team_name=team,
        status_score=90,
        formation="4-2-3-1",
        tactical={},
        emotion_profile={
            "pride": 0.3, "anger": 0.1, "fear": 0.2, "determination": 0.4,
        },
        psychology_profile={"confidence": 0.7},
        fatigue=0.2,
        media_exposure=0.6,
        conflict_heat=0.1,
        team_cohesion=0.8,
        coach_name="Coach",
    )


def squad_kernel():
    return PluginKernel(PluginRegistry([
        RostersPlugin(PROJECT_ROOT), SpatialPlugin(), AgentsPlugin(),
    ])).start(["gfs.core.society.agents"])


def test_real_roster_builds_starting_xi_and_bench_with_provenance():
    kernel = squad_kernel()
    team = kernel.context.require("agents").build(
        profile(), np.random.default_rng(42),
    )
    assert len(team.players) == 23
    assert sum(player.on_pitch for player in team.players) == 11
    assert sum(player.squad_role == "bench" for player in team.players) == 12
    assert team.players[0].name == "Alisson"
    assert team.formation_key == "4231"
    assert team.squad_provenance["fallback_used"] is False
    assert len(team.squad_provenance["roster_identity"]) == 64
    assert team.icon_player_id
    kernel.stop()


def test_synthetic_fallback_is_deterministic_and_explicit():
    kernel = squad_kernel()
    service = kernel.context.require("agents")
    first = service.build(profile("No Roster"), np.random.default_rng(42))
    second = service.build(profile("No Roster"), np.random.default_rng(42))
    assert len(first.players) == 11
    assert first.squad_provenance["fallback_used"] is True
    assert first.squad_provenance["fallback_reason"] == "roster_unavailable"
    for left, right in zip(first.players, second.players):
        assert left.player_id == right.player_id
        np.testing.assert_array_equal(left.position, right.position)
        np.testing.assert_array_equal(left.z_emo, right.z_emo)
    kernel.stop()


def test_roster_repository_sanitizes_nonfinite_values():
    repository = RosterRepository(PROJECT_ROOT)
    roster = repository.load("Example", {
        "team_id": "Example",
        "players": [],
        "invalid": float("nan"),
    })
    assert "invalid" not in roster
    assert len(repository.identity(roster)) == 64
