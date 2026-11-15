from pathlib import Path

import numpy as np
import pytest

from gfs_core.contracts.micro import MicroMatchConfig
from gfs_core.contracts.micro_state import (
    CrowdState,
    MatchAffectiveState,
    PlayerModulators,
    RefereeAffectiveState,
)
from gfs_core.contracts.society import TeamProfile
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.match.actions.log import BallLogService
from gfs_core.plugins.match.actions.plugin import ActionsPlugin
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def profile(team):
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


def execution_kernel():
    log = BallLogService(enabled=True)
    kernel = PluginKernel(PluginRegistry([
        RostersPlugin(PROJECT_ROOT),
        SpatialPlugin(),
        AgentsPlugin(),
        PhysicsPlugin(),
        ActionsPlugin(MicroMatchConfig.fast_demo(), ball_log=log),
    ])).start(["gfs.core.society.agents", "gfs.core.match.actions"])
    return kernel


def test_shot_execution_matches_reference_and_records_replay_entry():
    kernel = execution_kernel()
    squads = kernel.context.require("agents")
    home = squads.build(profile("Brazil"), np.random.default_rng(42))
    away = squads.build(profile("Argentina"), np.random.default_rng(42))
    state = MatchAffectiveState(home, away, RefereeAffectiveState(), CrowdState())
    intelligence = kernel.context.require("spatial").create_intelligence(
        MicroMatchConfig.fast_demo()
    )
    shots = kernel.context.require("actions").create_shot_engine(intelligence)
    outcome = shots.resolve_shot(
        state,
        home.players[9],
        PlayerModulators("shooter"),
        np.random.default_rng(42),
        forced_kind="driven",
    )
    assert outcome.xg == pytest.approx(0.10064194706128142)
    assert outcome.goal is False
    assert outcome.on_target is False
    assert shots.stats["home_shots"] == 1
    assert state.micro_xg_home == pytest.approx(outcome.xg)
    assert len(state.ball_path_log) == 1
    assert state.ball_path_log[0]["type"] == "shot"
    assert state.ball_path_log[0]["outcome"] == "OFF_TARGET"
    kernel.stop()


def test_aerial_execution_matches_reference_and_keeps_action_statistics():
    kernel = execution_kernel()
    squads = kernel.context.require("agents")
    home = squads.build(profile("Brazil"), np.random.default_rng(42))
    away = squads.build(profile("Argentina"), np.random.default_rng(42))
    state = MatchAffectiveState(home, away, RefereeAffectiveState(), CrowdState())
    aerial = kernel.context.require("actions").create_aerial_engine()
    trajectory, outcome = aerial.resolve_cross(
        state, home.players[1], np.random.default_rng(42),
    )
    assert outcome.contact == "header"
    assert outcome.xg_added == pytest.approx(0.004515266062638252)
    assert outcome.goal is False
    assert trajectory.landed.shape == (2,)
    assert aerial.stats["crosses"] == 1
    assert len(state.ball_path_log) == 0
    kernel.stop()
