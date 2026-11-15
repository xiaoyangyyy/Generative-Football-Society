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


def test_five_passing_steps_match_fixed_seed_reference():
    kernel = PluginKernel(PluginRegistry([
        RostersPlugin(PROJECT_ROOT),
        SpatialPlugin(),
        AgentsPlugin(),
        PhysicsPlugin(),
        ActionsPlugin(
            MicroMatchConfig.fast_demo(),
            ball_log=BallLogService(enabled=True),
            base_dir=PROJECT_ROOT,
        ),
    ])).start(["gfs.core.society.agents", "gfs.core.match.actions"])
    squads = kernel.context.require("agents")
    home = squads.build(profile("Brazil"), np.random.default_rng(42))
    away = squads.build(profile("Argentina"), np.random.default_rng(42))
    state = MatchAffectiveState(home, away, RefereeAffectiveState(), CrowdState())
    state.ball.possessor_id = home.players[9].player_id
    state.ball.possession_team_id = home.team_id
    state.ball.position = home.players[9].position.copy()
    intelligence = kernel.context.require("spatial").create_intelligence(
        MicroMatchConfig.fast_demo()
    )
    passing = kernel.context.require("actions").create_passing_engine(intelligence)
    assert passing.receiver_ranker is not None

    actions = [
        passing.step(
            state,
            [PlayerModulators("home")],
            [PlayerModulators("away")],
            np.random.default_rng(seed),
        )[0]
        for seed in range(42, 47)
    ]
    assert [action.pass_kind for action in actions] == [
        "long", "long", "short", "long", "long",
    ]
    assert [action.completed for action in actions] == [
        False, True, False, False, True,
    ]
    assert [action.success_p for action in actions] == pytest.approx([
        0.7235555169807472,
        0.7221149800078988,
        0.8551973765257518,
        0.7228838517314106,
        0.7226807896666435,
    ])
    assert passing.stats["home_attempts"] == 2
    assert passing.stats["away_attempts"] == 3
    assert passing.stats["home_completed"] == 0
    assert passing.stats["away_completed"] == 2
    assert len(state.ball_path_log) == 5
    assert len(state.pass_log) == 5

    lane = intelligence.lane_quality(
        state,
        home.players[5].position,
        home.players[9].position,
        away,
    )
    advantage = intelligence.numerical_advantage(
        state, home.players[9].position, True,
    )
    assert 0 < lane <= 1
    assert np.isfinite(advantage)
    kernel.stop()
