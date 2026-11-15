from pathlib import Path

import numpy as np

from gfs_core.contracts.events import MicroEvent, MicroEventType
from gfs_core.contracts.micro import MicroMatchConfig
from gfs_core.contracts.micro_state import CrowdState, MatchAffectiveState, RefereeAffectiveState
from gfs_core.contracts.society import TeamProfile
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.match.actions.plugin import ActionsPlugin
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin
from gfs_core.plugins.society.psychology.plugin import PsychologyPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def psychology_kernel():
    return PluginKernel(PluginRegistry([
        RostersPlugin(PROJECT_ROOT),
        SpatialPlugin(),
        AgentsPlugin(),
        PhysicsPlugin(),
        ActionsPlugin(MicroMatchConfig.fast_demo()),
        PsychologyPlugin(),
    ])).start(["gfs.core.society.agents", "gfs.core.society.psychology"])


def match_state(kernel):
    squads = kernel.context.require("agents")
    home = squads.build(TeamProfile("Brazil"), np.random.default_rng(42))
    away = squads.build(TeamProfile("Argentina"), np.random.default_rng(42))
    return MatchAffectiveState(home, away, RefereeAffectiveState(), CrowdState())


def test_affective_tick_applies_events_and_returns_modulators():
    kernel = psychology_kernel()
    state = match_state(kernel)
    coupling = kernel.context.require("psychology").create_coupling(
        MicroMatchConfig.fast_demo()
    )
    event = MicroEvent(
        0,
        MicroEventType.GOAL_SCORED,
        state.home.team_id,
        player_id=state.home.players[9].player_id,
    )
    modulators, scalars = coupling.step(state, 30.0, [event])
    assert state.home.score == 1
    assert state.clock_seconds == 30.0
    assert set(modulators) == {"home", "away"}
    assert len(modulators["home"]) == 23
    assert scalars["lambda_foul"] > 0
    assert np.isfinite(scalars["psi"])
    kernel.stop()


def test_discipline_emits_foul_events_with_forced_probability():
    kernel = psychology_kernel()
    state = match_state(kernel)
    config = MicroMatchConfig.fast_demo()
    config.discipline_foul_target_base = 10000
    coupling = kernel.context.require("psychology").create_coupling(config)
    home_mods = [
        coupling.compute_modulators(player, state)
        for player in state.home.players
    ]
    away_mods = [
        coupling.compute_modulators(player, state)
        for player in state.away.players
    ]
    events = kernel.context.require("psychology").emit_tick_discipline(
        state,
        home_mods,
        away_mods,
        cfg=config,
        lambda_foul=0.06,
        referee_strictness=0.5,
        drama_score=0.5,
        rng=np.random.default_rng(2),
    )
    assert events
    assert any(event.event_type == MicroEventType.FOUL_COMMITTED for event in events)
    kernel.stop()


def test_schedule_and_meso_aggregation_are_replayable():
    kernel = psychology_kernel()
    state = match_state(kernel)
    psychology = kernel.context.require("psychology")
    events = psychology.build_event_schedule(
        state,
        goals_home=1,
        goals_away=1,
        xg_home=1.2,
        xg_away=0.8,
        rng=np.random.default_rng(42),
    )
    assert len(events) == 63
    assert [event.t_sec for event in events] == sorted(event.t_sec for event in events)

    meso = psychology.create_meso()
    for team in (state.home, state.away):
        meso.record_tick(
            state,
            team_id=team.team_id,
            xg_for=0.2,
            xg_against=0.1,
            controversy=0.0,
            psi=0.1,
            emotion={"pride": 0.4},
            tactical={},
            tactical_base={},
            icon_shock=0.1,
        )
    packets = meso.flush_packets(state.home.team_id, state.away.team_id)
    assert len(packets) == 2
    assert all(packet.emotion_mean == {"pride": 0.4} for packet in packets)
    kernel.stop()
