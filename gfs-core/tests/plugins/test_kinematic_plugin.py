import numpy as np

from gfs_core.contracts.micro import MicroMatchConfig
from gfs_core.contracts.micro_state import CrowdState, MatchAffectiveState, RefereeAffectiveState
from gfs_core.contracts.society import TeamProfile
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin


def test_kinematic_layer_moves_players_and_keeps_goalkeepers_deep(tmp_path):
    plugins = [
        RostersPlugin(tmp_path), SpatialPlugin(), AgentsPlugin(), PhysicsPlugin(),
    ]
    kernel = PluginKernel(PluginRegistry(plugins)).start([
        "gfs.core.society.agents", "gfs.core.match.physics",
    ])
    squads = kernel.context.require("agents")
    home = squads.build(TeamProfile("Home"), np.random.default_rng(1))
    away = squads.build(TeamProfile("Away"), np.random.default_rng(2))
    away.attacks_high_x = False
    for player in away.players:
        player.position = kernel.context.require("spatial").mirror_for_away(player.position)
    state = MatchAffectiveState(home, away, RefereeAffectiveState(), CrowdState())
    layer = kernel.context.require("physics").create_kinematics(
        MicroMatchConfig.fast_demo()
    )
    initial = home.players[9].position.copy()
    layer.step(state, [], [])
    assert not np.array_equal(home.players[9].position, initial)
    assert all(np.all(np.isfinite(player.position)) for player in home.players)
    assert home.players[0].position[0] < 0.2
    assert away.players[0].position[0] > home.players[0].position[0]
    assert 0 <= home.phase <= 1
    assert 0 <= away.phase <= 1
    kernel.stop()
