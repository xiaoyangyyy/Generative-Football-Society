import numpy as np

from gfs_core.contracts.micro import (
    CoachAffectiveState,
    MatchAffectiveState,
    MicroMatchConfig,
    PlayerAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from gfs_core.contracts.micro_state import CrowdState
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.match.spatial.formation import (
    interpolate_anchors,
    mirror_for_away,
    roles_for_formation,
)
from gfs_core.plugins.match.spatial.keys import normalize_formation_key
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin


def make_state():
    home = TeamAffectiveState(
        team_id="H",
        players=[PlayerAffectiveState("h1", "Home", "ST", "H")],
        coach=CoachAffectiveState(team_id="H"),
    )
    away = TeamAffectiveState(
        team_id="A",
        players=[PlayerAffectiveState("a1", "Away", "ST", "A")],
        coach=CoachAffectiveState(team_id="A"),
    )
    state = MatchAffectiveState(
        home, away, RefereeAffectiveState(), CrowdState(),
    )
    state.ball.possessor_id = "h1"
    state.ball.possession_team_id = "H"
    return state


def test_formation_roles_and_mirror_are_stable():
    assert normalize_formation_key("4-2-3-1 - 4-3-3") == "4231"
    assert normalize_formation_key(None) == "433"
    assert len(roles_for_formation("4-3-3")) == 11
    home = interpolate_anchors("ST", 0.5)
    away = mirror_for_away(home)
    assert home[0] + away[0] == 1.0
    assert home[1] == away[1]


def test_spatial_plugin_builds_finite_nonnegative_grids():
    kernel = PluginKernel(PluginRegistry([SpatialPlugin()])).start()
    service = kernel.context.require("spatial")
    config = MicroMatchConfig.fast_demo()
    field = service.create_field(config)
    state = make_state()
    grid = field.step(state, 1.0)
    assert grid.rho_home.shape == (16, 11)
    assert np.all(grid.rho_home >= 0)
    assert np.all(grid.rho_away >= 0)
    assert np.all(np.isfinite(grid.press))
    assert 0 <= field.advantage_at(grid, np.array([0.5, 0.5]), True) <= 1
    kernel.stop()
