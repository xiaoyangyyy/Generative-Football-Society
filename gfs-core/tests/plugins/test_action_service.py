import numpy as np

from gfs_core.contracts.micro import MicroMatchConfig
from gfs_core.contracts.micro_state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    PlayerAffectiveState,
    PlayerModulators,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.match.actions.plugin import ActionsPlugin
from gfs_core.plugins.match.actions.sampling import mask_infeasible_actions, sample_action
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin


class Shots:
    stats = {"home_shots": 1, "away_shots": 0}

    @staticmethod
    def distance_to_goal(position, attacking_home):
        return 0.2

    @staticmethod
    def shot_utility_max(state, carrier, modulator):
        return 0.3


def make_state():
    carrier = PlayerAffectiveState(
        "h1", "Home", "ST", "H", position=np.array([0.8, 0.5]),
    )
    home = TeamAffectiveState("H", [carrier], coach=CoachAffectiveState(team_id="H"))
    away = TeamAffectiveState(
        "A",
        [PlayerAffectiveState("a1", "Away", "ST", "A")],
        coach=CoachAffectiveState(team_id="A"),
    )
    return MatchAffectiveState(home, away, RefereeAffectiveState(), CrowdState()), carrier


def test_action_plugin_selects_valid_action_with_fixed_seed():
    kernel = PluginKernel(PluginRegistry([
        SpatialPlugin(), PhysicsPlugin(), ActionsPlugin(MicroMatchConfig.fast_demo()),
    ])).start(["gfs.core.match.actions"])
    service = kernel.context.require("actions")
    state, carrier = make_state()
    selection = service.select(
        state, carrier, [PlayerModulators("h1")], [], Shots(), np.random.default_rng(42),
    )
    assert selection.choice == "shot"
    assert selection.probabilities.sum() == 1.0
    assert selection.probabilities[2] == 0.0
    assert selection.sampling_uniform == np.random.default_rng(42).random()
    assert selection.feasible_actions == frozenset({"pass", "shot", "hold"})
    kernel.stop()


def test_scheduled_action_and_feasibility_mask_are_explicit():
    kernel = PluginKernel(PluginRegistry([
        SpatialPlugin(), PhysicsPlugin(), ActionsPlugin(),
    ])).start(["gfs.core.match.actions"])
    state, carrier = make_state()
    selection = kernel.context.require("actions").select(
        state, carrier, [], [], Shots(), np.random.default_rng(42), planned_kind="pass",
    )
    assert selection.choice == "pass"
    probabilities = mask_infeasible_actions(
        ["pass", "shot"], [0.5, 0.5], {"pass"},
    )
    np.testing.assert_array_equal(probabilities, [1.0, 0.0])
    assert sample_action(["pass", "shot"], probabilities, 0.99) == "pass"
    kernel.stop()
