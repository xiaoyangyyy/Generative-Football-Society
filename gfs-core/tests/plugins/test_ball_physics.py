import numpy as np

from gfs_core.contracts.ball import BallActionParams
from gfs_core.contracts.micro import MicroMatchConfig
from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin


def test_physics_service_integrates_shot_and_pass_deterministically():
    kernel = PluginKernel(PluginRegistry([
        SpatialPlugin(), PhysicsPlugin(),
    ])).start(["gfs.core.match.physics"])
    physics = kernel.context.require("physics")
    config = MicroMatchConfig.fast_demo()
    params = BallActionParams(
        0.8, 0.2, 0.1, 12.0, np.array([0.0, 0.0, 1.0]), 0.1,
    )
    first = physics.integrate_trajectory(
        np.array([0.8, 0.5]), params, config, rng=np.random.default_rng(42),
    )
    second = physics.integrate_trajectory(
        np.array([0.8, 0.5]), params, config, rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(first.landed, second.landed)
    assert first.time_of_flight == second.time_of_flight
    assert first.peak_height > 0
    assert np.isfinite(first.curve_amount)

    passed = physics.integrate_pass_trajectory(
        np.array([0.3, 0.4]),
        np.array([0.7, 0.6]),
        params,
        config,
        rng=np.random.default_rng(42),
    )
    assert np.isfinite(passed.target_miss)
    assert np.isfinite(passed.lateral_dev)
    assert passed.time_of_flight > 0
    kernel.stop()
