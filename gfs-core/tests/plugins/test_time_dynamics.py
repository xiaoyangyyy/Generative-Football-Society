import numpy as np
import pytest

from gfs_core.plugins.match.physics.kinematics import integrate_position
from gfs_core.plugins.match.spatial.field import decay_density
from gfs_core.plugins.society.psychology.discipline import tick_foul_probability


def foul_probability(dt, scale=10.0):
    return tick_foul_probability(
        lambda_foul=0.06 * scale,
        lambda_foul_base=0.06,
        match_seconds=5400.0,
        dt=dt,
        referee_strictness=0.5,
        foul_target_base=14.0,
        foul_target_ref_slope=11.0,
    )


def test_poisson_interval_probability_composes_without_manual_cap():
    whole = foul_probability(30.0)
    first = foul_probability(15.0)
    second = foul_probability(15.0)
    assert whole > 0.35
    assert whole == pytest.approx(first + (1.0 - first) * second)
    assert 0.0 < whole < 1.0


def test_motion_and_density_decay_compose_across_dt():
    position = np.asarray([0.2, 0.2])
    velocity = np.asarray([0.001, 0.002])
    coarse = integrate_position(position, velocity, 2.0)
    fine = integrate_position(integrate_position(position, velocity, 1.0), velocity, 1.0)
    assert coarse == pytest.approx(fine)
    coarse_density = decay_density(np.asarray([2.0]), 0.08, 2.0)
    fine_density = decay_density(decay_density(np.asarray([2.0]), 0.08, 1.0), 0.08, 1.0)
    assert coarse_density == pytest.approx(fine_density)
    with pytest.raises(ValueError, match="timestep"):
        integrate_position(position, velocity, -1.0)
