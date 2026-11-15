import numpy as np

from gfs_studio.plugins.world_model.trace_codec import WorldModelTraceCodec


def test_kickoff_is_encoded_as_hold_without_outcome_labels():
    action = WorldModelTraceCodec().action(
        "kickoff", np.asarray([0.5, 0.5]), 30.0,
    )
    assert action[2] == 1.0
    assert np.all(action[14:17] == 0.0)
