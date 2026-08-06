import numpy as np
import pytest

from src.match_engine.world_model.rollout_calibration import (
    apply_rollout_blend,
    select_rollout_blend,
)


def test_rollout_blend_is_bounded_residual_shrinkage():
    initial = np.array([[0.2, 0.8]], dtype=np.float32)
    prediction = np.array([[0.8, 0.2]], dtype=np.float32)
    assert np.allclose(apply_rollout_blend(initial, prediction, 0.5), [[0.5, 0.5]])
    assert np.allclose(apply_rollout_blend(initial, prediction, -1), initial)
    assert np.allclose(apply_rollout_blend(initial, prediction, 2), prediction)


def test_select_rollout_blend_finds_general_residual_scale():
    initial = np.zeros((4, 2), dtype=np.float32)
    prediction = np.ones((4, 2), dtype=np.float32)
    target = np.full((4, 2), 0.4, dtype=np.float32)
    result = select_rollout_blend(
        initial, prediction, target, np.ones(2, dtype=np.float32),
        candidates=(0.2, 0.4, 0.8),
    )
    assert result["selected_blend"] == 0.4
    assert result["weighted_mse"] < result["persistence_weighted_mse"]


def test_select_rollout_blend_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        select_rollout_blend(
            np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)), np.ones(2)
        )


def test_runtime_imagination_applies_blend_only_to_multi_step_rollout():
    torch = pytest.importorskip("torch")
    from src.match_engine.world_model.action_codec import ACTION_DIM
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.observation import OBS_DIM

    cfg = WorldModelConfig(
        latent_dim=16, hidden_dim=32, ensemble_size=2,
        transition_ensemble_size=2, rollout_residual_blend=1.0,
    )
    model = build_model(cfg)
    observation = np.linspace(0.0, 1.0, OBS_DIM, dtype=np.float32)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0], action[13] = 1.0, 0.6
    raw_one = model.imagine(observation, action, steps=1)
    raw_two = model.imagine(observation, action, steps=2)
    model.cfg.rollout_residual_blend = 0.4
    calibrated_one = model.imagine(observation, action, steps=1)
    calibrated_two = model.imagine(observation, action, steps=2)
    assert np.allclose(calibrated_one.next_obs, raw_one.next_obs)
    assert np.allclose(
        calibrated_two.next_obs,
        observation + 0.4 * (raw_two.next_obs - observation),
        atol=1e-6,
    )
