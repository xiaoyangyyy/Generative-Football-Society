from pathlib import Path

import numpy as np
import pytest

from gfs_studio.api import world_model_status
from gfs_studio.plugins.world_model.action_codec import ACTION_DIM
from gfs_studio.plugins.world_model.observation import OBS_DIM
from gfs_studio.plugins.world_model.runtime import WorldModelRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "data" / "world-model" / "latent_wm_rollout_calibrated_candidate.pt"


def test_real_checkpoint_status_and_signature():
    runtime = WorldModelRuntime.load(CHECKPOINT)
    status = runtime.status()
    assert status["checkpoint_signature"] == "0e58e49cff6cd6e11f3598fb8c624aabaa290870ca0d64daaf33b17771e3ce28"
    assert status["checkpoint_version"] == 9
    assert status["transition_kind"] == "gru"
    assert status["transition_members"] == 3
    assert status["transition_ensemble_trained"] is True
    assert status["shot_quality"] == 0.0
    assert status["cross_quality"] == 0.0
    assert status["deployed"] is False
    assert world_model_status(PROJECT_ROOT)["checkpoint_signature"] == status["checkpoint_signature"]


def test_real_checkpoint_inference_is_deterministic_and_bounded():
    runtime = WorldModelRuntime.load(CHECKPOINT)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = 1.0
    action[13] = 0.7
    action[17] = 0.5
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    first = runtime.imagine(observation, action)
    second = runtime.imagine(observation, action)
    assert first.next_obs.shape == (OBS_DIM,)
    assert first.pass_success == pytest.approx(0.7065701263339648)
    assert first.progress_delta == pytest.approx(-0.4110029637813568)
    assert first.shot_goal_prob == pytest.approx(0.7162900567054749)
    assert first.uncertainty == pytest.approx(0.589681464817231)
    assert first.pass_success == second.pass_success
    assert np.array_equal(first.next_obs, second.next_obs)
    assert runtime.planner_authority(observation, kind="shot")["authorized"] is False


def test_invalid_runtime_inputs_are_sanitized_without_outcome_leakage():
    runtime = WorldModelRuntime.load(CHECKPOINT)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = 1.0
    action[13] = 0.7
    action[14:16] = np.nan
    observation = np.full(OBS_DIM, np.nan, dtype=np.float32)
    output = runtime.imagine(observation, action)
    assert np.isfinite(output.next_obs).all()
    assert np.isfinite(output.pass_success)
