import numpy as np
import pytest
import json

from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.shot_head import FrozenShotHead


def _head(*, accepted=True, brier=0.10, prior=0.12):
    latent_dim = 4
    size = latent_dim + ACTION_DIM
    return FrozenShotHead(
        base_checkpoint_sha256="sha256:" + "a" * 64,
        latent_dim=latent_dim,
        mean=np.zeros(size),
        scale=np.ones(size),
        weights=np.ones(size) * 0.01,
        bias=-2.0,
        calibration_scale=1.0,
        calibration_bias=0.0,
        evidence={"sealed_test": {
            "samples": 80, "goals": 10, "model_brier": brier,
            "physics_xg_prior_brier": prior, "sealed_only": True,
        }},
        accepted=accepted,
    )


def test_frozen_shot_head_requires_checkpoint_identity_and_sealed_gain():
    head = _head()
    head.assert_compatible("sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="different"):
        head.assert_compatible("sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="sealed authority"):
        _head(brier=0.13).assert_compatible("sha256:" + "a" * 64)


def test_frozen_shot_head_round_trip_and_prediction():
    restored = FrozenShotHead.from_dict(_head().to_dict())
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    probability = restored.predict(np.ones(4), action)
    assert 0.0 < probability < 1.0
    assert restored.enabled


def test_runtime_only_attaches_a_head_bound_to_its_exact_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.inference import WorldModelRuntime
    from src.match_engine.world_model.model import build_model, save_checkpoint
    from src.match_engine.world_model.observation import OBS_DIM

    cfg = WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=2)
    checkpoint = tmp_path / "wm.pt"
    save_checkpoint(str(checkpoint), build_model(cfg), cfg, {"validation": {}})
    runtime = WorldModelRuntime.load(str(checkpoint))
    assert runtime.shot_quality == 0.0
    assert runtime.shot_probability_source == "physics_xg_prior"
    size = cfg.latent_dim + ACTION_DIM
    artifact = FrozenShotHead(
        base_checkpoint_sha256=runtime.checkpoint_signature,
        latent_dim=cfg.latent_dim,
        mean=np.zeros(size), scale=np.ones(size), weights=np.zeros(size),
        bias=-2.0, calibration_scale=1.0, calibration_bias=0.0,
        evidence={"sealed_test": {
            "samples": 80, "goals": 10, "model_brier": 0.10,
            "physics_xg_prior_brier": 0.12, "sealed_only": True,
        }},
        accepted=True,
    )
    path = tmp_path / "shot.json"
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    runtime.attach_shot_head(str(path))
    assert runtime.shot_probability_source == "frozen_backbone_shot_head"
    assert runtime.frozen_shot_head is not None
    observation = np.ones(OBS_DIM, dtype=np.float32)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[1] = 1.0
    action[13] = 0.4
    expected_probability = artifact.predict(
        runtime.model.encode(
            pytest.importorskip("torch").from_numpy(observation).unsqueeze(0)
        ).detach().squeeze(0).numpy(),
        action,
    )
    assert runtime.planner_authority(
        observation, kind="shot",
    )["authorized"] is True
    assert runtime.score_shot_action(
        observation, action, attacking_home=True,
    ) == pytest.approx(expected_probability)


def test_joint_shot_head_is_diagnostic_only_without_sealed_promotion():
    pytest.importorskip("torch")
    from src.match_engine.world_model.action_codec import encode_shot_action
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.inference import WorldModelRuntime
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.observation import OBS_DIM

    cfg = WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=2)
    runtime = WorldModelRuntime(build_model(cfg), cfg, {
        "validation": {
            "planner_quality": 0.9,
            "shot_planner_quality": 0.9,
            "weighted_obs_mse": 0.01,
        },
    })
    observation = np.ones(OBS_DIM, dtype=np.float32)
    action = encode_shot_action(
        np.array([0.8, 0.5], dtype=np.float32), xg=0.23,
    )

    assert runtime.joint_shot_quality == pytest.approx(0.9)
    assert runtime.shot_quality == 0.0
    assert runtime.shot_probability_source == "physics_xg_prior"
    assert runtime.planner_authority(
        observation, kind="shot",
    )["authorized"] is False
    assert runtime.score_shot_action(
        observation, action, attacking_home=True,
    ) == pytest.approx(0.23)
