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

    cfg = WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=2)
    checkpoint = tmp_path / "wm.pt"
    save_checkpoint(str(checkpoint), build_model(cfg), cfg, {"validation": {}})
    runtime = WorldModelRuntime.load(str(checkpoint))
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
