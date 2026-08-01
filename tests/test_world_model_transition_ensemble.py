"""Transition ensemble members have independent supervision and trajectories."""

import pytest


def test_bootstrap_transition_loss_routes_samples_per_member():
    torch = pytest.importorskip("torch")
    from scripts.train_world_model import _bootstrap_transition_loss

    predictions = torch.tensor([
        [[1.0], [9.0]],
        [[8.0], [2.0]],
    ], requires_grad=True)
    target = torch.zeros((2, 1))
    feature_weights = torch.ones(1)
    bootstrap = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    loss, members = _bootstrap_transition_loss(
        predictions, target, feature_weights, bootstrap,
    )
    loss.backward()

    assert members.tolist() == pytest.approx([1.0, 4.0])
    assert loss.item() == pytest.approx(2.5)
    assert predictions.grad[0, 0, 0] != 0.0
    assert predictions.grad[0, 1, 0] == 0.0
    assert predictions.grad[1, 0, 0] == 0.0
    assert predictions.grad[1, 1, 0] != 0.0


def test_bootstrap_transition_loss_rejects_shared_weight_shape():
    torch = pytest.importorskip("torch")
    from scripts.train_world_model import _bootstrap_transition_loss

    with pytest.raises(ValueError, match="bootstrap shapes"):
        _bootstrap_transition_loss(
            torch.zeros((3, 4, 2)),
            torch.zeros((4, 2)),
            torch.ones(2),
            torch.ones(4),
        )


def test_changing_action_rollout_preserves_member_trajectories():
    torch = pytest.importorskip("torch")
    from src.match_engine.world_model.action_codec import ACTION_DIM
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.observation import OBS_DIM

    cfg = WorldModelConfig(
        latent_dim=16,
        hidden_dim=32,
        ensemble_size=2,
        transition_ensemble_size=2,
    )
    model = build_model(cfg).eval()
    observations = torch.rand((3, OBS_DIM))
    actions = torch.zeros((3, 2, ACTION_DIM))
    actions[:, 0, 0] = 1.0
    actions[:, 1, 1] = 1.0

    with torch.no_grad():
        rollout = model.transition_rollout_predictions(observations, actions)

    assert rollout.shape == (2, 3, OBS_DIM)
    assert torch.isfinite(rollout).all()
    assert torch.all((rollout >= 0.0) & (rollout <= 1.0))
    with pytest.raises(ValueError, match="batch, steps"):
        model.transition_rollout_predictions(
            observations, torch.zeros((3, ACTION_DIM)),
        )


def test_two_step_holdout_pairs_require_same_group_and_state_alignment():
    import numpy as np

    from scripts.train_world_model import _sequential_holdout_pairs

    observations = np.zeros((5, 3), dtype=np.float32)
    next_observations = np.zeros((5, 3), dtype=np.float32)
    groups = np.asarray(["a", "a", "b", "b", "b"])
    next_observations[0] = observations[1]
    next_observations[2] = observations[3]
    next_observations[3] = observations[4] + 0.5

    pairs = _sequential_holdout_pairs(
        observations,
        next_observations,
        groups,
        np.asarray([0, 1, 2, 3, 4]),
    )

    assert pairs.tolist() == [0, 2]
