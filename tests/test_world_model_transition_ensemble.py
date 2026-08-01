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


def test_imagination_exposes_member_aligned_intermediate_states():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from src.match_engine.world_model.action_codec import ACTION_DIM
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.observation import OBS_DIM

    model = build_model(WorldModelConfig(
        latent_dim=16,
        hidden_dim=32,
        ensemble_size=2,
        transition_ensemble_size=2,
    )).eval()
    output = model.imagine(
        np.full(OBS_DIM, 0.5, dtype=np.float32),
        np.zeros(ACTION_DIM, dtype=np.float32),
        steps=2,
    )
    trajectory = output.uncertainty_samples[
        "transition_state_trajectory"
    ]

    assert trajectory.shape == (2, 2, 1, OBS_DIM)
    assert np.isfinite(trajectory).all()
    assert np.allclose(
        trajectory[-1], output.uncertainty_samples["transition_states"]
    )


def test_two_step_pairs_require_same_group_split_and_state_alignment():
    import numpy as np

    from scripts.train_world_model import _sequential_transition_pairs

    observations = np.zeros((5, 3), dtype=np.float32)
    next_observations = np.zeros((5, 3), dtype=np.float32)
    groups = np.asarray(["a", "a", "b", "b", "b"])
    next_observations[0] = observations[1]
    next_observations[2] = observations[3]
    next_observations[3] = observations[4] + 0.5

    pairs = _sequential_transition_pairs(
        observations,
        next_observations,
        groups,
        np.asarray([0, 1, 2, 3, 4]),
    )

    assert pairs.tolist() == [0, 2]


def test_multi_step_curriculum_warms_up_then_reaches_configured_weight():
    from scripts.train_world_model import _multi_step_curriculum_weight

    weights = [
        _multi_step_curriculum_weight(0.25, epoch, 10, warmup_fraction=0.20)
        for epoch in range(10)
    ]

    assert weights[:2] == [0.0, 0.0]
    assert weights[2] > 0.0
    assert weights == sorted(weights)
    assert weights[-1] == pytest.approx(0.25)
    assert _multi_step_curriculum_weight(0.4, 0, 1) == pytest.approx(0.4)
    assert _multi_step_curriculum_weight(-1.0, 5, 10) == 0.0


def test_two_step_bootstrap_keeps_member_gradient_routing():
    torch = pytest.importorskip("torch")
    from scripts.train_world_model import _bootstrap_transition_loss

    rollout_predictions = torch.tensor([
        [[1.0], [10.0]],
        [[10.0], [2.0]],
    ], requires_grad=True)
    loss, member_losses = _bootstrap_transition_loss(
        rollout_predictions,
        torch.zeros((2, 1)),
        torch.ones(1),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    loss.backward()

    assert member_losses.tolist() == pytest.approx([1.0, 4.0])
    assert rollout_predictions.grad[0, 1, 0] == 0.0
    assert rollout_predictions.grad[1, 0, 0] == 0.0


def test_autoregressive_two_step_loss_reaches_each_dynamics_member():
    torch = pytest.importorskip("torch")
    from scripts.train_world_model import _bootstrap_transition_loss
    from src.match_engine.world_model.action_codec import ACTION_DIM
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.observation import OBS_DIM

    model = build_model(WorldModelConfig(
        latent_dim=16,
        hidden_dim=32,
        ensemble_size=2,
        transition_ensemble_size=2,
    )).train()
    observations = torch.rand((2, OBS_DIM))
    actions = torch.rand((2, 2, ACTION_DIM))
    targets = torch.rand((2, OBS_DIM))
    predictions = model.transition_rollout_predictions(observations, actions)
    loss, _ = _bootstrap_transition_loss(
        predictions,
        targets,
        torch.ones(OBS_DIM),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    loss.backward()

    primary_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.gru.parameters()
        if parameter.grad is not None
    )
    secondary_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.extra_grus[0].parameters()
        if parameter.grad is not None
    )
    assert primary_grad > 0.0
    assert secondary_grad > 0.0


def test_semantic_event_heads_are_member_aligned_and_trainable():
    torch = pytest.importorskip("torch")
    from scripts.train_world_model import _bootstrap_transition_loss
    from src.match_engine.world_model.action_codec import ACTION_DIM
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.observation import OBS_DIM
    from src.match_engine.world_model.semantic_event_training import (
        bootstrap_semantic_event_loss,
    )

    model = build_model(WorldModelConfig(
        latent_dim=16,
        hidden_dim=32,
        ensemble_size=2,
        transition_ensemble_size=2,
    )).train()
    observations = torch.rand((2, OBS_DIM))
    actions = torch.rand((2, ACTION_DIM))
    _, future_members = model.transition_predictions(observations, actions)
    logits = model.semantic_event_logits(
        observations, future_members, actions,
    )
    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    bootstrap = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    event_loss, _ = bootstrap_semantic_event_loss(
        logits, targets, bootstrap,
    )
    transition_loss, _ = _bootstrap_transition_loss(
        future_members,
        torch.rand((2, OBS_DIM)),
        torch.ones(OBS_DIM),
        bootstrap,
    )
    (event_loss + transition_loss).backward()

    assert logits.shape == (2, 2, 4)
    assert all(
        head.weight.grad is not None
        and float(head.weight.grad.abs().sum()) > 0.0
        for head in model.semantic_event_heads
    )
