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
