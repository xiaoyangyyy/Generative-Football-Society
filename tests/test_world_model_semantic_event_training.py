"""Learned semantic events use proper labels, bootstrap and grouped gates."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.world_model.inference import WorldModelRuntime
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.semantic_event_training import (
    bootstrap_semantic_event_loss,
    bootstrap_semantic_path_loss,
    semantic_event_validation,
    semantic_path_validation,
)
from src.match_engine.world_model.state_scales import (
    FALSIFIABLE_SEMANTIC_EVENTS,
    PREDICTIVE_MECHANISM_PATHS,
    predictive_mechanism_edges,
    predictive_mechanism_paths,
    semantic_event_targets,
)


def test_semantic_event_labels_are_future_only_and_attack_oriented():
    current = np.zeros((2, OBS_DIM), dtype=np.float32)
    future = np.zeros_like(current)
    current[:, 200] = 0.5
    current[0, -1] = 1.0
    current[1, -1] = 0.0
    future[0, 200] = 0.70
    future[1, 200] = 0.30
    future[0, 209] = 1.0
    future[1, 209] = 0.0
    future[0, 204] = 0.20
    future[1, 205] = 0.20

    labels = semantic_event_targets(current, future)

    assert labels.shape == (2, len(FALSIFIABLE_SEMANTIC_EVENTS))
    assert np.all(labels == 1.0)
    with pytest.raises(ValueError, match="matching shape"):
        semantic_event_targets(current, future[:, :-1])


def test_member_aligned_mechanism_projection_is_a_coherent_joint_not_marginals():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.50
    current[-1] = 1.0
    future = np.repeat(current[None, :], 8, axis=0)
    future[:4, 209] = 1.0
    future[:4, 200] = 0.60
    future[4:, 209] = 0.0
    future[4:, 200] = 0.50

    edge = predictive_mechanism_edges(
        current, future, ensemble_trained=True,
    )["retain_possession->positive_territorial_shift"]

    assert edge["available"]
    assert edge["cell_counts"]["driver_true_outcome_true"] == 4
    assert edge["cell_counts"]["driver_false_outcome_false"] == 4
    assert sum(edge["joint_probabilities"].values()) == pytest.approx(1.0)
    assert sum(edge["independence_null_probabilities"].values()) == (
        pytest.approx(1.0)
    )
    assert edge["conditional_lift"] > 0.0
    assert edge["association_only"]
    assert not edge["causal_interpretation"]


def test_three_event_path_preserves_dependence_beyond_pairwise_markov_edges():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.50
    current[-1] = 1.0
    future = np.repeat(current[None, :], 8, axis=0)
    future[2:4, 209] = 1.0
    future[4:6, 200] = 0.60
    future[6:8, 209] = 1.0
    future[6:8, 200] = 0.70

    path = predictive_mechanism_paths(
        current, future, ensemble_trained=True,
    )[
        "retain_possession->positive_territorial_shift->enter_final_third"
    ]

    assert path["available"]
    assert sum(path["joint_probabilities"].values()) == pytest.approx(1.0)
    assert sum(path["pairwise_markov_null_probabilities"].values()) == (
        pytest.approx(1.0)
    )
    assert path["conditional_mutual_information_bits"] > 0.0
    assert path["joint_probabilities"] != (
        path["pairwise_markov_null_probabilities"]
    )
    assert path["higher_order_dependence_only"]
    assert not path["causal_interpretation"]


def test_semantic_event_bce_keeps_member_bootstrap_routing():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 2, 4), requires_grad=True)
    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    bootstrap = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss, member_losses = bootstrap_semantic_event_loss(
        logits, targets, bootstrap,
    )
    loss.backward()

    assert loss.item() == pytest.approx(np.log(2.0))
    assert member_losses.tolist() == pytest.approx([np.log(2.0)] * 2)
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[1, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 0]) == 4
    assert torch.count_nonzero(logits.grad[1, 1]) == 4


def test_autoregressive_path_nll_normalizes_and_keeps_bootstrap_routing():
    torch = pytest.importorskip("torch")
    from src.match_engine.world_model.model import build_model
    from src.match_engine.world_model.config import WorldModelConfig

    model = build_model(WorldModelConfig(
        transition_ensemble_size=2, latent_dim=16, hidden_dim=32,
    ))
    logits = torch.zeros((2, 2, 3, 7), requires_grad=True)
    targets = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    bootstrap = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss, members = bootstrap_semantic_path_loss(
        logits, targets, bootstrap,
    )
    loss.backward()
    joint = model.semantic_path_joint_probabilities(logits.detach())
    assert torch.allclose(joint.sum(dim=-1), torch.ones_like(joint[..., 0]))
    assert loss.item() == pytest.approx(np.log(2.0))
    assert members.tolist() == pytest.approx([np.log(2.0)] * 2)
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[1, 0]) == 0


def test_semantic_path_validation_compares_complete_joint_distributions():
    targets = np.asarray([
        [0, 0, 0, 0], [1, 1, 1, 1],
        [1, 1, 1, 0], [0, 0, 0, 1],
    ], dtype=np.float32)
    groups = np.asarray(["a", "a", "b", "b"])
    projection = np.full((4, 3, 8), 1.0 / 8.0)
    learned = np.repeat(projection[None, ...], 2, axis=0)
    path_indices = ((0, 2, 1), (2, 1, 3), (0, 1, 3))
    for sample, target in enumerate(targets):
        for path_index, indices in enumerate(path_indices):
            cell = int(4 * target[indices[0]] + 2 * target[indices[1]] + target[indices[2]])
            learned[:, sample, path_index, :] = 0.02
            learned[:, sample, path_index, cell] = 0.86
    report = semantic_path_validation(
        learned, targets, projection, np.full((3, 8), 1.0 / 8.0), groups,
    )
    assert len(report["paths"]) == 3
    assert all(
        row["skill_vs_best_baseline"] > 0.0
        for row in report["paths"].values()
    )
    assert all(row["group_equal_weighting"] for row in report["paths"].values())


def test_grouped_event_validation_beats_both_non_leaking_baselines():
    targets = np.asarray([
        [index % 2 for _ in FALSIFIABLE_SEMANTIC_EVENTS]
        for index in range(8)
    ], dtype=np.float32)
    learned = np.where(targets > 0.5, 0.8, 0.2)
    member_probabilities = np.stack([learned, learned], axis=0)
    projection = np.full_like(targets, 0.5)
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

    report = semantic_event_validation(
        member_probabilities,
        targets,
        projection,
        np.full(len(FALSIFIABLE_SEMANTIC_EVENTS), 0.5),
        groups,
    )

    profile = report["events"]["retain_possession"]
    assert profile["learned_brier"] == pytest.approx(0.04)
    assert profile["projection_brier"] == pytest.approx(0.25)
    assert profile["training_rate_baseline_brier"] == pytest.approx(0.25)
    assert profile["skill_vs_best_baseline"] == pytest.approx(0.84)
    assert profile["groups"] == 4
    assert profile["group_equal_weighting"]


def _gate_runtime():
    profile = {
        "samples": 200,
        "groups": 12,
        "positives": 80,
        "negatives": 120,
        "learned_brier": 0.12,
        "best_baseline_brier": 0.16,
        "skill_vs_best_baseline": 0.25,
        "ensemble_gain_vs_member_mean": 0.01,
        "disagreement_error_correlation": 0.20,
        "calibration_error": 0.05,
    }
    return SimpleNamespace(
        meta={"validation": {"semantic_event_heads": {
            "trained": True,
            "one_step_optimization_steps": 100,
            "two_step_optimization_steps": 50,
            "one_step": {"events": {"retain_possession": dict(profile)}},
            "two_step": {"events": {"retain_possession": dict(profile)}},
        }}},
        model=SimpleNamespace(semantic_event_heads_trained=True),
    )


def test_semantic_event_gate_is_per_event_and_exact_rollout_depth():
    runtime = _gate_runtime()
    gate = WorldModelRuntime.semantic_event_head_gate(
        runtime, "retain_possession", rollout_steps=1,
    )

    assert gate["active"]
    assert 0.0 < gate["authority"] <= 0.5
    assert not WorldModelRuntime.semantic_event_head_gate(
        runtime, "enter_final_third", rollout_steps=1,
    )["active"]
    unsupported = WorldModelRuntime.semantic_event_head_gate(
        runtime, "retain_possession", rollout_steps=3,
    )
    assert not unsupported["active"]
    assert unsupported["reason"] == "rollout_depth_not_validated"

    runtime.meta["validation"]["semantic_event_heads"]["one_step"][
        "events"
    ]["retain_possession"]["calibration_error"] = 0.30
    assert not WorldModelRuntime.semantic_event_head_gate(
        runtime, "retain_possession", rollout_steps=1,
    )["active"]


def _path_gate_runtime():
    path = "retain_possession->positive_territorial_shift->enter_final_third"
    profile = {
        "samples": 200,
        "groups": 12,
        "occupied_cells": 8,
        "completion_positives": 40,
        "completion_negatives": 160,
        "learned_brier": 0.50,
        "best_baseline_brier": 0.65,
        "skill_vs_best_baseline": 1.0 - 0.50 / 0.65,
        "ensemble_gain_vs_member_mean": 0.01,
    }
    return SimpleNamespace(
        meta={"validation": {"semantic_path_heads": {
            "trained": True,
            "one_step_optimization_steps": 100,
            "two_step_optimization_steps": 50,
            "one_step": {"paths": {path: dict(profile)}},
            "two_step": {"paths": {path: dict(profile)}},
        }}},
        model=SimpleNamespace(semantic_path_heads_trained=True),
    ), path


def test_semantic_path_gate_is_per_path_and_exact_rollout_depth():
    runtime, path = _path_gate_runtime()
    gate = WorldModelRuntime.semantic_path_head_gate(
        runtime, path, rollout_steps=1,
    )

    assert gate["active"]
    assert 0.0 < gate["authority"] <= 0.5
    assert not WorldModelRuntime.semantic_path_head_gate(
        runtime, "unknown->mechanism->path", rollout_steps=1,
    )["active"]
    unsupported = WorldModelRuntime.semantic_path_head_gate(
        runtime, path, rollout_steps=3,
    )
    assert not unsupported["active"]
    assert unsupported["reason"] == "rollout_depth_not_validated"

    runtime.meta["validation"]["semantic_path_heads"]["one_step"][
        "paths"
    ][path]["occupied_cells"] = 3
    assert not WorldModelRuntime.semantic_path_head_gate(
        runtime, path, rollout_steps=1,
    )["active"]


def test_trainer_persists_real_one_and_two_step_event_evidence(
    tmp_path, monkeypatch,
):
    pytest.importorskip("torch")
    from src.match_engine.world_model.action_codec import ACTION_DIM
    from src.match_engine.world_model import recorder
    from src.match_engine.world_model.model import load_checkpoint
    from src.match_engine.world_model.schema import PASS_OUTCOME_INDEX
    from scripts import train_world_model

    rng = np.random.default_rng(123)
    observations, actions, futures, groups = [], [], [], []
    for group_index in range(10):
        states = rng.uniform(0.05, 0.95, size=(41, OBS_DIM)).astype(np.float32)
        states[:, -1] = float(group_index % 2 == 0)
        states[:, 209] = rng.integers(0, 2, size=41)
        for step in range(40):
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[0] = 1.0
            action[13] = 0.65
            action[PASS_OUTCOME_INDEX] = float((step + group_index) % 2)
            observations.append(states[step])
            actions.append(action)
            futures.append(states[step + 1])
            groups.append(f"match_{group_index}")
    arrays = (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(futures, dtype=np.float32),
        np.asarray(groups),
    )

    monkeypatch.setattr(
        recorder, "load_trace_batches", lambda *args, **kwargs: arrays,
    )
    monkeypatch.setenv("MATCH_WM_LATENT_DIM", "16")
    monkeypatch.setenv("MATCH_WM_HIDDEN_DIM", "32")
    monkeypatch.setenv("MATCH_WM_ENSEMBLE_SIZE", "2")
    monkeypatch.setenv("MATCH_WM_TRANSITION_ENSEMBLE_SIZE", "2")
    checkpoint = tmp_path / "semantic-events.pt"
    monkeypatch.setattr("sys.argv", [
        "train_world_model.py",
        "--epochs", "1",
        "--batch-size", "64",
        "--trace-dir", str(tmp_path / "unused"),
        "--out", str(checkpoint),
        "--multi-step-warmup-fraction", "0",
    ])

    train_world_model.main()
    model, _, meta = load_checkpoint(str(checkpoint))
    contract = meta["validation"]["semantic_event_heads"]
    path_contract = meta["validation"]["semantic_path_heads"]

    assert model.checkpoint_version == 9
    assert model.semantic_event_heads_trained
    assert contract["trained"]
    assert contract["one_step_optimization_steps"] > 0
    assert contract["two_step_optimization_steps"] > 0
    assert set(contract["one_step"]["events"]) == set(
        FALSIFIABLE_SEMANTIC_EVENTS
    )
    assert set(contract["two_step"]["events"]) == set(
        FALSIFIABLE_SEMANTIC_EVENTS
    )
    assert model.semantic_path_heads_trained
    assert path_contract["trained"]
    assert path_contract["one_step_optimization_steps"] > 0
    assert path_contract["two_step_optimization_steps"] > 0
    expected_paths = {"->".join(path) for path in PREDICTIVE_MECHANISM_PATHS}
    assert set(path_contract["one_step"]["paths"]) == expected_paths
    assert set(path_contract["two_step"]["paths"]) == expected_paths
