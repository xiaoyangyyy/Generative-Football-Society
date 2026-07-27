from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.match_engine.frame_world.model import FrameWorldConfig, GraphTemporalFrameWorldModel, load_frame_world
from src.match_engine.frame_world.schema import ENTITY_COUNT, FrameSequence, constant_velocity_baseline


ROOT = Path(__file__).resolve().parents[1]


def test_frame_sequence_rejects_nonfinite_hidden_values():
    positions = np.zeros((2, ENTITY_COUNT, 2), dtype=np.float32)
    positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        FrameSequence(
            "test",
            "match",
            np.array([0.0, 0.1]),
            positions,
            np.zeros_like(positions),
            np.zeros((2, ENTITY_COUNT), dtype=bool),
            np.r_[np.zeros(16), np.ones(16), -1],
            np.full(2, -1),
        )


def test_constant_velocity_baseline_is_bounded():
    result = constant_velocity_baseline(np.array([[1.0, 0.0]]), np.array([[2.0, -2.0]]), 1.0)
    np.testing.assert_allclose(result, [[1.15, -0.15]])


def test_frame_model_outputs_are_finite_and_checkpoint_roundtrips(tmp_path):
    cfg = FrameWorldConfig(hidden_dim=16, graph_heads=4, graph_layers=1)
    model = GraphTemporalFrameWorldModel(cfg).eval()
    shape = (2, cfg.history, ENTITY_COUNT)
    positions = torch.rand(*shape, 2)
    velocities = torch.zeros_like(positions)
    visible = torch.ones(shape, dtype=torch.bool)
    teams = torch.tensor([0] * 16 + [1] * 16 + [-1]).repeat(2, 1)
    mean, logvar, possession = model(positions, velocities, visible, teams)
    assert mean.shape == logvar.shape == (2, len(cfg.horizons), ENTITY_COUNT, 2)
    assert possession.shape == (2, 3)
    assert torch.isfinite(mean).all() and torch.isfinite(logvar).all()
    model.variance_log_offset.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    path = tmp_path / "frame.pt"
    torch.save(model.checkpoint(), path)
    loaded, loaded_cfg, _ = load_frame_world(path)
    assert loaded_cfg == cfg
    torch.testing.assert_close(loaded.variance_log_offset, model.variance_log_offset)


def test_frame_manifest_covers_three_providers_and_valid_hashes():
    manifest = json.loads((ROOT / "data/frame_world/v8/manifest.json").read_text())
    assert manifest["providers"] == ["metrica", "skillcorner", "sportec"]
    assert len(manifest["matches"]) == 19
    assert manifest["total_frames"] == sum(item["frames"] for item in manifest["matches"])
    for item in manifest["matches"]:
        path = ROOT / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
