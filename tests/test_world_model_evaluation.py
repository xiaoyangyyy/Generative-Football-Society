"""Tests for the reusable world-model capability baseline."""

import numpy as np
import pytest
import json

from src.data_engine.dataset_registry import file_sha256, verify_trace_manifest
from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.evaluation import (
    TransitionMetricAccumulator,
    binary_auc,
    expected_calibration_error,
    validate_transition_batch,
)
from src.match_engine.world_model.observation import OBS_DIM


def test_transition_contract_accepts_only_canonical_finite_batches():
    obs = np.zeros((2, OBS_DIM), dtype=np.float32)
    actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
    next_obs = np.ones((2, OBS_DIM), dtype=np.float32)
    validated = validate_transition_batch(obs, actions, next_obs)
    assert all(values.dtype == np.float32 for values in validated)

    with pytest.raises(ValueError, match="actions must have shape"):
        validate_transition_batch(obs, actions[:, :-1], next_obs)
    next_obs[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        validate_transition_batch(obs, actions, next_obs)


def test_transition_metrics_compare_against_persistence_by_feature_group():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    target = np.full(OBS_DIM, 0.5, dtype=np.float32)
    prediction = np.full(OBS_DIM, 0.25, dtype=np.float32)
    metrics = TransitionMetricAccumulator()
    metrics.update(prediction, target, current, uncertainty=0.2)
    report = metrics.finalize()
    assert report["samples"] == 1
    assert report["skill_vs_persistence"] == pytest.approx(0.75)
    assert set(report["feature_mse"]) == {
        "grid", "ball", "match_context", "players", "tactics", "attack_flag",
    }


def test_uncertainty_diagnostics_identify_high_error_predictions():
    metrics = TransitionMetricAccumulator()
    current = np.zeros(OBS_DIM, dtype=np.float32)
    target = np.ones(OBS_DIM, dtype=np.float32)
    for error_scale, uncertainty in ((0.1, 0.1), (0.2, 0.2), (0.8, 0.9), (0.9, 1.0)):
        prediction = target - error_scale
        metrics.update(prediction, target, current, uncertainty=uncertainty)
    diagnostics = metrics.finalize()["uncertainty"]
    assert diagnostics["error_correlation"] > 0.9
    assert diagnostics["high_uncertainty_error_lift"] > 1.0


def test_binary_metrics_are_calibrated_and_ranked_for_perfect_forecast():
    probability = np.array([0.05, 0.15, 0.85, 0.95])
    truth = np.array([False, False, True, True])
    assert binary_auc(probability, truth) == 1.0
    assert expected_calibration_error(probability, truth, bins=4) < 0.11


def test_frozen_trace_manifest_detects_dataset_mutation(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    row = {
        "obs": np.zeros(OBS_DIM).tolist(),
        "action": np.zeros(ACTION_DIM).tolist(),
        "next_obs": np.ones(OBS_DIM).tolist(),
    }
    trace = trace_dir / "match.jsonl"
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "source_root": str(trace_dir),
        "files": [{
            "path": trace.name, "group": "match", "split": "train",
            "sha256": file_sha256(trace), "bytes": trace.stat().st_size,
            "rows": 1, "passes": 0, "shots": 0, "goals": 0,
        }],
    }
    assert verify_trace_manifest(manifest) == trace_dir.resolve()

    trace.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_trace_manifest(manifest)
