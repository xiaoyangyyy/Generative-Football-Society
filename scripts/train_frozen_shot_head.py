#!/usr/bin/env python3
"""Train only the shot head on a frozen world-model latent representation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import (
    file_sha256,
    files_for_split,
    load_manifest,
    verify_trace_manifest,
    write_json_atomic,
)
from src.match_engine.world_model.recorder import load_trace_batches
from src.match_engine.world_model.schema import SHOT_GOAL_INDEX, strip_outcome_leakage
from src.match_engine.world_model.shot_head import FrozenShotHead


def _fit_logistic(features: np.ndarray, truth: np.ndarray, *, steps: int = 1200):
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    values = (features - mean) / scale
    weights = np.zeros(values.shape[1], dtype=float)
    prevalence = float(np.clip(truth.mean(), 1e-4, 1.0 - 1e-4))
    bias = float(np.log(prevalence / (1.0 - prevalence)))
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-np.clip(values @ weights + bias, -30, 30)))
        error = probability - truth
        weights -= 0.03 * (values.T @ error / len(values) + 0.002 * weights)
        bias -= 0.03 * float(error.mean())
    return mean, scale, weights, bias


def _fit_platt(logits: np.ndarray, truth: np.ndarray, *, steps: int = 500):
    scale, bias = 1.0, 0.0
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-np.clip(scale * logits + bias, -30, 30)))
        error = probability - truth
        scale -= 0.02 * float(np.mean(error * logits))
        bias -= 0.02 * float(error.mean())
        scale = float(np.clip(scale, 0.05, 10.0))
    return scale, bias


def _shot_features(runtime, trace_dir: str, manifest: dict, split: str):
    import torch

    obs, actions, _, _ = load_trace_batches(
        trace_dir, return_groups=True,
        allowed_files=files_for_split(manifest, {split}),
    )
    mask = actions[:, 1] > 0.5
    if not mask.any():
        raise ValueError(f"no shot samples in {split}")
    raw_actions = actions[mask]
    clean_actions = strip_outcome_leakage(raw_actions)
    with torch.no_grad():
        latent = runtime.model.encode(torch.from_numpy(obs[mask])).numpy()
    return (
        np.concatenate([latent, clean_actions], axis=1).astype(float),
        raw_actions[:, SHOT_GOAL_INDEX].astype(float),
        raw_actions[:, 13].astype(float),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trace-dir", default=str(ROOT / "data/world_model/traces"))
    parser.add_argument(
        "--out", default=str(ROOT / "data/world_model/frozen_shot_head_candidate.json"),
    )
    args = parser.parse_args()
    from src.match_engine.world_model.inference import WorldModelRuntime

    manifest = load_manifest(args.manifest)
    verify_trace_manifest(manifest, base_dir=ROOT)
    runtime = WorldModelRuntime.load(args.checkpoint)
    train_x, train_y, _ = _shot_features(runtime, args.trace_dir, manifest, "train")
    dev_x, dev_y, dev_prior = _shot_features(runtime, args.trace_dir, manifest, "dev")
    mean, scale, weights, bias = _fit_logistic(train_x, train_y)
    dev_logits = ((dev_x - mean) / scale) @ weights + bias
    cal_scale, cal_bias = _fit_platt(dev_logits, dev_y)
    dev_probability = 1.0 / (1.0 + np.exp(-np.clip(
        cal_scale * dev_logits + cal_bias, -30, 30,
    )))
    artifact = FrozenShotHead(
        base_checkpoint_sha256=runtime.checkpoint_signature,
        latent_dim=runtime.cfg.latent_dim,
        mean=mean, scale=scale, weights=weights, bias=bias,
        calibration_scale=cal_scale, calibration_bias=cal_bias,
        evidence={
            "training": {
                "samples": int(len(train_y)), "goals": int(train_y.sum()),
                "backbone_frozen": True, "sealed_test_used": False,
                "manifest_sha256": file_sha256(Path(args.manifest)),
            },
            "development": {
                "samples": int(len(dev_y)), "goals": int(dev_y.sum()),
                "model_brier": float(np.mean((dev_probability - dev_y) ** 2)),
                "physics_xg_prior_brier": float(np.mean((dev_prior - dev_y) ** 2)),
                "calibration_fit_split": "dev",
            },
        },
        accepted=False,
    )
    write_json_atomic(args.out, artifact.to_dict())
    print(artifact.to_dict()["evidence"])
    print(f"Candidate only; sealed validation required: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
