#!/usr/bin/env python3
"""Compare completion signals on the immutable world-model splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import files_for_split, load_manifest
from src.match_engine.world_model.recorder import load_trace_batches
from src.match_engine.world_model.schema import PASS_OUTCOME_INDEX


def _auc(prob: np.ndarray, truth: np.ndarray) -> float:
    positive, negative = prob[truth], prob[~truth]
    if not len(positive) or not len(negative):
        return 0.5
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def _best_ba(prob: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    best = (0.5, 0.5)
    for threshold in np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 199))):
        pred = prob >= threshold
        tpr = float(pred[truth].mean()) if truth.any() else 0.5
        tnr = float((~pred[~truth]).mean()) if (~truth).any() else 0.5
        score = 0.5 * (tpr + tnr)
        if score > best[0]:
            best = score, float(threshold)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data/world_model/dataset_manifest.json"))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    report = {}
    for split in ("train", "dev", "sealed_test"):
        allowed = files_for_split(manifest, {split})
        _obs, actions, _next, _groups = load_trace_batches(
            manifest["source_root"], allowed_files=allowed, return_groups=True
        )
        mask = (actions[:, 0] > 0.5) | (actions[:, 4] > 0.5)
        actions = actions[mask]
        truth = actions[:, PASS_OUTCOME_INDEX] > 0.5
        raw_prior = actions[:, 13]
        nonfinite = int((~np.isfinite(raw_prior)).sum())
        prior = np.clip(np.nan_to_num(raw_prior, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        ba, threshold = _best_ba(prior, truth)
        report[split] = {
            "samples": len(prior),
            "success_rate": float(truth.mean()),
            "prior_mean": float(prior.mean()),
            "prior_auc": _auc(prior, truth),
            "prior_best_ba": ba,
            "prior_best_threshold": threshold,
            "prior_brier": float(np.mean((prior - truth) ** 2)),
            "prior_std": float(prior.std()),
            "prior_nonfinite": nonfinite,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
