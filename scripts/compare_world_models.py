#!/usr/bin/env python3
"""Compare checkpoints on the same grouped holdout without retraining."""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--trace-dir", default="data/world_model/traces")
    args = parser.parse_args()

    import torch

    from src.match_engine.world_model.model import load_checkpoint
    from src.match_engine.world_model.recorder import load_trace_batches
    from src.match_engine.world_model.schema import (
        PASS_OUTCOME_INDEX, SHOT_GOAL_INDEX, observation_loss_weights, strip_outcome_leakage,
    )

    obs, actions, nxt, groups = load_trace_batches(args.trace_dir, return_groups=True)
    valid = (np.std(obs, axis=1) > 0.02) & (np.std(nxt, axis=1) > 0.02)
    obs, actions, nxt, groups = obs[valid], actions[valid], nxt[valid], groups[valid]
    groups = np.asarray([
        re.sub(r"_wm_collect$", "", re.sub(r"_trace_\d+$", "", str(value)))
        for value in groups
    ])
    rng = np.random.default_rng(42)
    unique = rng.permutation(np.unique(groups))
    val_groups = set(unique[: max(1, int(0.20 * len(unique)))].tolist())
    idx = np.flatnonzero(np.isin(groups, list(val_groups)))
    original_actions = actions[idx].copy()
    model_actions = strip_outcome_leakage(original_actions)
    is_pass = original_actions[:, 0] > 0.5
    is_shot = original_actions[:, 1] > 0.5
    weights = torch.from_numpy(observation_loss_weights())

    for path in args.checkpoints:
        model, _, meta = load_checkpoint(path)
        with torch.no_grad():
            _, pred, pass_logit, _, shot_logit, _ = model(
                torch.from_numpy(obs[idx]), torch.from_numpy(model_actions), h=None
            )
        mse = float(((((pred - torch.from_numpy(nxt[idx])) ** 2) * weights).mean()).item())
        pass_prob = torch.sigmoid(pass_logit.view(-1)).numpy()[is_pass]
        pass_true = original_actions[is_pass, PASS_OUTCOME_INDEX] > 0.5
        pass_pred = pass_prob >= 0.5
        tpr = float(pass_pred[pass_true].mean()) if pass_true.any() else 0.5
        tnr = float((~pass_pred[~pass_true]).mean()) if (~pass_true).any() else 0.5
        shot_prob = torch.sigmoid(shot_logit.view(-1)).numpy()[is_shot]
        shot_true = original_actions[is_shot, SHOT_GOAL_INDEX]
        print({
            "path": path,
            "version": model.checkpoint_version,
            "weighted_obs_mse": mse,
            "pass_balanced_accuracy": 0.5 * (tpr + tnr),
            "shot_brier": float(np.mean((shot_prob - shot_true) ** 2)),
            "rows": len(idx),
            "pass_samples": int(is_pass.sum()),
            "shot_samples": int(is_shot.sum()),
            "training_validation": meta.get("validation", {}),
        })


if __name__ == "__main__":
    main()
