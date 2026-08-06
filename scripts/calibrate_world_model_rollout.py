#!/usr/bin/env python3
"""Fit dev-only rollout shrinkage and refresh dependent validation evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_world_model import _match_group, _sequential_transition_pairs
from src.data_engine.dataset_registry import files_for_split, load_manifest
from src.match_engine.world_model.model import load_checkpoint, save_checkpoint
from src.match_engine.world_model.recorder import load_trace_batches
from src.match_engine.world_model.rollout_calibration import select_rollout_blend
from src.match_engine.world_model.schema import observation_loss_weights, strip_outcome_leakage
from src.match_engine.world_model.semantic_event_training import (
    semantic_event_validation,
    semantic_path_cell_rates,
    semantic_path_validation,
)
from src.match_engine.world_model.state_scales import (
    projected_semantic_event_probabilities,
    projected_semantic_path_probabilities,
    semantic_event_targets,
)


def _indices(groups: np.ndarray, manifest: dict, split: str) -> np.ndarray:
    names = files_for_split(manifest, {split})
    wanted = {_match_group(Path(name).stem) for name in names}
    return np.flatnonzero(np.isin(groups, list(wanted)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trace-dir", default="data/world_model/traces")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--blend", type=float,
        help="Optional dev-selected blend when a joint semantic constraint is applied.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    obs, raw_actions, nxt, raw_groups = load_trace_batches(
        args.trace_dir, return_groups=True,
    )
    groups = np.asarray([_match_group(value) for value in raw_groups], dtype=str)
    actions = strip_outcome_leakage(raw_actions)
    train_idx = _indices(groups, manifest, "train")
    dev_idx = _indices(groups, manifest, "dev")
    train_left = _sequential_transition_pairs(obs, nxt, groups, train_idx)
    dev_left = _sequential_transition_pairs(obs, nxt, groups, dev_idx)
    if len(dev_left) < 32 or len(np.unique(groups[dev_left])) < 4:
        raise SystemExit("grouped development rollout set is too small")

    model, cfg, meta = load_checkpoint(args.checkpoint)
    model.eval()
    cfg.rollout_residual_blend = 1.0
    model.cfg.rollout_residual_blend = 1.0
    dev_actions = np.stack([actions[dev_left], actions[dev_left + 1]], axis=1)
    with torch.no_grad():
        raw_members = model.transition_rollout_predictions(
            torch.from_numpy(obs[dev_left]), torch.from_numpy(dev_actions),
        )
    calibration = select_rollout_blend(
        obs[dev_left], raw_members.mean(dim=0).numpy(), nxt[dev_left + 1],
        observation_loss_weights(),
    )
    if args.blend is not None:
        selected = min(
            calibration["candidates"],
            key=lambda row: abs(float(row["blend"]) - float(args.blend)),
        )
        calibration.update({
            "method": "bounded_residual_grid_on_grouped_dev_with_semantic_constraint",
            "selected_blend": float(selected["blend"]),
            "weighted_mse": float(selected["weighted_mse"]),
            "skill_vs_persistence": float(selected["skill_vs_persistence"]),
            "selection_constraint": "at_least_two_two_step_semantic_events_above_gate",
        })
    cfg.rollout_residual_blend = calibration["selected_blend"]
    model.cfg.rollout_residual_blend = calibration["selected_blend"]
    with torch.no_grad():
        members = model.transition_rollout_predictions(
            torch.from_numpy(obs[dev_left]), torch.from_numpy(dev_actions),
        )
    target = torch.from_numpy(nxt[dev_left + 1])
    initial = torch.from_numpy(obs[dev_left])
    weights = torch.from_numpy(observation_loss_weights())
    mean_prediction = members.mean(dim=0)
    mse = float((((mean_prediction - target) ** 2) * weights).mean())
    persistence = float((((initial - target) ** 2) * weights).mean())
    member_mean = float(torch.stack([
        (((member - target) ** 2) * weights).mean() for member in members
    ]).mean())
    disagreement = torch.sqrt(torch.mean(
        torch.var(members, dim=0, unbiased=False), dim=1,
    )).numpy()
    error = torch.sqrt(torch.mean((mean_prediction - target) ** 2, dim=1)).numpy()
    correlation = float(np.corrcoef(disagreement, error)[0, 1]) if (
        np.std(disagreement) > 1e-12 and np.std(error) > 1e-12
    ) else 0.0

    train_pair_targets = semantic_event_targets(obs[train_left], nxt[train_left + 1])
    dev_pair_targets = semantic_event_targets(obs[dev_left], nxt[dev_left + 1])
    event_base = (train_pair_targets.sum(axis=0) + 0.5) / (len(train_pair_targets) + 1.0)
    path_base = semantic_path_cell_rates(train_pair_targets)
    action_context = torch.from_numpy(dev_actions).mean(dim=1)
    with torch.no_grad():
        event_members = torch.sigmoid(model.semantic_event_logits(
            torch.from_numpy(obs[dev_left]), members, action_context,
        )).numpy()
        path_members = model.semantic_path_joint_probabilities(
            model.semantic_path_logits(
                torch.from_numpy(obs[dev_left]), members, action_context,
            )
        ).numpy()
    projection_events = projected_semantic_event_probabilities(
        obs[dev_left], members.numpy(), ensemble_trained=True,
    )
    projection_paths = projected_semantic_path_probabilities(
        obs[dev_left], members.numpy(), ensemble_trained=True,
    )

    validation = meta.setdefault("validation", {})
    rollout = validation.setdefault("two_step_rollout", {})
    rollout.update({
        "samples": int(len(dev_left)),
        "groups": int(len(np.unique(groups[dev_left]))),
        "weighted_mse": mse,
        "persistence_weighted_mse": persistence,
        "skill_vs_persistence": 1.0 - mse / max(persistence, 1e-12),
        "member_mean_weighted_mse": member_mean,
        "ensemble_gain_vs_member_mean": member_mean - mse,
        "disagreement_error_correlation": correlation,
        "residual_calibration": calibration,
    })
    validation["semantic_event_heads"]["two_step"] = semantic_event_validation(
        event_members, dev_pair_targets, projection_events, event_base, groups[dev_left],
    )
    validation["semantic_path_heads"]["two_step"] = semantic_path_validation(
        path_members, dev_pair_targets, projection_paths, path_base, groups[dev_left],
    )
    meta["rollout_calibration"] = {
        "split": "dev",
        "manifest": args.manifest,
        **calibration,
    }
    save_checkpoint(args.out, model, cfg, meta=meta)
    print(json.dumps({
        "output": args.out,
        "selected_blend": calibration["selected_blend"],
        "dev_skill_vs_persistence": rollout["skill_vs_persistence"],
        "dev_semantic_active": {
            depth: [
                name for name, row in validation["semantic_event_heads"][depth]["events"].items()
                if row["skill_vs_best_baseline"] > 0
            ] for depth in ("one_step", "two_step")
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
