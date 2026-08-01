#!/usr/bin/env python3
"""Train latent GRU/Transformer world model from traces + ball_log backfill."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _merge(*arrays):
    parts = [a for a in arrays if len(a) > 0]
    if not parts:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        )
    return np.concatenate(parts, axis=0)


def _match_group(name: str) -> str:
    value = re.sub(r"_trace_\d+$", "", str(name))
    return re.sub(r"_wm_collect$", "", value)


def _bootstrap_transition_loss(
    predictions,
    target,
    feature_weights,
    bootstrap_weights,
):
    """Member-specific Bayesian-bootstrap transition objective."""
    if predictions.ndim != 3 or bootstrap_weights.shape != predictions.shape[:2]:
        raise ValueError("invalid transition bootstrap shapes")
    per_member_sample_error = (
        ((predictions - target.unsqueeze(0)) ** 2) * feature_weights
    ).mean(dim=2)
    member_losses = (
        (per_member_sample_error * bootstrap_weights).sum(dim=1)
        / bootstrap_weights.sum(dim=1).clamp_min(1.0)
    )
    return member_losses.mean(), member_losses


def _sequential_holdout_pairs(
    obs: np.ndarray,
    nxt: np.ndarray,
    groups: np.ndarray,
    validation_indices: np.ndarray,
    *,
    max_alignment_mae: float = 0.03,
) -> np.ndarray:
    """Find adjacent, state-aligned transitions inside held-out match groups."""
    membership = np.zeros(len(obs), dtype=bool)
    membership[np.asarray(validation_indices, dtype=int)] = True
    return np.asarray([
        index for index in range(len(obs) - 1)
        if membership[index]
        and membership[index + 1]
        and groups[index] == groups[index + 1]
        and float(np.mean(np.abs(nxt[index] - obs[index + 1])))
        <= float(max_alignment_mae)
    ], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trace-dir", type=str, default="")
    parser.add_argument("--ball-log-dir", type=str, default="")
    parser.add_argument("--use-ball-log", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--transition", type=str, default="gru", choices=("gru", "transformer"))
    parser.add_argument(
        "--transition-ensemble-size",
        type=int,
        default=0,
        help="Independent dynamics members; 0 uses MATCH_WM_TRANSITION_ENSEMBLE_SIZE.",
    )
    parser.add_argument("--dataset-manifest", type=str, default="")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch required: pip install torch") from exc

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["MATCH_WM_TRANSITION"] = args.transition

    from src.match_engine.world_model.config import WorldModelConfig, default_checkpoint_path, default_trace_dir
    from src.match_engine.world_model.model import build_model, save_checkpoint
    from src.match_engine.world_model.recorder import load_trace_batches
    from src.match_engine.world_model.ball_log_dataset import load_ball_log_dataset
    from src.match_engine.world_model.schema import (
        PASS_OUTCOME_INDEX,
        SHOT_GOAL_INDEX,
        observation_loss_weights,
        strip_outcome_leakage,
    )

    trace_dir = args.trace_dir or default_trace_dir(base_dir)
    ball_dir = args.ball_log_dir or os.path.join(base_dir, "outputs", "ball_log")
    out_path = args.out or default_checkpoint_path(base_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    allowed_files = None
    manifest = None
    if args.dataset_manifest:
        from src.data_engine.dataset_registry import files_for_split, load_manifest
        manifest = load_manifest(args.dataset_manifest)
        allowed_files = files_for_split(manifest, {"train", "dev"})
    obs_t, act_t, nxt_t, groups_t = load_trace_batches(
        trace_dir, return_groups=True, allowed_files=allowed_files
    )
    if args.use_ball_log:
        obs_b, act_b, nxt_b, groups_b = load_ball_log_dataset(ball_dir, return_groups=True)
        obs, act, nxt = _merge(obs_t, obs_b), _merge(act_t, act_b), _merge(nxt_t, nxt_b)
        groups = np.asarray(
            [_match_group(v) for v in np.concatenate([groups_t, groups_b])],
            dtype=str,
        )
        supervised = np.concatenate(
            [np.zeros(len(obs_t), dtype=bool), np.ones(len(obs_b), dtype=bool)]
        )
    else:
        obs, act, nxt = obs_t, act_t, nxt_t
        groups = np.asarray([_match_group(v) for v in groups_t], dtype=str)
        supervised = np.zeros(len(obs_t), dtype=bool)

    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
    nxt = np.nan_to_num(nxt, nan=0.0, posinf=1.0, neginf=0.0)
    act[:, 13] = np.nan_to_num(act[:, 13], nan=0.5, posinf=1.0, neginf=0.0)
    act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=0.0)
    obs = np.clip(obs, 0.0, 1.0)
    nxt = np.clip(nxt, 0.0, 1.0)
    valid = (np.std(obs, axis=1) > 0.02) & (np.std(nxt, axis=1) > 0.02)
    obs, act, nxt, supervised, groups = (
        obs[valid],
        act[valid],
        nxt[valid],
        supervised[valid],
        groups[valid],
    )

    if len(obs) < 128:
        raise SystemExit(
            f"Not enough training data ({len(obs)} rows). "
            "Run: python scripts/collect_world_model_traces.py --pairs 32"
        )

    is_pass = (act[:, 0] > 0.5) | (act[:, 4] > 0.5)
    is_shot = act[:, 1] > 0.5
    supervised = supervised | is_pass | is_shot
    pass_mask = (supervised & is_pass).astype(np.float32)
    shot_mask = (supervised & is_shot).astype(np.float32)
    pass_success = np.clip(act[:, PASS_OUTCOME_INDEX], 0, 1).astype(np.float32)
    shot_goal = np.clip(act[:, SHOT_GOAL_INDEX], 0, 1).astype(np.float32)
    attacking_home = obs[:, -1] > 0.5
    progress_before = np.where(attacking_home, obs[:, 200], 1.0 - obs[:, 200])
    progress_after = np.where(attacking_home, nxt[:, 200], 1.0 - nxt[:, 200])
    xg_delta = np.clip(progress_after - progress_before, -1.0, 1.0).astype(np.float32)
    act = strip_outcome_leakage(act)

    cfg = WorldModelConfig.from_env()
    cfg.transition_type = args.transition
    if args.transition_ensemble_size > 0:
        cfg.transition_ensemble_size = args.transition_ensemble_size
        cfg.__post_init__()
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=min(args.lr, 3e-4))
    obs_weights = torch.from_numpy(observation_loss_weights())
    rng = np.random.default_rng(42)
    unique_groups = rng.permutation(np.unique(groups))
    if manifest is not None:
        dev_files = files_for_split(manifest, {"dev"})
        val_groups = {_match_group(Path(name).stem) for name in dev_files}
    else:
        val_group_count = max(1, int(0.20 * len(unique_groups)))
        val_groups = set(unique_groups[:val_group_count].tolist())
    val_idx = np.flatnonzero(np.isin(groups, list(val_groups)))
    train_idx = np.flatnonzero(~np.isin(groups, list(val_groups)))
    if len(val_idx) < 32 or len(train_idx) < 128:
        raise SystemExit("Grouped holdout is too small; collect more independent matches.")

    train_shots = shot_goal[train_idx][shot_mask[train_idx] > 0.5]
    train_passes = pass_success[train_idx][pass_mask[train_idx] > 0.5]
    pass_pos_rate = float(train_passes.mean()) if len(train_passes) else 0.5
    shot_pos = float(train_shots.sum())
    shot_neg = float(len(train_shots) - shot_pos)
    # Goal probability is consumed directly as a calibrated probability. Keep
    # the proper scoring objective unweighted; class weights alter its meaning.
    shot_pos_weight = 1.0
    pass_bce = nn.BCEWithLogitsLoss(reduction="none")
    shot_bce = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(shot_pos_weight, dtype=torch.float32),
    )

    ds = TensorDataset(
        torch.from_numpy(obs[train_idx]),
        torch.from_numpy(act[train_idx]),
        torch.from_numpy(nxt[train_idx]),
        torch.from_numpy(pass_success[train_idx]),
        torch.from_numpy(shot_goal[train_idx]),
        torch.from_numpy(xg_delta[train_idx]),
        torch.from_numpy(pass_mask[train_idx]),
        torch.from_numpy(shot_mask[train_idx]),
    )
    loader_generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        drop_last=len(obs) > args.batch_size, generator=loader_generator,
    )

    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        n = 0
        for batch in loader:
            o, a, no, ps, sg, xg, pm, sm = batch
            z_members, pred_obs_members = model.transition_predictions(o, a)
            transition_bootstrap = torch.stack([
                torch.empty_like(pm).exponential_().clamp_max(4.0)
                for _ in range(model.transition_member_count)
            ], dim=0)
            obs_loss, _transition_losses = _bootstrap_transition_loss(
                pred_obs_members,
                no,
                obs_weights,
                transition_bootstrap,
            )
            outcome_members = [
                model.outcome_ensemble(member, a) for member in z_members
            ]
            pass_all = torch.stack([
                item[0] for item in outcome_members
            ], dim=0)
            progress_all = torch.stack([
                item[1] for item in outcome_members
            ], dim=0)
            shot_all = torch.stack([
                item[2] for item in outcome_members
            ], dim=0)
            pass_losses, shot_losses, progress_losses = [], [], []
            for head_idx in range(pass_all.shape[1]):
                outcome_bootstrap = (
                    torch.empty_like(pm).exponential_().clamp_max(4.0)
                )
                # v7 learns a bounded residual around the calibrated physics
                # prior. Proper BCE preserves probability meaning; class
                # weighting would move the optimum away from that prior.
                for member_index in range(model.transition_member_count):
                    joint_bootstrap = (
                        outcome_bootstrap
                        * transition_bootstrap[member_index]
                    )
                    pass_w = pm * joint_bootstrap
                    shot_w = sm * joint_bootstrap
                    pass_raw = pass_bce(
                        pass_all[member_index, head_idx].view(-1), ps,
                    )
                    shot_raw = shot_bce(
                        shot_all[member_index, head_idx].view(-1), sg,
                    )
                    pass_losses.append(
                        (pass_raw * pass_w).sum()
                        / pass_w.sum().clamp_min(1.0)
                    )
                    shot_losses.append(
                        (shot_raw * shot_w).sum()
                        / shot_w.sum().clamp_min(1.0)
                    )
                    progress_losses.append(
                        (((progress_all[
                            member_index, head_idx,
                        ].view(-1) - xg) ** 2) * joint_bootstrap).sum()
                        / joint_bootstrap.sum().clamp_min(1.0)
                    )
            pass_loss = torch.stack(pass_losses).mean()
            shot_loss = torch.stack(shot_losses).mean()
            progress_loss = torch.stack(progress_losses).mean()
            loss = (
                obs_loss
                + 0.35 * pass_loss
                + 0.30 * shot_loss
                + 0.20 * progress_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        print(
            f"epoch {epoch+1}/{args.epochs} loss={loss_sum/max(1,n):.4f} "
            f"n={len(train_idx)} val={len(val_idx)} transition={args.transition} "
            f"members={model.transition_member_count}"
        )

    model.eval()
    with torch.no_grad():
        vo = torch.from_numpy(obs[val_idx])
        va = torch.from_numpy(act[val_idx])
        vn = torch.from_numpy(nxt[val_idx])
        vz_members, vp_members = model.transition_predictions(vo, va)
        vp = vp_members.mean(dim=0)
        validation_outcomes = [
            model.outcome_ensemble(member, va) for member in vz_members
        ]
        vpass_all = torch.stack([
            item[0] for item in validation_outcomes
        ], dim=0)
        vxg_all = torch.stack([
            item[1] for item in validation_outcomes
        ], dim=0)
        vshot_all = torch.stack([
            item[2] for item in validation_outcomes
        ], dim=0)
        vxp = vxg_all.mean(dim=(0, 1))
        weighted_mse = float(((((vp - vn) ** 2) * obs_weights).mean()).item())
        transition_primary_mse = float(
            ((((vp_members[0] - vn) ** 2) * obs_weights).mean()).item()
        )
        transition_member_mean_mse = float(torch.stack([
            (((member - vn) ** 2) * obs_weights).mean()
            for member in vp_members
        ]).mean().item())
        transition_disagreement = torch.sqrt(torch.mean(
            torch.var(vp_members, dim=0, unbiased=False), dim=1,
        )).numpy()
        transition_error = torch.sqrt(torch.mean(
            (vp - vn) ** 2, dim=1,
        )).numpy()
        transition_disagreement_error_correlation = (
            float(np.corrcoef(
                transition_disagreement, transition_error,
            )[0, 1])
            if np.std(transition_disagreement) > 1e-12
            and np.std(transition_error) > 1e-12
            else 0.0
        )
        pair_left = _sequential_holdout_pairs(
            obs, nxt, groups, val_idx,
        )
        if len(pair_left):
            pair_actions = np.stack([
                act[pair_left], act[pair_left + 1],
            ], axis=1).astype(np.float32)
            rollout_members = model.transition_rollout_predictions(
                torch.from_numpy(obs[pair_left]),
                torch.from_numpy(pair_actions),
            )
            rollout_mean = rollout_members.mean(dim=0)
            rollout_target = torch.from_numpy(nxt[pair_left + 1])
            rollout_initial = torch.from_numpy(obs[pair_left])
            two_step_mse = float((
                ((rollout_mean - rollout_target) ** 2) * obs_weights
            ).mean().item())
            two_step_persistence_mse = float((
                ((rollout_initial - rollout_target) ** 2) * obs_weights
            ).mean().item())
            two_step_skill = float(
                1.0 - two_step_mse / max(1e-12, two_step_persistence_mse)
            )
            two_step_groups = int(len(np.unique(groups[pair_left])))
        else:
            two_step_mse = two_step_persistence_mse = two_step_skill = 0.0
            two_step_groups = 0
        progress_target = torch.from_numpy(xg[val_idx]).float().view(-1)
        progress_rmse = float(torch.sqrt(torch.mean(
            (vxp.view(-1) - progress_target) ** 2
        )).item())
        pm = pass_mask[val_idx] > 0.5
        sm = shot_mask[val_idx] > 0.5
        pass_logits = vpass_all[:, :, torch.from_numpy(pm), :].squeeze(-1)
        pass_targets = torch.from_numpy(pass_success[val_idx][pm]).float()
    # Class-balanced training learns a ranking score, not a calibrated probability.
    # Fit a two-parameter Platt map on dev only; the sealed test remains untouched.
    scale = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    calibrator = torch.optim.LBFGS([scale, bias], lr=0.2, max_iter=80)
    def closure():
        calibrator.zero_grad()
        calibrated_probability = torch.sigmoid(
            scale.clamp(0.05, 10.0) * pass_logits + bias
        ).mean(dim=(0, 1))
        loss = torch.nn.functional.binary_cross_entropy(
            calibrated_probability, pass_targets,
        )
        loss.backward()
        return loss
    calibrator.step(closure)
    pass_calibration = {"scale": float(scale.detach().clamp(0.05, 10.0)), "bias": float(bias.detach())}
    with torch.no_grad():
        pass_prob = torch.sigmoid(
            pass_calibration["scale"] * pass_logits + pass_calibration["bias"]
        ).mean(dim=(0, 1)).numpy()
        pass_true = pass_success[val_idx][pm] >= 0.5
        candidates = np.linspace(0.05, 0.95, 181)
        scores = []
        for threshold in candidates:
            candidate = pass_prob >= threshold
            ctpr = float(candidate[pass_true].mean()) if pass_true.any() else 0.5
            ctnr = float((~candidate[~pass_true]).mean()) if (~pass_true).any() else 0.5
            scores.append(0.5 * (ctpr + ctnr))
        pass_threshold = float(candidates[int(np.argmax(scores))])
        pass_pred = pass_prob >= pass_threshold
        pass_tpr = float(pass_pred[pass_true].mean()) if pass_true.any() else 0.5
        pass_tnr = float((~pass_pred[~pass_true]).mean()) if (~pass_true).any() else 0.5
        pass_balanced_accuracy = 0.5 * (pass_tpr + pass_tnr)
        shot_prob = torch.sigmoid(vshot_all).mean(
            dim=(0, 1),
        ).view(-1).numpy()[sm]
        shot_true = shot_goal[val_idx][sm]
        shot_brier = float(np.mean((shot_prob - shot_true) ** 2)) if sm.any() else 0.25
        transition_quality = float(np.clip(np.exp(-20.0 * weighted_mse), 0.0, 1.0))
        pass_skill = float(np.clip((pass_balanced_accuracy - 0.50) / 0.25, 0.0, 1.0))
        pass_sample_confidence = float(np.clip(pm.sum() / 500.0, 0.0, 1.0))
        pass_planner_quality = transition_quality * pass_skill * pass_sample_confidence
        shot_count = int(sm.sum())
        shot_goals = int(shot_true.sum()) if sm.any() else 0
        shot_rate = float(shot_true.mean()) if sm.any() else 0.0
        shot_baseline_brier = max(1e-6, shot_rate * (1.0 - shot_rate))
        shot_skill = float(np.clip(1.0 - shot_brier / shot_baseline_brier, 0.0, 1.0))
        shot_sample_confidence = float(
            np.sqrt(np.clip(shot_goals / 10.0, 0.0, 1.0))
            * np.sqrt(np.clip(shot_count / 100.0, 0.0, 1.0))
        )
        shot_planner_quality = transition_quality * shot_skill * shot_sample_confidence
        planner_quality = min(pass_planner_quality, shot_planner_quality)
    validation = {
        "weighted_obs_mse": weighted_mse,
        "transition_ensemble_size": model.transition_member_count,
        "transition_ensemble_trained": True,
        "transition_primary_mse": transition_primary_mse,
        "transition_member_mean_mse": transition_member_mean_mse,
        "transition_ensemble_gain_vs_primary": (
            transition_primary_mse - weighted_mse
        ),
        "transition_disagreement_error_correlation": (
            transition_disagreement_error_correlation
        ),
        "two_step_rollout": {
            "samples": int(len(pair_left)),
            "groups": two_step_groups,
            "weighted_mse": two_step_mse,
            "persistence_weighted_mse": two_step_persistence_mse,
            "skill_vs_persistence": two_step_skill,
            "action_sequence": "observed_changing_actions",
            "grouped_holdout": True,
        },
        "progress_rmse": progress_rmse,
        "pass_balanced_accuracy": pass_balanced_accuracy,
        "shot_brier": shot_brier,
        "transition_quality": transition_quality,
        "pass_planner_quality": pass_planner_quality,
        "shot_planner_quality": shot_planner_quality,
        "pass_samples": int(pm.sum()),
        "shot_samples": shot_count,
        "shot_goals": shot_goals,
        "planner_quality": planner_quality,
        "pass_calibration": pass_calibration,
        "pass_threshold": pass_threshold,
        "rows": int(len(val_idx)),
        "groups": int(len(val_groups)),
    }
    print(f"validation={validation}")

    save_checkpoint(
        out_path,
        model,
        cfg,
        meta={
            "rows": len(obs),
            "trace_dir": trace_dir,
            "ball_log_dir": ball_dir,
            "supervised_outcome_rows": int(supervised.sum()),
            "train_groups": int(len(unique_groups) - len(val_groups)),
            "shot_pos_weight": shot_pos_weight,
            "pass_pos_rate": pass_pos_rate,
            "validation": validation,
            "dataset_manifest": args.dataset_manifest or None,
            "sealed_test_used": False,
        },
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
