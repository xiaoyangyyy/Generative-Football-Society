#!/usr/bin/env python3
"""Train latent GRU/Transformer world model from traces + ball_log backfill."""

from __future__ import annotations

import argparse
import os
import sys

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
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch required: pip install torch") from exc

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["MATCH_WM_TRANSITION"] = args.transition

    from src.match_engine.world_model.config import WorldModelConfig, default_checkpoint_path, default_trace_dir
    from src.match_engine.world_model.model import build_model, save_checkpoint
    from src.match_engine.world_model.recorder import load_trace_batches
    from src.match_engine.world_model.ball_log_dataset import load_ball_log_dataset

    trace_dir = args.trace_dir or default_trace_dir(base_dir)
    ball_dir = args.ball_log_dir or os.path.join(base_dir, "outputs", "ball_log")
    out_path = args.out or default_checkpoint_path(base_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    obs_t, act_t, nxt_t = load_trace_batches(trace_dir)
    if args.use_ball_log:
        obs_b, act_b, nxt_b = load_ball_log_dataset(ball_dir)
        obs, act, nxt = _merge(obs_t, obs_b), _merge(act_t, act_b), _merge(nxt_t, nxt_b)
    else:
        obs, act, nxt = obs_t, act_t, nxt_t

    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
    nxt = np.nan_to_num(nxt, nan=0.0, posinf=1.0, neginf=0.0)
    act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=0.0)
    obs = np.clip(obs, 0.0, 1.0)
    nxt = np.clip(nxt, 0.0, 1.0)
    valid = (np.std(obs, axis=1) > 0.02) & (np.std(nxt, axis=1) > 0.02)
    obs, act, nxt = obs[valid], act[valid], nxt[valid]

    if len(obs) < 128:
        raise SystemExit(
            f"Not enough training data ({len(obs)} rows). "
            "Run: python scripts/collect_world_model_traces.py --pairs 32"
        )

    is_pass = act[:, 0] > 0.5
    is_shot = act[:, 1] > 0.5
    is_intercept = act[:, 4] > 0.5
    succ_col = act[:, 13] if act.shape[1] > 13 else act[:, 11]
    pass_success = np.where(
        is_pass,
        np.clip(succ_col, 0, 1),
        np.where(is_intercept, 0.15, 0.5),
    ).astype(np.float32)
    shot_goal = np.where(is_shot, np.clip(act[:, 13], 0, 1), 0.0).astype(np.float32)
    xg_delta = (nxt[:, 204] - obs[:, 204]) + (nxt[:, 205] - obs[:, 205])
    xg_delta = xg_delta.astype(np.float32)

    cfg = WorldModelConfig.from_env()
    cfg.transition_type = args.transition
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=min(args.lr, 3e-4))
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    ds = TensorDataset(
        torch.from_numpy(obs),
        torch.from_numpy(act),
        torch.from_numpy(nxt),
        torch.from_numpy(pass_success),
        torch.from_numpy(shot_goal),
        torch.from_numpy(xg_delta),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=len(obs) > args.batch_size)

    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        n = 0
        for batch in loader:
            o, a, no, ps, sg, xg = batch
            _z, pred_obs, pass_logit, xg_pred, shot_logit, _h = model(o, a, h=None)
            p_pass = torch.sigmoid(pass_logit.view(-1))
            p_shot = torch.sigmoid(shot_logit.view(-1))
            p_pass = torch.nan_to_num(p_pass, nan=0.5).clamp(0.0, 1.0)
            p_shot = torch.nan_to_num(p_shot, nan=0.5).clamp(0.0, 1.0)
            loss = (
                mse(pred_obs, no)
                + 0.35 * bce(p_pass, ps)
                + 0.30 * bce(p_shot, sg)
                + 0.25 * mse(xg_pred.view(-1), xg)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        print(
            f"epoch {epoch+1}/{args.epochs} loss={loss_sum/max(1,n):.4f} "
            f"n={len(obs)} transition={args.transition}"
        )

    save_checkpoint(
        out_path,
        model,
        cfg,
        meta={"rows": len(obs), "trace_dir": trace_dir, "ball_log_dir": ball_dir},
    )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
