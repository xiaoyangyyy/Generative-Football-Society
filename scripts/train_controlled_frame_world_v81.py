#!/usr/bin/env python3
"""Train strict provider-LODO action-conditioned frame models and audit rollouts."""

# This executable bootstraps the repository root before importing local packages.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.match_engine.frame_world.controlled import (
    ActionConditionedFrameWorld,
    ControlledFrameConfig,
    observed_actions,
    rollout,
    transition,
)
from src.match_engine.frame_world.schema import BALL_INDEX


class OneStepWindows(Dataset):
    def __init__(self, items, cfg, limit, seed, stride=5):
        self.cfg = cfg
        self.data = []
        self.index = []
        for item in items:
            raw = np.load(ROOT / item["file"], allow_pickle=False)
            seq = {
                k: raw[k]
                for k in ("timestamps", "positions", "velocities", "visible", "teams")
            }
            self.data.append(seq)
            valid = np.arange(cfg.history - 1, len(seq["timestamps"]) - 21)
            gaps = seq["timestamps"][valid + 20] - seq["timestamps"][valid]
            valid = valid[(gaps < 2.2) & (seq["visible"][valid].sum(1) >= 4)]
            self.index.extend((len(self.data) - 1, int(i)) for i in valid[::stride])
        rng = random.Random(seed)
        rng.shuffle(self.index)
        self.index = self.index[:limit]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        s, i = self.index[n]
        d = self.data[s]
        start = i - self.cfg.history + 1
        return tuple(
            torch.from_numpy(x)
            for x in (
                d["positions"][start : i + 1].astype(np.float32),
                d["velocities"][start : i + 1].astype(np.float32),
                d["visible"][start : i + 1].astype(bool),
                d["teams"].astype(np.int64),
                d["positions"][i + 1].astype(np.float32),
                d["visible"][i + 1].astype(bool),
                d["positions"][i + 1 : i + 21].astype(np.float32),
                d["visible"][i + 1 : i + 21].astype(bool),
            )
        )


def fold_split(manifest, held):
    remaining = [x for x in manifest["matches"] if x["provider"] != held]
    test = [x for x in manifest["matches"] if x["provider"] == held]
    train = []
    dev = []
    for provider in sorted({x["provider"] for x in remaining}):
        items = sorted(
            (x for x in remaining if x["provider"] == provider),
            key=lambda x: x["match_id"],
        )
        dev += items[-1:]
        train += items[:-1]
    return train, dev, test


def batch_loss(model, batch, device):
    pos, vel, vis, teams, target, target_vis, targets, rollout_vis = [
        x.to(device) for x in batch
    ]
    actions = observed_actions(vel, vis, model.cfg)
    pred, _, logvar = transition(model, pos, vel, vis, teams, actions)
    mask = vis[:, -1] & target_vis
    error = pred - target
    nll = 0.5 * (error.square() * torch.exp(-logvar) + logvar)
    one = (nll * mask[..., None]).sum() / mask.sum().clamp_min(1) / 2
    closed = rollout(model, pos, vel, vis, teams, 3)
    closed_mask = rollout_vis[:, :3] & vis[:, -1, None]
    rollout_error = (
        ((closed - targets[:, :3]).square() * closed_mask[..., None]).sum()
        / closed_mask.sum().clamp_min(1)
        / 2
    )
    return one + 2 * rollout_error


def planar_distance_m(delta):
    return torch.sqrt((delta[..., 0] * 105) ** 2 + (delta[..., 1] * 68) ** 2)


@torch.no_grad()
def one_step_metrics(model, loader, device):
    model.eval()
    total = base = number = 0.0
    for batch in loader:
        pos, vel, vis, teams, target, target_vis, _, _ = [x.to(device) for x in batch]
        actions = observed_actions(vel, vis, model.cfg)
        pred, _, _ = transition(model, pos, vel, vis, teams, actions)
        baseline = pos[:, -1] + vel[:, -1] / model.cfg.sample_rate_hz
        mask = vis[:, -1] & target_vis
        total += float((planar_distance_m(pred - target) * mask).sum())
        base += float((planar_distance_m(baseline - target) * mask).sum())
        number += int(mask.sum())
    return {
        "ade_m": total / number,
        "baseline_ade_m": base / number,
        "samples": int(number),
    }


@torch.no_grad()
def rollout_metrics(model, loader, device, max_batches=24):
    model.eval()
    horizons = (1, 5, 10, 20)
    sums = {h: {"m": 0.0, "b": 0.0, "n": 0} for h in horizons}
    finite = True
    bounded = True
    max_speed = 0.0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        pos, vel, vis, teams, _, _, targets, target_vis = [x.to(device) for x in batch]
        predicted = rollout(model, pos, vel, vis, teams, 20)
        dt = 1 / model.cfg.sample_rate_hz
        baseline = torch.stack(
            [pos[:, -1] + vel[:, -1] * (i * dt) for i in range(1, 21)], 1
        ).clamp(-0.15, 1.15)
        finite = finite and bool(torch.isfinite(predicted).all())
        bounded = bounded and bool(
            ((predicted >= -0.1501) & (predicted <= 1.1501)).all()
        )
        speed = (predicted[:, 1:] - predicted[:, :-1]) / dt
        max_speed = max(
            max_speed,
            float(
                torch.sqrt((speed[..., 0] * 105) ** 2 + (speed[..., 1] * 68) ** 2).max()
            ),
        )
        for h in horizons:
            mask = vis[:, -1] & target_vis[:, h - 1]
            sums[h]["m"] += float(
                (
                    planar_distance_m(predicted[:, h - 1] - targets[:, h - 1]) * mask
                ).sum()
            )
            sums[h]["b"] += float(
                (planar_distance_m(baseline[:, h - 1] - targets[:, h - 1]) * mask).sum()
            )
            sums[h]["n"] += int(mask.sum())
    return {
        "finite": finite,
        "bounded": bounded,
        "max_speed_mps": max_speed,
        "horizons": {
            str(h): {"ade_m": x["m"] / x["n"], "baseline_ade_m": x["b"] / x["n"]}
            for h, x in sums.items()
        },
    }


@torch.no_grad()
def intervention_metrics(model, loader, device):
    batch = next(iter(loader))
    pos, vel, vis, teams, *_ = [x.to(device) for x in batch]

    def zero(step, p, v, m, t):
        return torch.zeros((*p.shape[:1], p.shape[2], 3), device=device)

    def accelerate(step, p, v, m, t):
        action = zero(step, p, v, m, t)
        action[:, BALL_INDEX, 0] = model.cfg.max_acceleration * (0.85**step)
        action[:, BALL_INDEX, 2] = 1
        return action

    factual = rollout(model, pos, vel, vis, teams, 20, zero)
    treated = rollout(model, pos, vel, vis, teams, 20, accelerate)
    delta = treated[:, -1] - factual[:, -1]
    ball_dx = float((delta[:, BALL_INDEX, 0] * 105).mean())
    response = torch.sqrt(
        (delta[:, :BALL_INDEX, 0] * 105) ** 2 + (delta[:, :BALL_INDEX, 1] * 68) ** 2
    )
    return {
        "ball_forward_displacement_m": ball_dx,
        "other_entity_response_m": float(response.mean()),
        "finite": bool(torch.isfinite(delta).all()),
    }


def train_fold(held, manifest, args, device):
    cfg = ControlledFrameConfig()
    train, dev, test = fold_split(manifest, held)
    sets = [
        OneStepWindows(items, cfg, limit, seed)
        for items, limit, seed in (
            (train, args.train_windows, 11),
            (dev, 12000, 12),
            (test, 20000, 13),
        )
    ]
    loaders = [
        DataLoader(
            s,
            batch_size=256,
            shuffle=i == 0,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for i, s in enumerate(sets)
    ]
    torch.manual_seed(8100 + manifest["providers"].index(held))
    model = ActionConditionedFrameWorld(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loaders[0]:
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(float(loss))
        dev_metrics = one_step_metrics(model, loaders[1], device)
        history.append(
            {"epoch": epoch + 1, "loss": sum(losses) / len(losses), "dev": dev_metrics}
        )
        print(json.dumps({"held": held, **history[-1]}), flush=True)
        if best is None or dev_metrics["ade_m"] < best[0]:
            best = (
                dev_metrics["ade_m"],
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
            )
    model.load_state_dict(best[1])
    one = one_step_metrics(model, loaders[2], device)
    closed = rollout_metrics(model, loaders[2], device)
    intervention = intervention_metrics(model, loaders[2], device)
    path = ROOT / f"data/frame_world/frame_world_v81_lodo_{held}.pt"
    payload = model.checkpoint()
    payload["meta"] = {"held_provider": held}
    torch.save(payload, path)
    gates = {
        "one_step_noninferior": one["ade_m"] <= one["baseline_ade_m"] * 1.05,
        "rollout_finite": closed["finite"] and closed["bounded"],
        "rollout_2s_noninferior": closed["horizons"]["20"]["ade_m"]
        <= closed["horizons"]["20"]["baseline_ade_m"] * 1.05,
        "plausible_speed": closed["max_speed_mps"] < 45,
        "intervention_direction": intervention["ball_forward_displacement_m"] > 1,
        "intervention_finite": intervention["finite"],
    }
    gains = {
        "one_step": one["ade_m"] < one["baseline_ade_m"],
        "rollout_2s": closed["horizons"]["20"]["ade_m"]
        < closed["horizons"]["20"]["baseline_ade_m"],
    }
    return {
        "held_provider": held,
        "train_providers": sorted({x["provider"] for x in train}),
        "windows": {"train": len(sets[0]), "dev": len(sets[1]), "test": len(sets[2])},
        "one_step": one,
        "closed_loop": closed,
        "intervention": intervention,
        "gains": gains,
        "gates": gates,
        "ready": all(gates.values()),
        "artifact": str(path.relative_to(ROOT)),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-windows", type=int, default=60000)
    args = parser.parse_args()
    random.seed(81)
    np.random.seed(81)
    manifest = json.loads((ROOT / "data/frame_world/v8/manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [
        train_fold(provider, manifest, args, device)
        for provider in manifest["providers"]
    ]
    report = {
        "version": "8.1.0-candidate",
        "policy": "strict leave-one-provider-out; held provider excluded from training and dev selection",
        "device": str(device),
        "folds": folds,
        "gates": {
            "all_folds_ready": all(x["ready"] for x in folds),
            "strict_lodo": all(
                x["held_provider"] not in x["train_providers"] for x in folds
            ),
        },
    }
    report["candidate_ready"] = all(report["gates"].values())
    target = ROOT / "reports/acceptance/frame_world_v81.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["candidate_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
