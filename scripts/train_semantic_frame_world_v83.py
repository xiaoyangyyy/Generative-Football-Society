#!/usr/bin/env python3
"""Train actor-aware semantic frame transitions with strict provider LODO."""

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
from src.match_engine.frame_world.semantic import (
    SemanticActionFrameWorld,
    SemanticFrameConfig,
)
from src.match_engine.frame_world.schema import BALL_INDEX


class EventWindows(Dataset):
    def __init__(self, items, cfg, limit, seed):
        self.cfg = cfg
        self.data = []
        self.index = []
        rng = random.Random(seed)
        for item in items:
            frame = np.load(ROOT / item["file"], allow_pickle=False)
            labels = np.load(ROOT / item["actions"]["labels_file"], allow_pickle=False)
            d = {k: frame[k] for k in ("positions", "velocities", "visible", "teams")}
            d["labels"] = labels["labels"]
            d["directions"] = labels["directions"]
            d["actors"] = labels["actor_indices"]
            d["targets"] = labels["target_indices"]
            d["known"] = labels["known"]
            self.data.append(d)
            valid = np.arange(cfg.history - 1, len(d["labels"]) - cfg.horizon_steps)
            valid = valid[
                (d["visible"][valid].sum(1) >= 4)
                & (d["visible"][valid + cfg.horizon_steps].sum(1) >= 4)
            ]
            positive = valid[d["labels"][valid].any(1)]
            negative = valid[~d["labels"][valid].any(1)].tolist()
            rng.shuffle(negative)
            chosen = list(positive) + negative[: max(len(positive), 1000)]
            self.index.extend((len(self.data) - 1, int(i)) for i in chosen)
        rng.shuffle(self.index)
        self.index = self.index[:limit]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        s, i = self.index[n]
        d = self.data[s]
        start = i - self.cfg.history + 1
        target = i + self.cfg.horizon_steps
        semantic = np.r_[d["labels"][i], d["directions"][i]].astype(np.float32)
        return tuple(
            torch.from_numpy(x)
            for x in (
                d["positions"][start : i + 1].astype(np.float32),
                d["velocities"][start : i + 1].astype(np.float32),
                d["visible"][start : i + 1].astype(bool),
                d["teams"].astype(np.int64),
                semantic,
                d["actors"][i].astype(np.int64),
                d["targets"][i].astype(np.int64),
                d["known"].astype(bool),
                d["positions"][target].astype(np.float32),
                d["visible"][target].astype(bool),
            )
        )


def join_manifests():
    frames = json.loads((ROOT / "data/frame_world/v8/manifest.json").read_text())
    actions = json.loads(
        (ROOT / "data/frame_world/v82_actions/manifest.json").read_text()
    )
    lookup = {(x["provider"], x["match_id"]): x for x in actions["matches"]}
    return [
        {**x, "actions": lookup[(x["provider"], x["match_id"])]}
        for x in frames["matches"]
    ]


def split(items, held):
    test = [x for x in items if x["provider"] == held]
    rest = [x for x in items if x["provider"] != held]
    train = []
    dev = []
    for provider in sorted({x["provider"] for x in rest}):
        group = sorted(
            (x for x in rest if x["provider"] == provider), key=lambda x: x["match_id"]
        )
        dev += group[-1:]
        train += group[:-1]
    return train, dev, test


def loss_batch(model, batch, device):
    pos, vel, vis, teams, actions, actors, targets, known, target, target_vis = [
        x.to(device) for x in batch
    ]
    pred, logits = model.predict(pos, vel, vis, teams, actions, actors, targets)
    classes = actions[:, :3]
    mask = vis[:, -1] & target_vis
    weight = mask.float()
    ball_event = classes[:, :2].any(1)
    weight[:, BALL_INDEX] += 20 * ball_event
    if model.cfg.pressure_enabled:
        pressure = classes[:, 2] > 0
        for indices in (actors[:, 2], targets[:, 2]):
            valid = pressure & (indices >= 0) & (indices < weight.shape[1])
            rows = torch.arange(len(indices), device=device)[valid]
            weight[rows, indices[valid].long()] += 5
    mse = (
        ((pred - target).square() * weight[..., None]).sum()
        / weight.sum().clamp_min(1)
        / 2
    )
    element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        classes,
        reduction="none",
        pos_weight=torch.tensor([3.0, 30.0, 2.0], device=device),
    )
    bce = (element * known).sum() / known.sum().clamp_min(1)
    return mse + 0.05 * bce


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {
        "conditioned": 0.0,
        "zero": 0.0,
        "base": 0.0,
        "n": 0,
        "positive_n": 0,
        "positive_conditioned": 0.0,
        "positive_zero": 0.0,
        "ball_n": 0,
        "ball_conditioned": 0.0,
        "ball_zero": 0.0,
        "pressure_n": 0,
        "pressure_conditioned": 0.0,
        "pressure_zero": 0.0,
    }
    cls = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(3)]

    def meters(delta):
        return torch.sqrt((delta[..., 0] * 105) ** 2 + (delta[..., 1] * 68) ** 2)

    for batch in loader:
        pos, vel, vis, teams, actions, actors, targets, known, target, target_vis = [
            x.to(device) for x in batch
        ]
        pred, logits = model.predict(pos, vel, vis, teams, actions, actors, targets)
        zero, _ = model.predict(
            pos, vel, vis, teams, torch.zeros_like(actions), actors, targets
        )
        base = (
            pos[:, -1]
            + vel[:, -1] * (model.cfg.horizon_steps / model.cfg.sample_rate_hz)
        ).clamp(-0.15, 1.15)
        mask = vis[:, -1] & target_vis
        n = int(mask.sum())
        sums["n"] += n
        for key, value in (("conditioned", pred), ("zero", zero), ("base", base)):
            sums[key] += float((meters(value - target) * mask).sum())
        classes = actions[:, :3]
        positive = classes.any(1)
        pmask = mask & positive[:, None]
        pn = int(pmask.sum())
        sums["positive_n"] += pn
        sums["positive_conditioned"] += float((meters(pred - target) * pmask).sum())
        sums["positive_zero"] += float((meters(zero - target) * pmask).sum())
        ball_event = (
            classes[:, :2].any(1)
            & (torch.linalg.vector_norm(actions[:, 3:], dim=1) > 0.5)
            & mask[:, BALL_INDEX]
        )
        sums["ball_n"] += int(ball_event.sum())
        sums["ball_conditioned"] += float(
            (meters(pred[:, BALL_INDEX] - target[:, BALL_INDEX]) * ball_event).sum()
        )
        sums["ball_zero"] += float(
            (meters(zero[:, BALL_INDEX] - target[:, BALL_INDEX]) * ball_event).sum()
        )
        if model.cfg.pressure_enabled:
            for indices in (actors[:, 2], targets[:, 2]):
                valid = (classes[:, 2] > 0) & (indices >= 0) & (indices < mask.shape[1])
                rows = torch.arange(len(indices), device=device)[valid]
                slots = indices[valid].long()
                supervised = mask[rows, slots]
                sums["pressure_n"] += int(supervised.sum())
                sums["pressure_conditioned"] += float(
                    (meters(pred[rows, slots] - target[rows, slots]) * supervised).sum()
                )
                sums["pressure_zero"] += float(
                    (meters(zero[rows, slots] - target[rows, slots]) * supervised).sum()
                )
        guessed = torch.sigmoid(logits) >= 0.5
        for i in range(3):
            valid = known[:, i]
            cls[i]["tp"] += int((guessed[:, i] & classes[:, i].bool() & valid).sum())
            cls[i]["fp"] += int((guessed[:, i] & ~classes[:, i].bool() & valid).sum())
            cls[i]["fn"] += int((~guessed[:, i] & classes[:, i].bool() & valid).sum())
    metrics = {k: sums[k] / sums["n"] for k in ("conditioned", "zero", "base")}
    metrics["positive_conditioned"] = sums["positive_conditioned"] / max(
        1, sums["positive_n"]
    )
    metrics["positive_zero"] = sums["positive_zero"] / max(1, sums["positive_n"])
    metrics["ball_event_conditioned"] = sums["ball_conditioned"] / max(
        1, sums["ball_n"]
    )
    metrics["ball_event_zero"] = sums["ball_zero"] / max(1, sums["ball_n"])
    metrics["ball_events"] = sums["ball_n"]
    metrics["pressure_local_conditioned"] = sums["pressure_conditioned"] / max(
        1, sums["pressure_n"]
    )
    metrics["pressure_local_zero"] = sums["pressure_zero"] / max(1, sums["pressure_n"])
    metrics["pressure_local_entities"] = sums["pressure_n"]
    metrics["f1"] = [2 * x["tp"] / max(1, 2 * x["tp"] + x["fp"] + x["fn"]) for x in cls]
    return metrics


def train_fold(held, items, args, device):
    train, dev, test = split(items, held)
    pressure_enabled = any(x["actions"]["availability"]["pressure"] for x in train)
    cfg = SemanticFrameConfig(
        pressure_enabled=pressure_enabled, actor_roles_enabled=pressure_enabled
    )
    sets = [
        EventWindows(x, cfg, n, s)
        for x, n, s in ((train, args.windows, 31), (dev, 12000, 32), (test, 20000, 33))
    ]
    loaders = [
        DataLoader(x, batch_size=256, shuffle=i == 0, num_workers=0)
        for i, x in enumerate(sets)
    ]
    torch.manual_seed(8300 + len(held))
    model = SemanticActionFrameWorld(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loaders[0]:
            opt.zero_grad(set_to_none=True)
            loss = loss_batch(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            losses.append(float(loss))
        devm = evaluate(model, loaders[1], device)
        history.append(
            {"epoch": epoch + 1, "loss": sum(losses) / len(losses), "dev": devm}
        )
        print(json.dumps({"held": held, **history[-1]}), flush=True)
        score = devm["positive_conditioned"]
        if best is None or score < best[0]:
            best = (score, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    metrics = evaluate(model, loaders[2], device)
    path = ROOT / f"data/frame_world/frame_world_v83_actor_lodo_{held}.pt"
    payload = model.checkpoint()
    payload["meta"] = {"held_provider": held, "pressure_enabled": pressure_enabled}
    torch.save(payload, path)
    direction_available = metrics["ball_events"] >= 20
    gates = {
        "finite": all(np.isfinite(x) for x in metrics.values() if isinstance(x, float)),
        "direction_conditioning_gain": not direction_available
        or metrics["ball_event_conditioned"] < metrics["ball_event_zero"],
        "positive_conditioning_noninferior": metrics["positive_conditioned"]
        <= metrics["positive_zero"] * 1.05,
        "transition_noninferior": metrics["conditioned"] <= metrics["base"] * 1.05,
    }
    return {
        "held_provider": held,
        "train_providers": sorted({x["provider"] for x in train}),
        "pressure_enabled_from_training_evidence": pressure_enabled,
        "direction_evidence_available": direction_available,
        "metrics": metrics,
        "gates": gates,
        "ready": all(gates.values()),
        "artifact": path.relative_to(ROOT).as_posix(),
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--windows", type=int, default=50000)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = join_manifests()
    folds = [
        train_fold(x, items, args, device)
        for x in ("metrica", "skillcorner", "sportec")
    ]
    report = {
        "version": "8.3.0-candidate",
        "policy": "actor-aware strict provider LODO; pressure intervention enabled only from training evidence",
        "folds": folds,
        "gates": {
            "strict_lodo": all(
                x["held_provider"] not in x["train_providers"] for x in folds
            ),
            "evidence_switch": all(
                x["pressure_enabled_from_training_evidence"]
                == ("skillcorner" in x["train_providers"])
                for x in folds
            ),
            "all_folds_ready": all(x["ready"] for x in folds),
        },
    }
    report["candidate_ready"] = all(report["gates"].values())
    target = ROOT / "reports/acceptance/frame_world_v83.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["candidate_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
