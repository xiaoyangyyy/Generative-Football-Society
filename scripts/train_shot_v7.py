#!/usr/bin/env python3
"""Train and evaluate the cross-provider v7 shot probability candidate."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.match_engine.shot_decision import SHOT_FEATURES, ShotEvidence, fit_shot_probability_model

def load_data() -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / "data/external/statsbomb360/derived/shots.csv")]
    frames += [pd.read_csv(p) for p in sorted((ROOT / "data/external/sportec/derived").glob("*_shots.csv"))]
    data = pd.concat(frames, ignore_index=True)
    x, y = data.x.to_numpy(float), data.y.to_numpy(float)
    dx = np.where(data.provider.eq("sportec"), np.minimum(x, 1.0 - x), 1.0 - x)
    lateral = np.abs(y - 0.5)
    data["goal_distance"] = np.hypot(dx, lateral)
    data["goal_angle"] = np.arctan2(0.107, np.maximum(dx, 1e-4) + lateral)
    data["centrality"] = 1.0 - np.minimum(1.0, 2.0 * lateral)
    for name in ("nearest_defender", "keeper_distance"):
        if name not in data:
            data[name] = np.nan
    return data

def split(data):
    parts = {name: [] for name in ("train", "dev", "test")}
    rng = np.random.default_rng(20260724)
    for _, provider in data.groupby("provider"):
        matches = provider.match_id.astype(str).unique(); rng.shuffle(matches)
        n = len(matches); nd = max(1, round(.15 * n)); nt = max(1, round(.15 * n))
        groups = {"dev": matches[:nd], "test": matches[nd:nd + nt], "train": matches[nd + nt:]}
        for name, selected in groups.items():
            parts[name].append(provider[provider.match_id.astype(str).isin(selected)])
    return tuple(pd.concat(parts[name], ignore_index=True) for name in ("train", "dev", "test"))

def isotonic(probability, truth, bins=30):
    groups = np.array_split(np.argsort(probability), min(bins, len(probability)))
    blocks = [[float(probability[g].mean()), float(truth[g].sum()), len(g)] for g in groups if len(g)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] / blocks[i][2] <= blocks[i + 1][1] / blocks[i + 1][2]: i += 1
        else:
            a, b = blocks[i], blocks[i + 1]; n = a[2] + b[2]
            blocks[i:i + 2] = [[(a[0] * a[2] + b[0] * b[2]) / n, a[1] + b[1], n]]
            i = max(0, i - 1)
    return tuple(b[0] for b in blocks), tuple(b[1] / b[2] for b in blocks)

def auc(y, p):
    order = np.argsort(p); ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    positive = y == 1; n = positive.sum()
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * (~positive).sum()))

def main() -> int:
    data = load_data(); train, dev, test = split(data); columns = list(SHOT_FEATURES)
    model = fit_shot_probability_model(train[columns].to_numpy(float), train.goal.to_numpy(float))
    dp = np.array([model.predict(row) for row in dev[columns].to_numpy(float)])
    knots, values = isotonic(dp, dev.goal.to_numpy(float))
    model = replace(model, calibration_knots=knots, calibration_values=values)
    probability = np.array([model.predict(row) for row in test[columns].to_numpy(float)])
    truth = test.goal.to_numpy(float); baseline = np.full(len(test), train.goal.mean())
    brier = float(np.mean((probability - truth) ** 2)); base = float(np.mean((baseline - truth) ** 2))
    evidence = ShotEvidence(len(test), int(truth.sum()), test.match_id.nunique(), test.provider.nunique(), brier, base)
    report = {"version": "7.0.0-candidate", "split_policy": "provider-stratified match holdout",
      "rows": {"train": len(train), "dev": len(dev), "test": len(test)}, "providers": sorted(data.provider.unique()),
      "test": {"shots": len(test), "goals": int(truth.sum()), "matches": int(test.match_id.nunique()),
               "auc": auc(truth, probability), "brier": brier, "baseline_brier": base},
      "calibration": {"isotonic_blocks": len(knots), "fit_split": "dev"},
      "gate": evidence.gate(), "candidate_ready": evidence.enabled}
    (ROOT / "data/world_model/shot_v7_candidate.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    target = ROOT / "reports/acceptance/shot_v7.json"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0 if evidence.enabled else 1

if __name__ == "__main__": raise SystemExit(main())
