#!/usr/bin/env python3
"""Train on period 1 and evaluate receiver choice on held-out period 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.receiver_ranker import FEATURES as FEATURE_SCHEMA

FEATURES = list(FEATURE_SCHEMA)


def _metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    work = frame[["event_id", "is_receiver"]].copy()
    work["score"] = scores
    ranks = []
    for _, group in work.groupby("event_id", sort=False):
        ordered = group.sort_values("score", ascending=False)["is_receiver"].to_numpy()
        ranks.append(int(np.flatnonzero(ordered == 1)[0]) + 1)
    values = np.asarray(ranks)
    return {"events": len(values), "top1": float(np.mean(values == 1)), "top3": float(np.mean(values <= 3)), "mrr": float(np.mean(1.0 / values))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=[str(ROOT / "data/external/metrica/pass_candidates_game1.csv")])
    parser.add_argument("--out", default=str(ROOT / "reports/acceptance/receiver_ranker.json"))
    parser.add_argument("--model-out", default=str(ROOT / "data/world_model/receiver_ranker.json"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.data], ignore_index=True)
    games = sorted(data.game.unique())
    if len(games) >= 2:
        train, test = data[data.game == games[0]].copy(), data[data.game == games[-1]].copy()
        split = f"game{games[0]}-train_game{games[-1]}-sealed"
    else:
        train, test = data[data.period == 1].copy(), data[data.period == 2].copy()
        split = "period1-train_period2-sealed"
    mean = train[FEATURES].mean().to_numpy()
    scale = train[FEATURES].std().replace(0, 1).to_numpy()
    x_train = (train[FEATURES].to_numpy() - mean) / scale
    x_test = (test[FEATURES].to_numpy() - mean) / scale
    y = train.is_receiver.to_numpy(dtype=float)
    weights = np.zeros(len(FEATURES), dtype=float)
    bias = 0.0
    for _ in range(1200):
        logits = np.clip(x_train @ weights + bias, -20, 20)
        prob = 1.0 / (1.0 + np.exp(-logits))
        error = prob - y
        weights -= 0.08 * ((x_train.T @ error) / len(y) + 0.002 * weights)
        bias -= 0.08 * float(error.mean())
    learned = _metrics(test, x_test @ weights + bias)
    distance = _metrics(test, -test.distance.to_numpy())
    random_mrr = float(test.groupby("event_id").size().map(lambda n: np.mean(1.0 / np.arange(1, n + 1))).mean())
    report = {
        "split": split,
        "features": FEATURES,
        "train_events": int(train.event_id.nunique()),
        "test_events": int(test.event_id.nunique()),
        "distance_baseline": distance,
        "learned_ranker": learned,
        "random_expected_mrr": random_mrr,
        "weights": dict(zip(FEATURES, weights.tolist())),
    }
    report["gate"] = {
        "beats_distance_top1": learned["top1"] > distance["top1"],
        "beats_distance_mrr": learned["mrr"] > distance["mrr"],
        "minimum_test_events": learned["events"] >= 500,
    }
    report["candidate_ready"] = all(report["gate"].values())
    write_json_atomic(args.out, report)
    model = {
        "version": 1,
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "training_source": "Metrica Sample Game 1",
        "sealed_evaluation": report,
        "usage": "receiver utility prior only; never pass completion probability",
    }
    if report["candidate_ready"]:
        write_json_atomic(args.model_out, model)
    print(json.dumps(report, indent=2))
    return 0 if report["candidate_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
