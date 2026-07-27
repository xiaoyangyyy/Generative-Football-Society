#!/usr/bin/env python3
"""Train and group-evaluate the v7 joint pass model across open providers."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.match_engine.joint_pass_model import PASS_FEATURES, fit_joint_pass_model, pass_model_gate


def _read_many(pattern: str) -> pd.DataFrame:
    files = sorted(ROOT.glob(pattern))
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else pd.DataFrame()


def _assign_splits(frame: pd.DataFrame) -> pd.Series:
    mapping = {}
    for provider, group in frame[["provider", "match_id"]].drop_duplicates().groupby("provider"):
        matches = sorted(group.match_id.astype(str))
        count = len(matches)
        for index, match in enumerate(matches):
            fraction = (index + 0.5) / count
            mapping[(provider, match)] = "train" if fraction < 0.70 else ("dev" if fraction < 0.85 else "test")
    return pd.Series([mapping[(provider, str(match))] for provider, match in zip(frame.provider, frame.match_id)], index=frame.index)


def _auc(probability: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=bool)
    positive, negative = int(truth.sum()), int((~truth).sum())
    if not positive or not negative:
        return 0.5
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    return float((ranks[truth].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _ece(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(probability[mask].mean()) - float(truth[mask].mean()))
    return result


def _completion_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict:
    truth = frame.completion.to_numpy(dtype=float) >= 0.5
    prediction = probability >= threshold
    tpr = float(prediction[truth].mean()) if truth.any() else 0.5
    tnr = float((~prediction[~truth]).mean()) if (~truth).any() else 0.5
    prior = np.clip(frame.physics_prior.to_numpy(dtype=float), 0.01, 0.99)
    match_auc = []
    for _, indices in frame.groupby(["provider", "match_id"]).groups.items():
        local = np.asarray(list(indices), dtype=int)
        local_truth = truth[frame.index.get_indexer(local)]
        if local_truth.any() and (~local_truth).any():
            match_auc.append(_auc(probability[frame.index.get_indexer(local)], local_truth))
    return {
        "samples": len(frame), "matches": int(frame.match_id.nunique()),
        "balanced_accuracy": 0.5 * (tpr + tnr), "auc": _auc(probability, truth),
        "brier": float(np.mean((probability - truth) ** 2)), "ece": _ece(probability, truth),
        "physics_prior_auc": _auc(prior, truth), "physics_prior_brier": float(np.mean((prior - truth) ** 2)),
        "worst_match_auc": float(np.quantile(match_auc, 0.10)) if match_auc else 0.5,
    }


def _receiver_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    work = frame[["event_id", "is_receiver"]].copy()
    work["score"] = scores
    ranks = []
    for _, group in work.groupby("event_id", sort=False):
        if group.is_receiver.sum() != 1 or len(group) < 2:
            continue
        ordered = group.sort_values("score", ascending=False).is_receiver.to_numpy()
        ranks.append(int(np.flatnonzero(ordered == 1)[0]) + 1)
    values = np.asarray(ranks)
    if not len(values):
        return {"events": 0, "top1": 0.0, "top3": 0.0, "mrr": 0.0}
    return {"events": len(values), "top1": float(np.mean(values == 1)), "top3": float(np.mean(values <= 3)), "mrr": float(np.mean(1.0 / values))}


def _fit_platt(probability: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    y = np.asarray(truth, dtype=float)
    scale, bias = 1.0, 0.0
    for _ in range(40):
        logits = np.clip(scale * x + bias, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        weight = np.maximum(p * (1.0 - p), 1e-6)
        gradient = np.array([np.sum((p - y) * x), np.sum(p - y)])
        hessian = np.array([[np.sum(weight * x * x), np.sum(weight * x)], [np.sum(weight * x), np.sum(weight)]])
        step = np.linalg.solve(hessian + 1e-6 * np.eye(2), gradient)
        scale, bias = scale - float(step[0]), bias - float(step[1])
        scale = float(np.clip(scale, 0.05, 10.0))
        if np.linalg.norm(step) < 1e-7:
            break
    return scale, bias


def _fit_isotonic(probability: np.ndarray, truth: np.ndarray, bins: int = 40) -> tuple[tuple[float, ...], tuple[float, ...]]:
    order = np.argsort(probability)
    groups = np.array_split(order, min(bins, len(order)))
    x = [float(probability[group].mean()) for group in groups if len(group)]
    y = [float(np.asarray(truth)[group].mean()) for group in groups if len(group)]
    weights = [float(len(group)) for group in groups if len(group)]
    blocks = [[xv, yv, weight] for xv, yv, weight in zip(x, y, weights)]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][1] <= blocks[index + 1][1]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[2] + right[2]
        merged = [(left[0] * left[2] + right[0] * right[2]) / weight, (left[1] * left[2] + right[1] * right[2]) / weight, weight]
        blocks[index:index + 2] = [merged]
        index = max(0, index - 1)
    knots = tuple(float(block[0]) for block in blocks)
    values = tuple(float(np.clip(0.9 * block[1] + 0.1 * block[0], 0.001, 0.999)) for block in blocks)
    return knots, values


def main() -> int:
    sportec = _read_many("data/external/sportec/derived/*_passes.csv")
    statsbomb = pd.read_csv(ROOT / "data/external/statsbomb360/derived/passes.csv")
    skill_pass = pd.read_csv(ROOT / "data/external/skillcorner/derived/passes.csv")
    skill_receiver = pd.read_csv(ROOT / "data/external/skillcorner/derived/receiver_candidates.csv")
    metrica = _read_many("data/external/metrica/pass_candidates_game*.csv")
    frames = []
    sportec.loc[sportec.is_receiver.ne(1), "completion"] = np.nan
    frames.append(sportec)
    for frame in (statsbomb, skill_pass):
        frame["is_receiver"] = 1
        frames.append(frame)
    skill_receiver["completion"] = np.nan
    skill_receiver["physics_prior"] = 0.5
    frames.append(skill_receiver)
    if not metrica.empty:
        metrica["provider"] = "metrica"
        metrica["match_id"] = "game" + metrica.game.astype(str)
        metrica["completion"] = np.nan
        metrica["physics_prior"] = 0.5
        frames.append(metrica)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["match_id"] = data.match_id.astype(str)
    data["event_id"] = data.provider.astype(str) + ":" + data.event_id.astype(str)
    for feature in PASS_FEATURES:
        if feature not in data:
            data[feature] = np.nan
    data["split"] = _assign_splits(data)
    train = data[data.split.eq("train")].copy()
    model = fit_joint_pass_model(
        train[list(PASS_FEATURES)].to_numpy(), train.event_id.to_numpy(),
        train.is_receiver.fillna(0).to_numpy(), train.completion.to_numpy(),
        train.physics_prior.fillna(0.5).to_numpy(), steps=100, learning_rate=0.025,
    )
    def predictions(frame: pd.DataFrame, *, score_receivers: bool = False):
        values = frame[list(PASS_FEATURES)].to_numpy(dtype=float)
        result = [model.predict(row, physics_prior=prior) for row, prior in zip(values, frame.physics_prior.fillna(0.5))]
        receiver = np.zeros(len(frame), dtype=float)
        if score_receivers:
            for indices in frame.groupby("event_id", sort=False).indices.values():
                if len(indices) >= 2:
                    receiver[indices] = model.score_receiver_candidates(values[indices])
        return receiver, np.array([item.completion_probability for item in result])
    dev = data[data.split.eq("dev") & data.completion.notna() & data.is_receiver.eq(1)].copy().reset_index(drop=True)
    _, dev_probability = predictions(dev)
    calibration_scale, calibration_bias = _fit_platt(dev_probability, dev.completion.to_numpy())
    model = replace(model, calibration_scale=calibration_scale, calibration_bias=calibration_bias)
    _, dev_probability = predictions(dev)
    calibration_knots, calibration_values = _fit_isotonic(dev_probability, dev.completion.to_numpy())
    model = replace(model, calibration_knots=calibration_knots, calibration_values=calibration_values)
    _, dev_probability = predictions(dev)
    thresholds = np.linspace(0.05, 0.95, 181)
    threshold = max(thresholds, key=lambda value: _completion_metrics(dev, dev_probability, value)["balanced_accuracy"])
    test_completion = data[data.split.eq("test") & data.completion.notna() & data.is_receiver.eq(1)].copy().reset_index(drop=True)
    _, test_probability = predictions(test_completion)
    completion = _completion_metrics(test_completion, test_probability, float(threshold))
    test_receiver = data[data.split.eq("test") & data.is_receiver.notna()].copy().reset_index(drop=True)
    receiver_score, _ = predictions(test_receiver, score_receivers=True)
    receiver = _receiver_metrics(test_receiver, receiver_score)
    gate = pass_model_gate(completion, {"auc": completion["physics_prior_auc"], "brier": completion["physics_prior_brier"]})
    report = {
        "version": "7.0.0-candidate", "split_policy": "provider-stratified match holdout",
        "rows": {name: int(data.split.eq(name).sum()) for name in ("train", "dev", "test")},
        "providers": sorted(data.provider.unique()), "completion": completion,
        "receiver": receiver, "threshold": float(threshold), "gate": gate,
        "calibration": {"scale": calibration_scale, "bias": calibration_bias, "isotonic_blocks": len(calibration_knots), "fit_split": "dev"},
        "candidate_ready": all(gate.values()) and receiver["mrr"] >= 0.40,
    }
    payload = model.to_dict()
    payload["evaluation"] = report
    output = ROOT / "data/world_model/joint_pass_v7_candidate.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / "reports/acceptance/joint_pass_v7.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
