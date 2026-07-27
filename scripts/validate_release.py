#!/usr/bin/env python3
"""Evidence-producing release gate for the sealed world-model test set."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import files_for_split, load_manifest, write_json_atomic
from src.match_engine.world_model.schema import PASS_OUTCOME_INDEX, SHOT_GOAL_INDEX


def _ece(prob: np.ndarray, truth: np.ndarray, bins: int = 8) -> float:
    if not len(prob):
        return math.nan
    score = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if mask.any():
            score += float(mask.mean()) * abs(float(prob[mask].mean()) - float(truth[mask].mean()))
    return score


def _auc(prob: np.ndarray, truth: np.ndarray) -> float:
    positive, negative = prob[truth], prob[~truth]
    if not len(positive) or not len(negative):
        return 0.5
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def _load_rows(trace_dir: Path, names: set[str]) -> dict[str, list[dict]]:
    groups = {}
    for name in sorted(names):
        path = trace_dir / name
        groups[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data/world_model/dataset_manifest.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "data/world_model/latent_wm.pt"))
    parser.add_argument("--out", default=str(ROOT / "reports/acceptance/release_metrics.json"))
    parser.add_argument("--max-starts", type=int, default=100000)
    args = parser.parse_args()

    from src.match_engine.world_model.inference import WorldModelRuntime

    manifest = load_manifest(args.manifest)
    trace_dir = Path(manifest["source_root"])
    sealed = files_for_split(manifest, {"sealed_test"})
    groups = _load_rows(trace_dir, sealed)
    runtime = WorldModelRuntime.load(args.checkpoint)
    horizons = (1, 3, 5, 10)
    errors = {h: [] for h in horizons}
    pass_prob, pass_true, pass_prior, shot_prob, shot_true = [], [], [], [], []
    starts = 0
    for rows in groups.values():
        for start in range(len(rows)):
            if starts >= args.max_starts:
                break
            base = rows[start]
            current = np.asarray(base["obs"], dtype=np.float32)
            for step in range(1, max(horizons) + 1):
                index = start + step - 1
                if index >= len(rows):
                    break
                action = np.asarray(rows[index]["action"], dtype=np.float32)
                output = runtime.imagine(current, action, steps=1)
                current = np.asarray(output.next_obs, dtype=np.float32)
                if step == 1:
                    if action[0] > 0.5:
                        pass_prob.append(output.pass_success)
                        pass_true.append(action[PASS_OUTCOME_INDEX])
                        pass_prior.append(action[13] if np.isfinite(action[13]) else 0.5)
                    if action[1] > 0.5:
                        shot_prob.append(output.shot_goal_prob)
                        shot_true.append(action[SHOT_GOAL_INDEX])
                if step in errors:
                    target = np.asarray(rows[index]["next_obs"], dtype=np.float32)
                    errors[step].append(float(np.mean((current - target) ** 2)))
            starts += 1
        if starts >= args.max_starts:
            break
    pp, pt = np.asarray(pass_prob), np.asarray(pass_true) >= 0.5
    prior = np.clip(np.asarray(pass_prior), 0.0, 1.0)
    sp, st = np.asarray(shot_prob), np.asarray(shot_true)
    pred = pp >= runtime.pass_threshold
    tpr = float(pred[pt].mean()) if pt.any() else 0.5
    tnr = float((~pred[~pt]).mean()) if (~pt).any() else 0.5
    finite_errors = {h: np.asarray(v)[np.isfinite(v)] for h, v in errors.items()}
    metrics = {
        "checkpoint": args.checkpoint,
        "dataset_manifest": args.manifest,
        "split": "sealed_test",
        "groups": len(groups),
        "starts": starts,
        "rollout_mse": {str(h): float(np.mean(v)) if len(v) else None for h, v in finite_errors.items()},
        "rollout_nonfinite": {str(h): len(errors[h]) - len(finite_errors[h]) for h in horizons},
        "pass": {
            "samples": len(pp),
            "balanced_accuracy": 0.5 * (tpr + tnr),
            "auc": _auc(pp, pt),
            "brier": float(np.mean((pp - pt) ** 2)),
            "ece": _ece(pp, pt),
            "physics_prior_auc": _auc(prior, pt),
            "physics_prior_brier": float(np.mean((prior - pt) ** 2)),
            "physics_prior_ece": _ece(prior, pt),
        },
        "shot": {"samples": len(sp), "goals": int(st.sum()), "brier": float(np.mean((sp-st)**2)) if len(sp) else None, "ece": _ece(sp, st)},
    }
    enough_shots = len(sp) >= 40 and st.sum() >= 3
    metrics["gates"] = {
        "one_step_mse": metrics["rollout_mse"]["1"] is not None and metrics["rollout_mse"]["1"] < 0.02 and metrics["rollout_nonfinite"]["1"] == 0,
        "ten_step_mse": metrics["rollout_mse"]["10"] is not None and metrics["rollout_mse"]["10"] < 0.08 and metrics["rollout_nonfinite"]["10"] == 0,
        "pass_balanced_accuracy": metrics["pass"]["balanced_accuracy"] >= 0.52,
        "pass_preserves_prior_ranking": metrics["pass"]["auc"] >= metrics["pass"]["physics_prior_auc"] - 0.005,
        "pass_calibration": metrics["pass"]["brier"] <= metrics["pass"]["physics_prior_brier"],
        "shot_evidence": bool(enough_shots),
    }
    metrics["release_ready"] = all(metrics["gates"].values())
    write_json_atomic(args.out, metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
