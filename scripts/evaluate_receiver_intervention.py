#!/usr/bin/env python3
"""Matched-seed simulation safety gate for the real-tracking receiver prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import app
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.micro_config import MicroMatchConfig


def _values(summary) -> dict[str, float]:
    passes = summary.passes_home + summary.passes_away
    completed = (
        summary.pass_completion_home * summary.passes_home
        + summary.pass_completion_away * summary.passes_away
    )
    return {
        "completion": completed / max(1, passes),
        "passes": float(passes),
        "xg": float(summary.micro_xg_home + summary.micro_xg_away),
        "shots": float(summary.shots_home + summary.shots_away),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / "reports/acceptance/receiver_intervention.json"))
    args = parser.parse_args()
    baseline, treatment = [], []
    for seed in range(100, 100 + args.samples):
        off = MicroMatchConfig.fast_demo()
        on = MicroMatchConfig.fast_demo()
        on.enable_receiver_ranker = True
        baseline.append(_values(app.run_micro_match("Brazil", "Argentina", seed=seed, config=off)))
        treatment.append(_values(app.run_micro_match("Brazil", "Argentina", seed=seed, config=on)))
    report = {"samples": args.samples, "metrics": {}}
    for metric in baseline[0]:
        before = np.asarray([row[metric] for row in baseline])
        after = np.asarray([row[metric] for row in treatment])
        delta = after - before
        report["metrics"][metric] = {
            "baseline_mean": float(before.mean()),
            "treatment_mean": float(after.mean()),
            "mean_delta": float(delta.mean()),
            "max_abs_delta": float(np.max(np.abs(delta))),
        }
    completion_shift = abs(report["metrics"]["completion"]["mean_delta"])
    volume_shift = abs(report["metrics"]["passes"]["mean_delta"]) / max(1.0, report["metrics"]["passes"]["baseline_mean"])
    report["gate"] = {
        "completion_shift_bounded": completion_shift <= 0.03,
        "pass_volume_shift_bounded": volume_shift <= 0.15,
        "finite": all(np.isfinite(value) for row in baseline + treatment for value in row.values()),
    }
    report["simulation_safe"] = all(report["gate"].values())
    write_json_atomic(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["simulation_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
