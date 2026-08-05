#!/usr/bin/env python3
"""Compare sim micro metrics to observable contract (hard + soft constraints)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines

REPORTS = ROOT / "reports"


def _bootstrap_mean_intervals(
    rows: list[dict], *, seed: int, draws: int = 2000,
) -> dict[str, dict[str, float | int]]:
    """Deterministic match-level bootstrap intervals for scalar observables."""
    if not rows:
        return {}
    excluded = {"fixture"}
    keys = sorted(set.intersection(*(
        {key for key, value in row.items()
         if key not in excluded and isinstance(value, (int, float))}
        for row in rows
    )))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(draws, len(rows)))
    report = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        means = values[indices].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        report[key] = {
            "samples": len(values),
            "mean": float(values.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
    return report


def _parse_fixtures(tokens: list[str] | None) -> list[tuple[str, str]]:
    if not tokens:
        return DEFAULT_FIXTURES
    out: list[tuple[str, str]] = []
    for token in tokens:
        if ":" in token:
            h, a = token.split(":", 1)
            out.append((h.strip(), a.strip()))
    return out or DEFAULT_FIXTURES


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=3, help="Seeds per fixture")
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--fixtures", nargs="*", default=None, help="Pairs HOME:AWAY")
    p.add_argument("--quick", action="store_true", help="One fixture, one sample")
    p.add_argument("--pipeline", choices=["M0", "M1", "T0"], default="M0")
    p.add_argument("--with-style-contrast", action="store_true")
    p.add_argument("--report", type=str, default="", help="Write JSON report path")
    args = p.parse_args()

    if args.quick:
        args.samples = 1
        args.fixtures = ["Mexico:South Korea"]
        if args.match_seconds >= 5400.0:
            args.match_seconds = 1200.0

    contract = load_contract()
    baselines = load_statsbomb_baselines()
    fixtures = _parse_fixtures(args.fixtures)
    spec = PIPELINE_PRESETS[args.pipeline]

    with ablation_context(spec) as cfg:
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=args.samples,
            seed_start=args.seed_start,
            match_seconds=args.match_seconds,
            spec=spec,
            cfg=cfg,
        )

    if not rows:
        raise SystemExit("No fixtures simulated — check team names.")

    evaluation = evaluate_rows(rows, contract=contract, baselines=baselines)
    out = {
        "source": baselines["source"],
        "pipeline": args.pipeline,
        "fixtures": [f"{h}_vs_{a}" for h, a in fixtures],
        "samples_total": len(rows),
        "raw_rows": rows,
        "uncertainty": {
            "method": "deterministic_match_bootstrap",
            "draws": 2000,
            "seed": args.seed_start,
            "metrics": _bootstrap_mean_intervals(
                rows, seed=args.seed_start, draws=2000,
            ),
        },
        "report": evaluation["hard_metrics"],
        "soft_constraints": evaluation["soft_constraints"],
        "failed_metrics": evaluation["failed_hard"],
        "failed_soft": evaluation["failed_soft"],
        "all_pass": evaluation["all_pass"],
        "all_hard_pass": evaluation["all_hard_pass"],
        "all_soft_pass": evaluation["all_soft_pass"],
    }
    print(json.dumps(out, indent=2))

    report_path = Path(args.report) if args.report else REPORTS / f"calibration_{args.pipeline.lower()}.json"
    if args.report or not args.quick:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.with_style_contrast:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "benchmark_style_contrast.py"), "--samples", "2"],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        if r.returncode != 0:
            return r.returncode

    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
