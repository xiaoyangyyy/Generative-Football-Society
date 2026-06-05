#!/usr/bin/env python3
"""Run M0 / M1 / T0 calibration pipelines and write contract reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "calibration_baselines"


def main() -> int:
    p = argparse.ArgumentParser(description="Run calibration pipeline baselines (M0/M1/T0).")
    p.add_argument("--pipelines", nargs="*", default=["M0", "M1", "T0"])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="One fixture, one sample per pipeline")
    p.add_argument("--profile", default=None, help="Calibration profile name")
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    profile = load_profile(args.profile)
    fixtures = [("Mexico", "South Korea")] if args.quick else DEFAULT_FIXTURES
    samples = 1 if args.quick else args.samples
    match_seconds = 1200.0 if args.quick else args.match_seconds

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "profile": profile.to_dict(),
        "pipelines": {},
        "all_pass": True,
    }

    exit_code = 0
    for name in args.pipelines:
        if name not in PIPELINE_PRESETS:
            raise SystemExit(f"Unknown pipeline: {name}")
        spec = PIPELINE_PRESETS[name]
        with ablation_context(spec) as cfg:
            rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=fixtures,
                samples=samples,
                seed_start=args.seed_start,
                match_seconds=match_seconds,
                spec=spec,
                cfg=cfg,
            )
        evaluation = evaluate_rows(rows, contract=contract, baselines=baselines)
        payload = {
            "pipeline": name,
            "description": spec.description,
            "fixtures": [r["fixture"] for r in rows],
            "samples_total": len(rows),
            "hard_metrics": evaluation["hard_metrics"],
            "soft_constraints": evaluation["soft_constraints"],
            "failed_hard": evaluation["failed_hard"],
            "failed_soft": evaluation["failed_soft"],
            "all_pass": evaluation["all_pass"],
        }
        out_path = REPORTS / f"{name}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary["pipelines"][name] = {
            "all_pass": evaluation["all_pass"],
            "failed_hard": evaluation["failed_hard"],
            "failed_soft": evaluation["failed_soft"],
            "report_path": str(out_path.relative_to(ROOT)),
        }
        if not evaluation["all_pass"]:
            summary["all_pass"] = False
            exit_code = 1

    index_path = REPORTS / "index.json"
    index_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
