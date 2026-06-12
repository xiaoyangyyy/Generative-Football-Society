#!/usr/bin/env python3
"""Narrative anomaly gate — low_block stress fixtures + WM stack (M1)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "narrative_gate.json"

# Worst-case pairs from full WC2026 run (low_block vs low_block anomalies).
STRESS_FIXTURES = [
    ("Canada", "Bosnia and Herzegovina"),
    ("Bosnia and Herzegovina", "Qatar"),
    ("Brazil", "Morocco"),
    ("United States", "Australia"),
    ("Argentina", "Austria"),
    ("France", "Iraq"),
    ("Mexico", "Czech Republic"),
    ("Netherlands", "Sweden"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Narrative anomaly gate (WM + low_block stress).")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--quick", action="store_true", help="1 sample, 20 min, 2 fixtures")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
    from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
    from src.match_engine.calibration.narrative_gate import evaluate_narrative_rows
    from src.match_engine.tactical_catalog import TACTICAL_PRESETS

    fixtures = STRESS_FIXTURES[:2] if args.quick else STRESS_FIXTURES
    samples = 1 if args.quick else args.samples
    match_seconds = 1200.0 if args.quick else args.match_seconds
    tac = dict(TACTICAL_PRESETS["low_block"])

    spec = PIPELINE_PRESETS["M1"]
    with ablation_context(spec) as cfg:
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=samples,
            match_seconds=match_seconds,
            spec=spec,
            cfg=cfg,
            seed_start=42,
            tactical_override_home=tac,
            tactical_override_away=tac,
        )

    if not rows:
        print("No rows simulated.", file=sys.stderr)
        return 2

    evaluation = evaluate_narrative_rows(rows)
    out = {
        "gate": "narrative_anomaly",
        "pipeline": "M1",
        "tactical_preset": "low_block",
        "fixtures": [f"{h}_vs_{a}" for h, a in fixtures],
        "samples_per_fixture": samples,
        "match_seconds": match_seconds,
        **evaluation,
    }
    report_path = Path(args.report) if args.report else REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if evaluation["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
