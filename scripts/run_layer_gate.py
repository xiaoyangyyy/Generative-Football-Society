#!/usr/bin/env python3
"""Phase 5 — additive layer gates (M1 / C1) vs M0 baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import load_contract, load_statsbomb_baselines
from src.match_engine.calibration.layer_gate import evaluate_layer_rows

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "narrative_gates"
M0_GATE = ROOT / "reports" / "calibration_gate.json"


def main() -> int:
    p = argparse.ArgumentParser(description="M1/C1 layer gates relative to M0 (Phase 5).")
    p.add_argument("--layers", nargs="*", default=["M1", "C1"], choices=["M1", "C1"])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-m0-check", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    contract = load_contract()
    baselines = load_statsbomb_baselines()

    if not args.skip_m0_check and M0_GATE.is_file():
        m0_gate = json.loads(M0_GATE.read_text(encoding="utf-8"))
        m0_absolute_passed = bool(m0_gate.get("all_pass", False))
        if not m0_absolute_passed:
            print("M0 calibration gate has not passed — run run_calibration_gate.py first.", file=sys.stderr)
            return 1
    elif not args.skip_m0_check:
        print("No M0 gate report found — run run_calibration_gate.py first.", file=sys.stderr)
        return 1
    else:
        m0_absolute_passed = True

    fixtures = [("Mexico", "South Korea")] if args.quick else DEFAULT_FIXTURES
    samples = 1 if args.quick else args.samples
    match_seconds = 1200.0 if args.quick else args.match_seconds

    REPORTS.mkdir(parents=True, exist_ok=True)
    m0_spec = PIPELINE_PRESETS["M0"]
    with ablation_context(m0_spec) as cfg:
        m0_rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=samples,
            match_seconds=match_seconds,
            spec=m0_spec,
            cfg=cfg,
        )

    index: dict = {"reference": "M0", "layers": {}, "all_pass": True}
    exit_code = 0

    for layer in args.layers:
        spec = PIPELINE_PRESETS[layer]
        with ablation_context(spec) as cfg:
            layer_rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=fixtures,
                samples=samples,
                match_seconds=match_seconds,
                spec=spec,
                cfg=cfg,
            )
        report = evaluate_layer_rows(
            m0_rows,
            layer_rows,
            layer=layer,
            contract=contract,
            baselines=baselines,
            m0_absolute_passed=m0_absolute_passed,
        )
        out_path = REPORTS / f"{layer}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        index["layers"][layer] = {
            "all_pass": report["all_pass"],
            "delta_loss": report["delta_loss"],
            "score_ratio_vs_m0": report["score_ratio_vs_m0"],
            "violations": report["violations"],
            "report": str(out_path.relative_to(ROOT)),
        }
        if not report["all_pass"]:
            index["all_pass"] = False
            exit_code = 1
        print(json.dumps(report, indent=2))

    index_path = REPORTS / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    if index["all_pass"]:
        print("\nLAYER GATES PASSED — M1/C1 additive layers OK vs M0.", file=sys.stderr)
    else:
        print("\nLAYER GATES FAILED — see reports/narrative_gates/", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
