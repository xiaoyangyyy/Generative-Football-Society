#!/usr/bin/env python3
"""Full calibration gate — must pass before Phase 3+ work."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "calibration_gate.json"


def main() -> int:
    p = argparse.ArgumentParser(description="Full calibration gate (hard + soft). Exit 0 only if all pass.")
    p.add_argument("--pipeline", default="M0", choices=["M0", "M1", "C1", "T0"])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--fixtures", nargs="*", default=None, help="Optional HOME:AWAY list")
    p.add_argument("--enable-phi-gradient-move", action="store_true", help="Enable E1 phi→kinematics for this gate run")
    p.add_argument("--report", default=None, help="Output JSON path (default reports/calibration_gate.json)")
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    fixtures = DEFAULT_FIXTURES
    if args.fixtures:
        fixtures = []
        for token in args.fixtures:
            if ":" in token:
                h, a = token.split(":", 1)
                fixtures.append((h.strip(), a.strip()))

    spec = PIPELINE_PRESETS[args.pipeline]
    report_path = Path(args.report) if args.report else REPORT
    if args.enable_phi_gradient_move and args.report is None:
        report_path = ROOT / "reports" / "calibration_gate_e1.json"
    with ablation_context(spec) as cfg:
        if args.enable_phi_gradient_move:
            cfg.enable_phi_gradient_move = True
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=args.samples,
            match_seconds=args.match_seconds,
            spec=spec,
            cfg=cfg,
        )

    if not rows:
        print("No rows simulated.", file=sys.stderr)
        return 2

    evaluation = evaluate_rows(rows, contract=contract, baselines=baselines)
    out = {
        "gate": "calibration_full",
        "pipeline": args.pipeline,
        "fixtures": [f"{h}_vs_{a}" for h, a in fixtures],
        "samples_total": len(rows),
        "match_seconds": args.match_seconds,
        "enable_phi_gradient_move": bool(args.enable_phi_gradient_move),
        "phi_move_scale": cfg.phi_move_scale,
        "w_phi_move": cfg.w_phi_move,
        "hard_metrics": evaluation["hard_metrics"],
        "soft_constraints": evaluation["soft_constraints"],
        "failed_hard": evaluation["failed_hard"],
        "failed_soft": evaluation["failed_soft"],
        "all_pass": evaluation["all_pass"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    if out["all_pass"]:
        print("\nGATE PASSED — safe to proceed to next phase.", file=sys.stderr)
        return 0
    print("\nGATE FAILED — fix metrics before proceeding.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
