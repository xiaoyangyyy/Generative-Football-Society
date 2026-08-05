#!/usr/bin/env python3
"""Run one formal matched-seed ablation variant against frozen M0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sim_vs_open_data import _bootstrap_mean_intervals
from src.match_engine.calibration.ablation import (
    ADDITIVE_LAYERS,
    SUBTRACTIVE_ABLATIONS,
    ablation_context,
)
from src.match_engine.calibration.benchmark_core import (
    DEFAULT_FIXTURES,
    run_micro_benchmark_rows,
)
from src.match_engine.calibration.contract import (
    evaluate_rows,
    load_contract,
    load_statsbomb_baselines,
)
from src.match_engine.calibration.objective import calibration_loss, calibration_score

VARIANTS = {**SUBTRACTIVE_ABLATIONS, **ADDITIVE_LAYERS}


def _evaluation_from_frozen(report: dict) -> dict:
    return {
        "hard_metrics": report["report"],
        "soft_constraints": report["soft_constraints"],
        "failed_hard": report["failed_metrics"],
        "failed_soft": report["failed_soft"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = VARIANTS[args.variant]
    with ablation_context(spec) as cfg:
        rows = run_micro_benchmark_rows(
            root=str(ROOT), fixtures=DEFAULT_FIXTURES, samples=3,
            seed_start=42, match_seconds=5400.0, spec=spec, cfg=cfg,
        )
    evaluation = evaluate_rows(
        rows, contract=load_contract(), baselines=load_statsbomb_baselines(),
    )
    frozen = json.loads(
        (ROOT / "data/evaluation/m0_formal_baseline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_evaluation = _evaluation_from_frozen(frozen)
    baseline_loss = calibration_loss(baseline_evaluation)
    loss = calibration_loss(evaluation)
    payload = {
        "schema_version": 1,
        "protocol": "six_fixtures_three_seeds_full_90min_matched_to_m0",
        "variant": args.variant,
        "description": spec.description,
        "samples_total": len(rows),
        "loss": loss,
        "score": calibration_score(evaluation),
        "baseline_loss": baseline_loss,
        "delta_loss": loss - baseline_loss,
        "all_pass": evaluation["all_pass"],
        "failed_metrics": evaluation["failed_hard"],
        "failed_soft": evaluation["failed_soft"],
        "uncertainty": {
            "method": "deterministic_match_bootstrap",
            "draws": 2000,
            "seed": 42,
            "metrics": _bootstrap_mean_intervals(rows, seed=42, draws=2000),
        },
        "report": evaluation["hard_metrics"],
        "soft_constraints": evaluation["soft_constraints"],
        "raw_rows": rows,
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "variant": args.variant,
        "samples_total": len(rows),
        "all_pass": evaluation["all_pass"],
        "delta_loss": payload["delta_loss"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
